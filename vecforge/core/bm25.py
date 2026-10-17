# VecForge — Universal Local-First Vector Database
# Copyright (c) 2026 Suneel Bose K · ArcGX TechLabs Private Limited
# Built by Suneel Bose K (Founder & CEO, ArcGX TechLabs)
#
# Licensed under the Business Source License 1.1 (BSL 1.1)
# Free for personal, research, open-source, and non-commercial use.
# Commercial use requires a separate license from ArcGX TechLabs.
# See LICENSE file in the project root or contact: suneelbose@arcgx.in

"""
BM25 keyword search engine for VecForge.

VecForge 2.0 ships its own incremental inverted-index BM25 implementation.
The 1.x engine wrapped ``rank_bm25.BM25Okapi``, which has two structural
problems for a database that grows one document at a time:

* every ``add`` rebuilt the whole index — O(N·L) per insert, so ingesting
  N documents cost O(N²·L) in total;
* every query scored **every** document for **every** query term in a
  Python loop — O(V·N) per search, ~0.5 s per term at one million docs.

The new engine keeps a growable posting list per term, so an insert costs
O(L) (L = tokens in the document) and a query costs O(Σ postings of the
query terms), which is what an inverted index is for. Deletions are
tombstones (O(1)) and the whole index serialises to a compact, pickle-free
NumPy byte format so a reopened vault does not re-tokenise its corpus.

Scoring uses the Lucene/“BM25+” style IDF, ``log(1 + (N - df + 0.5) /
(df + 0.5))``, which is always positive, instead of the classic Okapi IDF
that goes negative for terms appearing in more than half the corpus.

Built by Suneel Bose K · ArcGX TechLabs Private Limited.
"""

from __future__ import annotations

import io
import json
import logging
import math
import re
from collections import Counter
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)

_TOKEN_RE = re.compile(r"\b\w+\b")
_MAGIC = b"VFBM25\x00\x02"
_INITIAL_CAPACITY = 4


@dataclass
class BM25Result:
    """A single BM25 search result.

    Attributes:
        doc_index: Index of the document in the corpus (insertion position).
        score: BM25 relevance score (higher = more relevant).
    """

    doc_index: int
    score: float


class _PostingList:
    """Growable (doc_index, term_frequency) arrays for one term.

    Appends are amortised O(1) via capacity doubling; searches read the
    live prefix ``ids[:n]`` without copying.
    """

    __slots__ = ("ids", "tfs", "n")

    def __init__(self, capacity: int = _INITIAL_CAPACITY) -> None:
        self.ids: NDArray[np.int64] = np.empty(capacity, dtype=np.int64)
        self.tfs: NDArray[np.float32] = np.empty(capacity, dtype=np.float32)
        self.n = 0

    def append(self, doc_index: int, tf: int) -> None:
        if self.n == len(self.ids):
            new_cap = max(_INITIAL_CAPACITY, len(self.ids) * 2)
            self.ids = np.resize(self.ids, new_cap)
            self.tfs = np.resize(self.tfs, new_cap)
        self.ids[self.n] = doc_index
        self.tfs[self.n] = tf
        self.n += 1

    def extend(self, ids: NDArray[np.int64], tfs: NDArray[np.float32]) -> None:
        needed = self.n + len(ids)
        if needed > len(self.ids):
            new_cap = max(needed, len(self.ids) * 2, _INITIAL_CAPACITY)
            self.ids = np.resize(self.ids, new_cap)
            self.tfs = np.resize(self.tfs, new_cap)
        self.ids[self.n : needed] = ids
        self.tfs[self.n : needed] = tfs
        self.n = needed

    def view(self) -> tuple[NDArray[np.int64], NDArray[np.float32]]:
        return self.ids[: self.n], self.tfs[: self.n]


class BM25Engine:
    """Incremental inverted-index BM25 keyword search engine.

    Maintains per-term posting lists that grow in place, so documents can
    be added one at a time without ever rebuilding the index. Deleted
    documents are tombstoned and excluded from results; call
    :meth:`compact_positions` (via ``VecForge.compact()``) to rebuild
    without them.

    Built by Suneel Bose K · ArcGX TechLabs Private Limited.

    Args:
        k1: Term-frequency saturation parameter. Defaults to 1.5.
        b: Document-length normalisation (0 = none, 1 = full). Defaults
            to 0.75.

    Performance:
        Add:    O(L) per document, L = tokens (no rebuild)
        Search: O(Σ df(term)) over query terms + O(N) top-k selection
        Delete: O(1) tombstone
        Typical: <1ms search at 100k docs for ordinary queries

    Example:
        >>> engine = BM25Engine()
        >>> engine.add_documents(["patient with diabetes", "hip fracture case"])
        >>> results = engine.search("diabetes", top_k=1)
        >>> results[0].doc_index
        0
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self._k1 = float(k1)
        self._b = float(b)
        self._postings: dict[str, _PostingList] = {}
        self._doc_len: list[int] = []
        self._doc_len_arr: NDArray[np.float32] | None = None  # lazy cache
        self._total_len = 0
        self._deleted: set[int] = set()

    # ─── Properties ───

    @property
    def count(self) -> int:
        """Return number of live (non-deleted) documents.

        Performance:
            Time: O(1)
        """
        return len(self._doc_len) - len(self._deleted)

    @property
    def size(self) -> int:
        """Return number of positions ever added, including tombstones.

        Performance:
            Time: O(1)
        """
        return len(self._doc_len)

    @property
    def vocabulary_size(self) -> int:
        """Return number of distinct terms in the index.

        Performance:
            Time: O(1)
        """
        return len(self._postings)

    @property
    def deleted_count(self) -> int:
        """Return number of tombstoned documents.

        Performance:
            Time: O(1)
        """
        return len(self._deleted)

    # ─── Tokenisation ───

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """Tokenize text into lowercase word tokens.

        Simple regex tokenizer — BM25 benefits more from exact matching
        than from linguistic analysis, and it keeps the index language-
        agnostic.

        Args:
            text: Raw text string to tokenize.

        Returns:
            List of lowercase word tokens.

        Performance:
            Time: O(L) where L = length of text
        """
        return _TOKEN_RE.findall(text.lower())

    # ─── Mutation ───

    def add_documents(self, texts: list[str]) -> list[int]:
        """Add documents to the index incrementally.

        Args:
            texts: List of document texts to add.

        Returns:
            List of positions (doc_index values) assigned to the texts.

        Performance:
            Time: O(total tokens) — no index rebuild

        Example:
            >>> engine = BM25Engine()
            >>> engine.add_documents(["doc one", "doc two", "doc three"])
            [0, 1, 2]
            >>> engine.count
            3
        """
        if not texts:
            return []

        start = len(self._doc_len)
        batch_terms: dict[str, tuple[list[int], list[int]]] = {}

        for offset, text in enumerate(texts):
            tokens = self._tokenize(text)
            self._doc_len.append(len(tokens))
            self._total_len += len(tokens)
            doc_index = start + offset
            for term, tf in Counter(tokens).items():
                bucket = batch_terms.get(term)
                if bucket is None:
                    bucket = ([], [])
                    batch_terms[term] = bucket
                bucket[0].append(doc_index)
                bucket[1].append(tf)

        for term, (ids, tfs) in batch_terms.items():
            plist = self._postings.get(term)
            if plist is None:
                plist = _PostingList(capacity=max(_INITIAL_CAPACITY, len(ids)))
                self._postings[term] = plist
            if len(ids) == 1:
                plist.append(ids[0], tfs[0])
            else:
                plist.extend(
                    np.asarray(ids, dtype=np.int64),
                    np.asarray(tfs, dtype=np.float32),
                )

        self._doc_len_arr = None
        logger.debug(
            "BM25 index: +%d docs (total=%d, vocab=%d)",
            len(texts),
            len(self._doc_len),
            len(self._postings),
        )
        return list(range(start, start + len(texts)))

    def add_document(self, text: str) -> int:
        """Add a single document to the index.

        Args:
            text: Document text to add.

        Returns:
            Position (doc_index) assigned to the document.

        Performance:
            Time: O(L) — incremental, no rebuild
        """
        return self.add_documents([text])[0]

    def remove(self, doc_index: int) -> bool:
        """Tombstone a document so it no longer appears in results.

        Args:
            doc_index: Position returned by :meth:`add_documents`.

        Returns:
            True if the position existed and was not already deleted.

        Performance:
            Time: O(1)
        """
        if 0 <= doc_index < len(self._doc_len) and doc_index not in self._deleted:
            self._deleted.add(doc_index)
            return True
        return False

    def reset(self) -> None:
        """Reset the index, removing all documents.

        Performance:
            Time: O(1)
        """
        self._postings = {}
        self._doc_len = []
        self._doc_len_arr = None
        self._total_len = 0
        self._deleted = set()

    # ─── Search ───

    def _idf(self, df: int) -> float:
        n = len(self._doc_len)
        return math.log(1.0 + (n - df + 0.5) / (df + 0.5))

    def _doc_lengths(self) -> NDArray[np.float32]:
        if self._doc_len_arr is None or len(self._doc_len_arr) != len(self._doc_len):
            self._doc_len_arr = np.asarray(self._doc_len, dtype=np.float32)
        return self._doc_len_arr

    def get_scores(self, query: str) -> NDArray[np.float32]:
        """Score every position for a query (0.0 where no term matches).

        Args:
            query: Search query string.

        Returns:
            float32 array of length :attr:`size`; tombstoned positions are 0.

        Performance:
            Time: O(Σ df(term)) + O(size) allocation
        """
        size = len(self._doc_len)
        scores = np.zeros(size, dtype=np.float32)
        if size == 0:
            return scores

        query_tokens = self._tokenize(query)
        if not query_tokens:
            return scores

        avgdl = self._total_len / size if size else 1.0
        if avgdl <= 0.0:
            avgdl = 1.0
        doc_len = self._doc_lengths()
        k1, b = self._k1, self._b

        for term, qtf in Counter(query_tokens).items():
            plist = self._postings.get(term)
            if plist is None or plist.n == 0:
                continue
            ids, tfs = plist.view()
            idf = self._idf(plist.n) * qtf
            denom = tfs + k1 * (1.0 - b + b * doc_len[ids] / avgdl)
            contrib = idf * tfs * (k1 + 1.0) / denom
            np.add.at(scores, ids, contrib.astype(np.float32))

        if self._deleted:
            scores[np.fromiter(self._deleted, dtype=np.int64)] = 0.0
        return scores

    def search(
        self,
        query: str,
        top_k: int = 10,
        allowed: NDArray[np.int64] | None = None,
    ) -> list[BM25Result]:
        """Search for documents matching the query keywords.

        Args:
            query: Search query string.
            top_k: Number of top results to return.
            allowed: Optional array of positions to restrict the search to
                (e.g. the positions belonging to one namespace). Positions
                outside this set are never returned.

        Returns:
            List of BM25Result sorted by descending score. Only documents
            with a positive score are returned, so the list may be shorter
            than ``top_k``.

        Performance:
            Time: O(Σ df(term)) + O(size) for top-k selection
            Typical: <1ms at 100k docs

        Example:
            >>> results = engine.search("diabetes treatment", top_k=5)
            >>> for r in results:
            ...     print(f"Doc {r.doc_index}: score={r.score:.4f}")
        """
        if len(self._doc_len) == 0 or top_k <= 0:
            return []

        scores = self.get_scores(query)

        if allowed is not None:
            allowed_arr = np.asarray(allowed, dtype=np.int64)
            if allowed_arr.size == 0:
                return []
            sub_scores = scores[allowed_arr]
            effective_k = min(top_k, sub_scores.size)
            local_top = np.argpartition(sub_scores, -effective_k)[-effective_k:]
            local_top = local_top[np.argsort(sub_scores[local_top])[::-1]]
            return [
                BM25Result(
                    doc_index=int(allowed_arr[i]),
                    score=float(sub_scores[i]),
                )
                for i in local_top
                if sub_scores[i] > 0.0
            ]

        # perf: only positions with a matching term can score > 0
        matched = np.flatnonzero(scores)
        if matched.size == 0:
            return []
        effective_k = min(top_k, matched.size)
        matched_scores = scores[matched]
        if effective_k < matched.size:
            part = np.argpartition(matched_scores, -effective_k)[-effective_k:]
        else:
            part = np.arange(matched.size)
        order = part[np.argsort(matched_scores[part])[::-1]]
        return [
            BM25Result(doc_index=int(matched[i]), score=float(matched_scores[i]))
            for i in order
        ]

    # ─── Persistence ───

    def to_bytes(self) -> bytes:
        """Serialise the index to a compact, pickle-free byte string.

        Returns:
            Bytes that :meth:`from_bytes` can restore. Uses NumPy arrays
            and a JSON header only — safe to load from an untrusted file.

        Performance:
            Time: O(total postings)
        """
        terms = list(self._postings.keys())
        counts = np.fromiter(
            (self._postings[t].n for t in terms), dtype=np.int64, count=len(terms)
        )
        total = int(counts.sum()) if len(counts) else 0
        ids = np.empty(total, dtype=np.int64)
        tfs = np.empty(total, dtype=np.float32)
        pos = 0
        for t in terms:
            v_ids, v_tfs = self._postings[t].view()
            ids[pos : pos + len(v_ids)] = v_ids
            tfs[pos : pos + len(v_tfs)] = v_tfs
            pos += len(v_ids)

        vocab_blob = np.frombuffer("\n".join(terms).encode("utf-8"), dtype=np.uint8)
        header = {
            "k1": self._k1,
            "b": self._b,
            "total_len": self._total_len,
            "n_terms": len(terms),
        }
        buf = io.BytesIO()
        buf.write(_MAGIC)
        np.savez(
            buf,
            header=np.frombuffer(json.dumps(header).encode("utf-8"), dtype=np.uint8),
            vocab=vocab_blob,
            counts=counts,
            ids=ids,
            tfs=tfs,
            doc_len=np.asarray(self._doc_len, dtype=np.int64),
            deleted=np.fromiter(sorted(self._deleted), dtype=np.int64),
        )
        return buf.getvalue()

    @classmethod
    def from_bytes(cls, data: bytes) -> BM25Engine:
        """Restore an index serialised by :meth:`to_bytes`.

        Args:
            data: Bytes produced by :meth:`to_bytes`.

        Returns:
            A fully populated BM25Engine.

        Raises:
            ValueError: If the bytes are not a VecForge BM25 blob.

        Performance:
            Time: O(total postings)
        """
        if not data.startswith(_MAGIC):
            raise ValueError(
                "Not a VecForge BM25 index blob.\n"
                "VecForge by Suneel Bose K · ArcGX TechLabs"
            )
        with np.load(io.BytesIO(data[len(_MAGIC) :]), allow_pickle=False) as npz:
            header = json.loads(npz["header"].tobytes().decode("utf-8"))
            vocab_bytes = npz["vocab"].tobytes().decode("utf-8")
            counts = npz["counts"]
            ids = npz["ids"]
            tfs = npz["tfs"]
            doc_len = npz["doc_len"]
            deleted = npz["deleted"]

        engine = cls(k1=float(header["k1"]), b=float(header["b"]))
        terms = vocab_bytes.split("\n") if header["n_terms"] else []
        pos = 0
        for term, n in zip(terms, counts.tolist(), strict=True):
            plist = _PostingList(capacity=max(_INITIAL_CAPACITY, n))
            plist.extend(ids[pos : pos + n], tfs[pos : pos + n])
            pos += n
            engine._postings[term] = plist
        engine._doc_len = doc_len.astype(np.int64).tolist()
        engine._total_len = int(header["total_len"])
        engine._deleted = {int(d) for d in deleted.tolist()}
        return engine
