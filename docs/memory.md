# 🧠 AI Memory Guide

Give Claude, GPT, or any agent a memory that survives the end of a session.

Built by Suneel Bose K · ArcGX TechLabs Private Limited.

---

## What it is

`MemoryStore` is a memory layer on top of a VecForge vault. Every memory is a
document in the namespace `memory:<agent>` with structured metadata (kind,
importance, tags, session, access statistics), so it is searchable with the
same hybrid dense + keyword engine, encrypted at rest if you want, and
restored instantly on the next open.

```python
from vecforge import MemoryStore

mem = MemoryStore("~/.vecforge/memory.db", agent="claude")

mem.remember("User prefers concise answers without em-dashes", kind="preference", importance=0.9)
mem.remember("The vecforge repo uses black and ruff", kind="fact", tags=["tooling"])
mem.remember("Decided to ship 2.0 with the memory layer", kind="decision")

for m in mem.recall("how should I format my answers?", top_k=3):
    print(f"{m.score:.2f}  [{m.kind}]  {m.content}")
```

The first call creates `~/.vecforge/memory.db`. Use `":memory:"` for a
throw-away store and `model_name="hash"` for a zero-download embedder.

---

## Memory kinds

| Kind | Use it for |
|---|---|
| `preference` | How the user likes things done |
| `decision` | Choices that were made, and why |
| `fact` | Durable facts about the user, project, or world |
| `insight` | Conclusions the assistant reached |
| `task` | Open work items |
| `summary` | Session summaries (written by `end_session`) |
| `episode` | What happened (events, conversations) |
| `note` | Anything else (the default for unknown kinds) |

`context_pack()` lists preferences and decisions first, so the things that
shape behaviour are never trimmed away.

---

## Remembering

```python
m = mem.remember(
    "Deploys happen on Fridays",
    kind="fact",
    importance=0.7,        # 0.0–1.0, used during recall
    tags=["ops", "release"],
    source="user",         # free-form origin label
    metadata={"ticket": "OPS-42"},
)
```

### Deduplication

`remember()` never stores the same thing twice. An exact duplicate, or a
near-duplicate (cosine similarity ≥ `dedup_threshold`, default 0.92) of the
same kind, **reinforces** the existing memory instead: its importance and
access count go up and the tags are merged. The returned `Memory` has
`deduplicated=True` in that case.

```python
again = mem.remember("Deploys happen on Fridays", kind="fact")
assert again.deduplicated and again.memory_id == m.memory_id
```

Pass `dedup=False` to force a new record.

---

## Recalling

```python
memories = mem.recall(
    "deployment schedule",
    top_k=5,
    kinds=["fact", "decision"],   # optional
    tags=["ops"],                 # at least one of these
    since=time.time() - 7 * 86400, # created in the last week
    min_importance=0.3,
    recency_weight=0.15,
    importance_weight=0.25,
)
```

The score blends three signals:

```
score = (1 − rw − iw) · relevance + iw · importance + rw · recency
```

`relevance` is the vault's hybrid search score, `recency` decays with a
30-day half-life from the last access, and `rw` / `iw` are the weights above.
Each recall records the access (`access_count`, `last_accessed`), so
memories that keep proving useful float up. Pass `touch=False` to look
without leaving a trace.

`recall("")` with an empty query returns the most important recent memories.

---

## Sessions

Sessions give the assistant a notion of "last time".

```python
session = mem.start_session(metadata={"project": "vecforge"})
mem.remember("Rewrote the Grover reranker", kind="episode")   # attributed to the session
mem.end_session("Fixed the quantum reranker and shipped 2.0")  # stored as a summary memory

# next conversation
mem.start_session()
print(mem.last_session().summary)
print(mem.context_pack("quantum"))
```

`context_pack()` returns a markdown briefing ready to paste into a system
prompt:

```
## Memory for agent 'claude' — relevant to: quantum

### Last session (2026-10-17 09:40)
Fixed the quantum reranker and shipped 2.0

### Preferences
- User prefers concise answers without em-dashes

### Facts
- The vecforge repo uses black and ruff [tooling]
```

It is trimmed to `max_chars` (default 4000, roughly 1000 tokens).

---

## Editing and housekeeping

```python
mem.reinforce(m.memory_id, delta=0.1)              # bump importance + access count
mem.update(m.memory_id, content="Deploys moved to Thursdays", tags=["ops"])
mem.forget(m.memory_id)
mem.forget_many([...])
mem.clear(kind="episode")                           # or clear() for everything
mem.list_memories(kinds=["task"], limit=20)         # browse without a query
mem.session_memories(session.session_id)
mem.stats()
```

Export and import keep memories portable between machines and agents:

```python
mem.export_json("claude-memory.json")
other = MemoryStore("~/.vecforge/memory.db", agent="gpt")
other.import_json("claude-memory.json")   # duplicates are skipped
```

---

## Several agents, one vault

Agents are isolated by namespace, so one encrypted vault can hold memory
for many assistants:

```python
from vecforge import VecForge, MemoryStore

vault = VecForge("~/.vecforge/memory.db", encryption_key=os.environ["VECFORGE_KEY"])
claude = MemoryStore(vault=vault, agent="claude")
coder = MemoryStore(vault=vault, agent="code-assistant")
```

A store created with `vault=` does not close the vault when it is closed.

---

## Performance

| Operation | Cost |
|---|---|
| `remember` | one embedding + one SQLite insert (+ one dedup search) |
| `recall` | one hybrid search + O(k) rescoring |
| open | index snapshot restore — milliseconds, no rebuild |

The MCP server (`vecforge mcp`) refreshes the index snapshot every 25 writes
and at shutdown, so the next session starts instantly. See the
[MCP Guide](mcp.md).
