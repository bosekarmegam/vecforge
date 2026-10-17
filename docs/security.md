# 🔐 Security Guide

VecForge enterprise security: encryption, RBAC, audit logging, namespace
isolation, backups.

Built by Suneel Bose K · ArcGX TechLabs Private Limited.

---

## Design Philosophy

Local-first means your data never leaves the machine unless you ship it.
Every storage decision considers encryption; every multi-user path checks a
role; every mutation can be audited.

---

## AES-256 Encryption at Rest

```python
import os
from vecforge import VecForge

db = VecForge("secure_vault.db", encryption_key=os.environ["VECFORGE_KEY"])
db.add("Top secret patient data")
db.close()
```

Requirements: `pip install "vecforge[encryption]"` (or `sqlcipher3` with the
system library). Without SQLCipher, VecForge falls back to plain SQLite and
warns. Keys are escaped before being passed to SQLCipher, so any string is a
valid key; a wrong key fails at open time.

Index snapshots (FAISS vectors and the BM25 inverted index) are stored inside
the same encrypted database, in chunks, so nothing sensitive is written to a
sidecar file.

**Key management:** keep keys in environment variables, rotate by creating a
new vault and migrating, and remember that a lost key means lost data.

---

## Role-Based Access Control (RBAC)

| Role | Read | Write | Delete | Namespace | Manage Keys | Backup |
|---|---|---|---|---|---|---|
| `admin` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| `read-write` | ✅ | ✅ | ✅ | ❌ | ❌ | ❌ |
| `read-only` | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ |
| `none` (unregistered key) | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |

Keys are registered by an admin and persisted in the vault as SHA-256
digests — raw keys are never stored:

```python
admin = VecForge("vault.db")                       # no key = local admin
admin.register_api_key(os.environ["TEAM_KEY"], "read-only")
admin.close()

reader = VecForge("vault.db", api_key=os.environ["TEAM_KEY"])
reader.search("query")      # ✅
reader.add("new doc")       # ❌ VecForgePermissionError

stranger = VecForge("vault.db", api_key="guess")
stranger.search("query")    # ❌ unregistered keys get nothing (1.x gave read-only)
```

`admin.revoke_api_key(key)` and `admin.list_api_keys()` complete the set.

### REST server

`vecforge serve --api-key …` (or `$VECFORGE_API_KEY`) requires the
`X-API-Key` header on every request and binds `127.0.0.1` by default.

---

## Audit Logging

```python
db = VecForge("vault", audit_log="audit.jsonl")
db.add("sensitive document")
db.search("query")
db.delete(doc_id)
```

Each line in `audit.jsonl`:

```json
{"timestamp": 1709654400.0, "actor": "local-admin", "operation": "add",
 "doc_id": "a1b2c3d4-...", "namespace": "default", "metadata": {"chars": 1500}}
```

Search events record the query text; store the log with the same care as
the vault. Read it back with `AuditLogger("audit.jsonl").read_log(since=…,
actor=…, operation=…)`.

---

## Namespace Isolation

```python
db.create_namespace("tenant_a")
db.create_namespace("tenant_b")
db.add("Tenant A secret", namespace="tenant_a")
db.add("Tenant B secret", namespace="tenant_b")

results = db.search("secret", namespace="tenant_a")
assert all(r.namespace == "tenant_a" for r in results)
```

Isolation is enforced at three levels: SQL (`WHERE namespace = ?`), the
FAISS/BM25 retrievers (ID selectors restrict the search to the namespace's
positions), and a final check during hydration. The AI memory layer uses
one namespace per agent.

---

## Backup & Restore

```python
db.backup("backups/vault-2026-10-17.db")     # consistent copy while open

from vecforge.security.snapshots import SnapshotManager
snap = SnapshotManager("my_vault.db")
path = snap.create_snapshot("backups/")      # SQLite online backup, WAL-safe
snap.list_snapshots("backups/")
snap.restore_snapshot(path)                  # removes stale -wal/-shm journals
```

Encrypted vaults stay encrypted in backups. For off-site copies see
`CloudSync` in the [Multimodal Guide](multimodal.md#opt-in-cloud-backup) —
it only accepts files that are already encrypted.

---

## Deletion Protection

```python
db = VecForge("vault", deletion_protection=True)
db.delete(doc_id)            # ❌ DeletionProtectedError
db.upsert("v2", doc_id=doc_id)   # ❌ also blocked (it would replace data)
```
