# 🔐 Security Guide

VecForge enterprise security: encryption, RBAC, audit logging, and namespace isolation.

Built by Suneel Bose K · ArcGX TechLabs Private Limited.

---

## Design Philosophy

Security is **never optional** in VecForge. Every storage decision considers encryption. Every multi-user design considers RBAC. Every log is audit-capable.

---

## AES-256 Encryption at Rest

Encrypt your vault with SQLCipher AES-256:

```python
import os
from vecforge import VecForge

# Always use environment variables — never hardcode keys
db = VecForge(
    "secure_vault.db",
    encryption_key=os.environ["VECFORGE_KEY"],
)

db.add("Top secret patient data")
db.close()

# The .db file is unreadable without the key
raw = open("secure_vault.db", "rb").read()
assert b"Top secret" not in raw  # Encrypted!
```

### Requirements
- Install `sqlcipher3`: `pip install sqlcipher3`
- SQLCipher C library must be available on the system
- Without SQLCipher, VecForge falls back to standard SQLite with a warning

### Key Management Best Practices
- Store keys in environment variables: `VECFORGE_KEY`
- Use a key length of at least 8 characters
- Rotate keys by creating a new vault and migrating data
- Lost keys = lost data — there is no recovery

---

## Role-Based Access Control (RBAC)

VecForge supports three roles:

| Role | Read | Write | Delete | Namespace | Manage Keys | Backup |
|---|---|---|---|---|---|---|
| `admin` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| `read-write` | ✅ | ✅ | ✅ | ❌ | ❌ | ❌ |
| `read-only` | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ |

### Usage

```python
# No API key = local admin (full access)
db = VecForge("vault")

# API key with read-only access
db = VecForge("vault", api_key="read-only-key-abc")
db.search("query")     # ✅ Works
db.add("new doc")      # ❌ Raises VecForgePermissionError

# Permission check in your code
from vecforge.security.rbac import RBACManager

rbac = RBACManager(api_key="key123", key_roles={"key123": "read-only"})
rbac.require("read")   # ✅ OK
rbac.require("write")  # ❌ Raises VecForgePermissionError
```

---

## Audit Logging

Track every mutating operation with append-only JSONL logs:

```python
db = VecForge(
    "vault",
    audit_log="audit.jsonl",
)

db.add("sensitive document")
db.search("query")
db.delete(doc_id)
```

### Audit Log Format

Each line in `audit.jsonl`:
```json
{
    "timestamp": 1709654400.0,
    "actor": "local-admin",
    "operation": "add",
    "doc_id": "a1b2c3d4-...",
    "namespace": "default",
    "metadata": {"chars": 1500}
}
```

### Reading Audit Logs

```python
from vecforge.security.audit import AuditLogger

audit = AuditLogger("audit.jsonl")

# Read all events
events = audit.read_log()

# Filter by time range
events = audit.read_log(since=1709654000.0, until=1709655000.0)

# Filter by actor or operation
events = audit.read_log(actor="local-admin", operation="delete")
```

---

## Namespace Isolation

Namespaces provide hard multi-tenant boundaries:

```python
db = VecForge("multi_tenant.db")

db.create_namespace("tenant_a")
db.create_namespace("tenant_b")

# Data is isolated
db.add("Tenant A secret", namespace="tenant_a")
db.add("Tenant B secret", namespace="tenant_b")

# Tenant A search never returns Tenant B data
results = db.search("secret", namespace="tenant_a")
assert all(r.namespace == "tenant_a" for r in results)
```

### SQL-Level Isolation
Every SQL query in VecForge includes `WHERE namespace = ?` — this is enforced at the storage layer, not the application layer.

---

## Backup & Restore

```python
from vecforge.security.snapshots import SnapshotManager

snap = SnapshotManager("my_vault.db")

# Create timestamped backup
backup_path = snap.create_snapshot("backups/")
# → "backups/my_vault_20260305_112500.db"

# List available backups
snapshots = snap.list_snapshots("backups/")

# Restore from backup
snap.restore_snapshot(backup_path)
```

### Encryption Preserved
Snapshots preserve the encryption state — encrypted vaults remain encrypted in backups.

---

## Deletion Protection

Prevent accidental data loss:

```python
db = VecForge("vault", deletion_protection=True)

db.add("important doc")
db.delete(doc_id)  # ❌ Raises DeletionProtectedError
```
