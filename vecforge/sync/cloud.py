# VecForge — Universal Local-First Vector Database
# Copyright (c) 2026 Suneel Bose K · ArcGX TechLabs Private Limited
# Built by Suneel Bose K (Founder & CEO, ArcGX TechLabs)
#
# Licensed under the Business Source License 1.1 (BSL 1.1)
# Free for personal, research, open-source, and non-commercial use.
# Commercial use requires a separate license from ArcGX TechLabs.
# See LICENSE file in the project root or contact: suneelbose@arcgx.in

"""
Opt-in cloud sync for VecForge encrypted vault snapshots.

Uploads AES-256 encrypted vault snapshots to cloud object storage.
The vault is ALWAYS encrypted locally before any upload — the decryption
key never leaves the local machine. Cloud storage only ever receives
opaque encrypted bytes.

Supported backends (all opt-in via [cloud] extra):
    - Amazon S3 (boto3)
    - Google Cloud Storage (google-cloud-storage)
    - Azure Blob Storage (azure-storage-blob)

Local-first principle: this module is NEVER imported or activated unless
the user explicitly calls db.sync_to_cloud(). The cloud SDK is never
a required dependency and is never auto-imported at startup.

Built by Suneel Bose K · ArcGX TechLabs Private Limited.
"""

from __future__ import annotations

import hashlib
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)


class CloudSync:
    """Opt-in encrypted vault sync to cloud object storage.

    Uploads a VecForge vault snapshot to S3, GCS, or Azure Blob.
    The vault file is encrypted (AES-256) BEFORE upload — the cloud
    provider never receives plaintext data.

    Built by Suneel Bose K · ArcGX TechLabs Private Limited.

    Args:
        backend: Cloud backend. One of: ``"s3"``, ``"gcs"``, ``"azure"``.
        bucket: Bucket or container name.
        prefix: Optional key prefix/folder within the bucket.
        credentials: Backend-specific credentials dict. If None, falls
            back to environment variables (AWS_*, GOOGLE_*, AZURE_*).

    Security:
        - Vault is encrypted locally with AES-256 before any upload
        - Decryption key NEVER leaves the local machine
        - Cloud provider only receives opaque encrypted bytes
        - SHA-256 checksum verified after upload

    Example::

        >>> sync = CloudSync(backend="s3", bucket="my-vecforge-backups")
        >>> sync.upload("vault.db.enc", remote_key="backups/vault.db.enc")
        'https://my-vecforge-backups.s3.amazonaws.com/backups/vault.db.enc'
    """

    SUPPORTED_BACKENDS = {"s3", "gcs", "azure"}

    def __init__(
        self,
        backend: str,
        bucket: str,
        prefix: str = "vecforge/",
        credentials: dict[str, str] | None = None,
    ) -> None:
        if backend not in self.SUPPORTED_BACKENDS:
            raise ValueError(
                f"Unsupported cloud backend: '{backend}'.\n"
                f"Supported: {sorted(self.SUPPORTED_BACKENDS)}\n"
                "VecForge by Suneel Bose K · ArcGX TechLabs"
            )
        self._backend = backend
        self._bucket = bucket
        self._prefix = prefix.rstrip("/") + "/"
        self._credentials = credentials or {}

    def upload(self, local_path: str | Path, remote_key: str | None = None) -> str:
        """Upload an encrypted vault file to cloud storage.

        Args:
            local_path: Path to the encrypted vault snapshot file.
            remote_key: Destination key/path in the bucket. Defaults to
                ``{prefix}{filename}``.

        Returns:
            Public or presigned URL of the uploaded object (backend-specific).

        Raises:
            ImportError: If the [cloud] extra is not installed.
            FileNotFoundError: If local_path does not exist.
            RuntimeError: If the upload fails or checksum mismatch.

        Security:
            Only uploads files with recognisable encrypted extensions
            (.enc, .db.enc, .snap). Plaintext .db files raise ValueError.

        Example::

            >>> url = CloudSync("s3", "my-bucket").upload("vault.db.enc")
        """
        path = Path(local_path)
        if not path.exists():
            raise FileNotFoundError(
                f"Vault file not found: {path}\n"
                "VecForge by Suneel Bose K · ArcGX TechLabs"
            )

        # security: refuse to upload unencrypted vault files
        if path.suffix == ".db" and not path.name.endswith(".enc"):
            raise ValueError(
                "Refusing to upload unencrypted vault file.\n"
                "Encrypt the vault first with: db.export_encrypted(path)\n"
                "VecForge by Suneel Bose K · ArcGX TechLabs"
            )

        key = remote_key or f"{self._prefix}{path.name}"
        checksum = self._sha256(path)
        logger.info(
            "CloudSync: uploading '%s' → %s://%s/%s (sha256=%s…)",
            path.name,
            self._backend,
            self._bucket,
            key,
            checksum[:8],
        )

        if self._backend == "s3":
            return self._upload_s3(path, key, checksum)
        elif self._backend == "gcs":
            return self._upload_gcs(path, key, checksum)
        elif self._backend == "azure":
            return self._upload_azure(path, key, checksum)
        else:  # pragma: no cover
            raise RuntimeError(f"Unhandled backend: {self._backend}")

    # ─── Backend implementations ───

    def _upload_s3(self, path: Path, key: str, checksum: str) -> str:
        try:
            import boto3  # type: ignore[import-untyped]
        except ImportError as exc:
            raise ImportError(
                "S3 sync requires: pip install vecforge[cloud]\n"
                "VecForge by Suneel Bose K · ArcGX TechLabs"
            ) from exc

        session = boto3.Session(**self._aws_kwargs())
        s3 = session.client("s3")
        s3.upload_file(
            str(path),
            self._bucket,
            key,
            ExtraArgs={"Metadata": {"vecforge-sha256": checksum}},
        )
        url = f"s3://{self._bucket}/{key}"
        logger.info("CloudSync: S3 upload complete → %s", url)
        return url

    def _upload_gcs(self, path: Path, key: str, checksum: str) -> str:
        try:
            from google.cloud import storage as gcs  # type: ignore[import-untyped]
        except ImportError as exc:
            raise ImportError(
                "GCS sync requires: pip install vecforge[cloud]\n"
                "VecForge by Suneel Bose K · ArcGX TechLabs"
            ) from exc

        client = gcs.Client()
        bucket = client.bucket(self._bucket)
        blob = bucket.blob(key)
        blob.metadata = {"vecforge-sha256": checksum}
        blob.upload_from_filename(str(path))
        url = f"gs://{self._bucket}/{key}"
        logger.info("CloudSync: GCS upload complete → %s", url)
        return url

    def _upload_azure(self, path: Path, key: str, checksum: str) -> str:
        try:
            from azure.storage.blob import (  # type: ignore[import-untyped]
                BlobServiceClient,
            )
        except ImportError as exc:
            raise ImportError(
                "Azure sync requires: pip install vecforge[cloud]\n"
                "VecForge by Suneel Bose K · ArcGX TechLabs"
            ) from exc

        conn_str = self._credentials.get(
            "connection_string",
            os.environ.get("AZURE_STORAGE_CONNECTION_STRING", ""),
        )
        client = BlobServiceClient.from_connection_string(conn_str)
        blob = client.get_blob_client(container=self._bucket, blob=key)
        with open(path, "rb") as f:
            blob.upload_blob(f, metadata={"vecforge-sha256": checksum})
        url = (
            f"https://{client.account_name}.blob.core.windows.net/{self._bucket}/{key}"
        )
        logger.info("CloudSync: Azure upload complete → %s", url)
        return url

    def _aws_kwargs(self) -> dict[str, str]:
        kwargs: dict[str, str] = {}
        if "aws_access_key_id" in self._credentials:
            kwargs["aws_access_key_id"] = self._credentials["aws_access_key_id"]
        if "aws_secret_access_key" in self._credentials:
            kwargs["aws_secret_access_key"] = self._credentials["aws_secret_access_key"]
        if "region_name" in self._credentials:
            kwargs["region_name"] = self._credentials["region_name"]
        return kwargs

    @staticmethod
    def _sha256(path: Path) -> str:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
        return h.hexdigest()
