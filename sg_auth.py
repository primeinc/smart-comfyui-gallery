# sg_auth.py — one-way password hashing for SmartGallery.
#
# Passwords are stored ONLY as Argon2id hashes. There is no supported way to
# recover a stored password from this module. The single exception is
# migrate_legacy_passwords(), which performs a one-time decrypt-then-rehash
# of pre-existing Fernet ciphertexts left over from the old reversible
# scheme; its decryption helper is module-private and is never exposed as a
# general-purpose "decrypt this password" API.

import os
import sqlite3

from argon2 import PasswordHasher
from argon2.exceptions import Argon2Error, InvalidHashError, VerificationError, VerifyMismatchError

# Library defaults (time_cost, memory_cost, parallelism, hash_len, salt_len,
# type=Argon2id) are appropriate here; no tuning required.
ph = PasswordHasher()

# Sentinel stored for accounts whose password cannot be used to log in
# (e.g. undecryptable legacy ciphertext). An admin must set a new password
# via the User Manager before the account can authenticate again.
UNUSABLE_PASSWORD = "!"

# Fernet tokens are urlsafe-base64 and always start with the encoded
# version byte 0x80 -> "gAAAA...". Argon2 hashes always start with
# "$argon2". Anything else is neither a recognized legacy ciphertext nor a
# valid modern hash.
_LEGACY_PREFIX = "gAAAA"
_ARGON2_PREFIX = "$argon2"


def is_legacy_ciphertext(stored: object) -> bool:
    """True iff `stored` looks like a legacy Fernet-encrypted password."""
    return isinstance(stored, str) and stored.startswith(_LEGACY_PREFIX)


def hash_password(pw: str) -> str:
    """Return a new Argon2id hash for `pw`."""
    return ph.hash(pw)


def verify_password(stored: str, candidate: str) -> tuple[bool, bool]:
    """Verify `candidate` against `stored`.

    Returns (valid, needs_rehash). Never raises: any input that isn't a
    well-formed Argon2 hash (None, empty, the UNUSABLE_PASSWORD sentinel,
    a legacy Fernet ciphertext, or arbitrary garbage) simply fails
    verification rather than raising.
    """
    if not stored or not candidate or not isinstance(stored, str):
        return False, False
    if not stored.startswith(_ARGON2_PREFIX):
        return False, False

    try:
        ph.verify(stored, candidate)
    except (VerifyMismatchError, VerificationError, InvalidHashError, Argon2Error):
        return False, False
    except Exception:
        # Defense in depth: verification must never raise into caller code.
        return False, False

    try:
        needs_rehash = ph.check_needs_rehash(stored)
    except Exception:
        needs_rehash = False

    return True, needs_rehash


def _decrypt_legacy(ciphertext: str, key: bytes) -> str | None:
    """Decrypt one legacy Fernet ciphertext, or None on any failure.

    Module-private: this is the only place decryption logic lives, and it
    exists solely to support one-time migration. It is deliberately not
    reachable as `sg_auth.decrypt*` from outside this module.
    """
    from cryptography.fernet import Fernet  # Lazy: only migration needs this.

    try:
        return Fernet(key).decrypt(ciphertext.encode()).decode()
    except Exception:
        return None


def migrate_legacy_passwords(conn: sqlite3.Connection, key_file_path: str) -> dict:
    """One-time migration of legacy Fernet-encrypted passwords to Argon2id.

    For every `users` row whose password is legacy ciphertext: decrypt with
    the key at `key_file_path` (if present), hash the plaintext with
    Argon2id, and overwrite the column. Rows that fail to decrypt (or for
    which no key file exists) are set to UNUSABLE_PASSWORD and reported as
    failed — the account needs an admin-issued password reset. Once no
    legacy ciphertext remains in `users`, the key file is deleted.

    Idempotent: rows already migrated (or never legacy to begin with) are
    left untouched and counted as skipped. Safe to call with no key file
    and/or no users table state at all.

    Returns {migrated, failed, skipped, key_deleted}.
    """
    report = {"migrated": 0, "failed": 0, "skipped": 0, "key_deleted": False}

    rows = conn.execute("SELECT user_id, password FROM users").fetchall()
    legacy_rows = [(row[0], row[1]) for row in rows if is_legacy_ciphertext(row[1])]
    report["skipped"] = len(rows) - len(legacy_rows)

    key: bytes | None = None
    if os.path.exists(key_file_path):
        try:
            with open(key_file_path, "rb") as f:
                key = f.read()
        except OSError:
            key = None

    for user_id, ciphertext in legacy_rows:
        plaintext = _decrypt_legacy(ciphertext, key) if key is not None else None
        if plaintext is not None:
            conn.execute(
                "UPDATE users SET password = ? WHERE user_id = ?",
                (hash_password(plaintext), user_id),
            )
            report["migrated"] += 1
        else:
            conn.execute(
                "UPDATE users SET password = ? WHERE user_id = ?",
                (UNUSABLE_PASSWORD, user_id),
            )
            report["failed"] += 1

    conn.commit()

    remaining = conn.execute(
        "SELECT 1 FROM users WHERE password LIKE 'gAAAA%' LIMIT 1"
    ).fetchone()
    if remaining is None and os.path.exists(key_file_path):
        try:
            os.remove(key_file_path)
            report["key_deleted"] = True
        except OSError:
            report["key_deleted"] = False

    return report
