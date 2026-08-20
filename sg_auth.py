# sg_auth.py — one-way password hashing for SmartGallery.
#
# Passwords are stored ONLY as Argon2id hashes. There is no supported way to
# recover a stored password from this module. The single exception is
# migrate_legacy_passwords(), which performs a one-time decrypt-then-rehash
# of legacy Fernet ciphertexts (a reversible storage scheme this module
# supersedes); its decryption helper is module-private and is never exposed
# as a general-purpose "decrypt this password" API.

import contextlib
import logging
import os
import secrets
import sqlite3

from argon2 import PasswordHasher
from argon2.exceptions import Argon2Error, InvalidHashError, VerificationError, VerifyMismatchError
from cryptography.fernet import Fernet  # Lazy: only migration needs this.

_logger = logging.getLogger(__name__)

# Library defaults (time_cost, memory_cost, parallelism, hash_len, salt_len,
# type=Argon2id) are appropriate here; no tuning required.
ph = PasswordHasher()

# Stored in the password column for accounts that cannot log in (e.g.
# undecryptable legacy ciphertext). It is not a password and no password
# hashes to it, so nothing can match; an admin must set a new one via the
# User Manager before the account authenticates again.
LOGIN_DISABLED = "!"

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
    """Return an Argon2id hash of `pw`, salted freshly on every call."""
    return ph.hash(pw)


def verify_password(stored: str, candidate: str) -> tuple[bool, bool]:
    """Verify `candidate` against `stored`.

    Returns (valid, needs_rehash). Never raises: any input that isn't a
    well-formed Argon2 hash (None, empty, the LOGIN_DISABLED sentinel,
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
        _logger.debug("handled a failure in verify_password", exc_info=True)
        return False, False

    try:
        needs_rehash = ph.check_needs_rehash(stored)
    except Exception:
        _logger.debug("handled a failure in verify_password", exc_info=True)
        needs_rehash = False

    return True, needs_rehash


def constant_time_equals(a, b) -> bool:
    """Total, constant-time equality for two secrets that may be any type.

    Unlike secrets.compare_digest on str, this never raises: str operands
    containing non-ASCII characters, and non-str JSON operands (int/list),
    are handled by normalizing to UTF-8 bytes first. Returns False for any
    input that cannot be meaningfully compared.
    """
    try:
        ab = a if isinstance(a, bytes) else str(a).encode("utf-8")
        bb = b if isinstance(b, bytes) else str(b).encode("utf-8")
    except Exception:
        _logger.debug("handled a failure in constant_time_equals", exc_info=True)
        return False
    return secrets.compare_digest(ab, bb)


# A fixed decoy hash so authentication can perform an Argon2id verification
# even when no user row is found, keeping login timing independent of whether
# a username exists (mitigates account enumeration via response latency).
_DECOY_HASH = ph.hash("sg_auth_decoy_password_do_not_use")


def dummy_verify() -> None:
    """Run one Argon2id verification against a throwaway hash and discard the
    result. Call on the user-not-found path so it costs the same as a real
    verify. Never raises."""
    with contextlib.suppress(Exception):
        ph.verify(_DECOY_HASH, "x")


def _decrypt_legacy(ciphertext: str, key: bytes) -> str | None:
    """Decrypt one legacy Fernet ciphertext, or None on any failure.

    Module-private: this is the only place decryption logic lives, and it
    exists solely to support one-time migration. It is deliberately not
    reachable as `sg_auth.decrypt*` from outside this module.
    """

    try:
        return Fernet(key).decrypt(ciphertext.encode()).decode()
    except Exception:
        _logger.debug("handled a failure in _decrypt_legacy", exc_info=True)
        return None


def migrate_legacy_passwords(conn: sqlite3.Connection, key_file_path: str) -> dict:
    """One-time migration of legacy Fernet-encrypted passwords to Argon2id.

    For every `users` row whose password is legacy ciphertext: decrypt with
    the key at `key_file_path` (if present), hash the plaintext with
    Argon2id, and overwrite the column. Rows that fail to decrypt (or for
    which no key file exists) are set to LOGIN_DISABLED and reported as
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
                (LOGIN_DISABLED, user_id),
            )
            report["failed"] += 1

    conn.commit()

    # GLOB (case-sensitive) so this completeness gate matches
    # is_legacy_ciphertext()'s case-sensitive startswith exactly; a
    # case-insensitive LIKE could disagree and wrongly retain the key file.
    remaining = conn.execute("SELECT 1 FROM users WHERE password GLOB 'gAAAA*' LIMIT 1").fetchone()
    if remaining is None and os.path.exists(key_file_path):
        try:
            os.remove(key_file_path)
            report["key_deleted"] = True
        except OSError:
            report["key_deleted"] = False

    return report
