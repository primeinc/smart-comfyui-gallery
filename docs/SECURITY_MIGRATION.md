# Security Migration: Password Storage (WI-31)

## What was wrong

SmartGallery stored user passwords as **reversible ciphertext**. On first
run it generated a symmetric Fernet key at `.sqlite_cache/system.key` and
used it to encrypt every password before writing it to the `users` table.
Login decrypted the stored value and compared it to the submitted password.

This meant:

- Anyone with read access to the database *and* the key file (both on the
  same disk, in the same cache directory) could recover every user's
  plaintext password directly.
- The admin User Manager API (`GET /galleryout/api/admin/users`) actively
  decrypted and returned every password in a `plain_password` field — the
  UI displayed plaintext passwords to any admin/manager session.
- Compromise of the key file alone was sufficient to decrypt the entire
  user table, past or present, with no way to detect or limit the damage
  after the fact (no forward secrecy, no per-user salt, no work factor).

Reversible storage of authentication secrets is unacceptable regardless of
where the key lives: a symmetric key that can decrypt every password is
itself a single point of total compromise.

## What changed

- Passwords are now hashed one-way with **Argon2id** (`argon2-cffi`, via
  the new `sg_auth.py` module). Argon2id is memory-hard, salted per
  password, and tunable in cost — the current parameters are the library's
  defaults, which are appropriate for an interactive login.
- There is **no decrypt function** and **no plaintext-password API**
  anywhere in the codebase. `sg_auth` exposes no public symbol matching
  `decrypt*`; the only decryption logic that exists is a module-private
  helper used exclusively by the one-time legacy migration described
  below, and it is never reachable from request-handling code.
- The admin User Manager endpoint no longer returns a `password` or
  `plain_password` field for any user, under any role.
- Login timing for wrong-password attempts goes through Argon2's constant
  work verification path rather than a raw string comparison of decrypted
  plaintext.

## How migration runs

Migration is **automatic and runs once at server startup**, immediately
before `ensure_admin_user()`, as part of `initialize_gallery()`:

1. Every row in `users` whose `password` column looks like a legacy Fernet
   token (`gAAAA...`) is a migration candidate.
2. If `.sqlite_cache/system.key` exists, each candidate's ciphertext is
   decrypted with it, the resulting plaintext is hashed with Argon2id, and
   the `password` column is overwritten with the hash. The plaintext is
   never logged or stored.
3. Once no legacy ciphertext remains anywhere in `users`, the key file is
   deleted. From that point on, decrypting old passwords is no longer
   possible even in principle — the only key that could do it is gone.
4. A one-line summary (migrated / failed / key file deleted) is printed to
   the console using the existing startup log style.

Migration is idempotent: rows already hashed (or already flagged unusable)
are left untouched on every subsequent boot, and the key file is only ever
deleted, never recreated.

## What happens to undecryptable rows

A row is marked **unusable** (`password` set to a sentinel value that can
never match any submitted password) when:

- its ciphertext fails to decrypt with the available key (corruption, a
  key/data mismatch, etc.), or
- the key file is missing entirely while legacy ciphertext still exists
  (the ciphertext is then permanently dead — there is no key left to try).

These accounts **cannot log in** until an administrator sets a new
password for them through the User Manager. This is a strict trade-off:
the alternative would be keeping a decrypt path alive indefinitely "just in
case," which reintroduces exactly the vulnerability this migration exists
to close.

## Why the reversible form cannot survive

A reversible password store is a standing liability with no offsetting
benefit for an authentication system: passwords only ever need to be
*verified*, never *recovered*, and any code path that can produce
plaintext from storage is also a code path an attacker can use for the
same purpose. Removing `encrypt_password`/`decrypt_password` and the
`system.key` file — rather than merely hardening them — is the only
change that actually eliminates the exposure, which is why this was a
mandatory precondition for the rest of WI-31 rather than an incremental
improvement.
