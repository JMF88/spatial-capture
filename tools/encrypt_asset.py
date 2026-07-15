#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""
Encrypt a scene asset so the pipeline can stay public while the room stays private.

The problem: GitHub Pages has no access control. A public repo serves public files,
full stop. But the interesting thing to show is a photoreal scan of a real, personal
space -- and "public code, private data" is the split you actually want.

The fix is to put the boundary in the file rather than the host: AES-256-GCM the
asset, commit the ciphertext, and have the viewer ask for a passphrase and decrypt
in the browser via WebCrypto. No server, no auth provider, no third-party host, and
the repo stays a static site anyone can read, fork and run. Whoever has the phrase
sees the room; everyone else reads the code.

What this is and is not:

  IS   real AES-256-GCM with a PBKDF2-SHA256-derived key. GCM is authenticated, so a
       wrong passphrase fails cleanly rather than rendering garbage, and tampering
       with the ciphertext is detected.
  NOT  revocable. The ciphertext is public and permanent -- forks and caches outlive
       any later deletion. Rotating means re-encrypting with a new phrase, and the
       old blob stays decryptable by anyone who kept it and knew the old phrase.
  NOT  stronger than the passphrase. An attacker has the ciphertext and unlimited
       offline guesses. PBKDF2 at 600k iterations makes each guess expensive, not
       impossible. Use a long random phrase (a 5-word diceware line, not "shelf123").

That threat model is the honest one for this job: it keeps a private room out of
search engines and away from casual visitors, and gates it behind something you hand
to one person. It is not a secrets manager. If you need revocation or an audit trail,
put the asset behind real auth (Cloudflare Access, a signed URL) instead.

File format -- little-endian, no ambiguity:

    magic    5s   b"SCAP1"
    kdf_iter u32  PBKDF2-HMAC-SHA256 iteration count
    salt     16s  PBKDF2 salt
    iv       12s  AES-GCM nonce
    body     ..   ciphertext || 16-byte GCM tag

Usage:
    python tools/encrypt_asset.py docs/viewer/assets/scene.spz          # prompts
    python tools/encrypt_asset.py scene.spz --out scene.spz.enc --passphrase-file p.txt
    python tools/encrypt_asset.py scene.spz.enc --decrypt --out check.spz

Dependencies: cryptography.
"""
from __future__ import annotations

import argparse
import getpass
import os
import struct
import sys
from pathlib import Path

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

MAGIC = b"SCAP1"
HEADER = struct.Struct("<5sI16s12s")   # magic, iterations, salt, iv
DEFAULT_ITERS = 600_000                # OWASP 2023 floor for PBKDF2-HMAC-SHA256
SALT_LEN, IV_LEN = 16, 12


def derive(passphrase: str, salt: bytes, iters: int) -> bytes:
    return PBKDF2HMAC(
        algorithm=hashes.SHA256(), length=32, salt=salt, iterations=iters,
    ).derive(passphrase.encode("utf-8"))


def encrypt(plain: bytes, passphrase: str, iters: int = DEFAULT_ITERS) -> bytes:
    salt = os.urandom(SALT_LEN)
    iv = os.urandom(IV_LEN)
    key = derive(passphrase, salt, iters)
    header = HEADER.pack(MAGIC, iters, salt, iv)
    # The header is authenticated as AAD, so nobody can quietly downgrade the
    # iteration count or swap the salt on a blob and have it still verify.
    body = AESGCM(key).encrypt(iv, plain, header)
    return header + body


def decrypt(blob: bytes, passphrase: str) -> bytes:
    if len(blob) < HEADER.size or not blob.startswith(MAGIC):
        raise ValueError("not a SCAP1 encrypted asset")
    magic, iters, salt, iv = HEADER.unpack(blob[:HEADER.size])
    if magic != MAGIC:
        raise ValueError("bad magic")
    key = derive(passphrase, salt, iters)
    return AESGCM(key).decrypt(iv, blob[HEADER.size:], blob[:HEADER.size])


def read_passphrase(args) -> str:
    if args.passphrase_file:
        return Path(args.passphrase_file).read_text(encoding="utf-8").strip()
    p = getpass.getpass("passphrase: ")
    if not args.decrypt:
        if p != getpass.getpass("confirm: "):
            print("error: passphrases differ", file=sys.stderr)
            raise SystemExit(2)
        if len(p) < 12:
            print("warning: short passphrase. The ciphertext is public and guessable "
                  "offline - use a long random phrase.", file=sys.stderr)
    return p


def main() -> int:
    ap = argparse.ArgumentParser(description="AES-256-GCM a scene asset for a public repo.")
    ap.add_argument("path", type=Path)
    ap.add_argument("--out", type=Path, help="default: <path>.enc, or strip .enc when decrypting")
    ap.add_argument("--decrypt", action="store_true")
    ap.add_argument("--passphrase-file", help="read the phrase from a file (keep it gitignored)")
    ap.add_argument("--iterations", type=int, default=DEFAULT_ITERS)
    args = ap.parse_args()

    if not args.path.is_file():
        print(f"error: {args.path} not found", file=sys.stderr)
        return 2

    data = args.path.read_bytes()
    phrase = read_passphrase(args)

    if args.decrypt:
        out = args.out or args.path.with_suffix("") if args.path.suffix == ".enc" else args.out
        if out is None:
            print("error: --out required", file=sys.stderr)
            return 2
        try:
            plain = decrypt(data, phrase)
        except InvalidTag:
            print("error: wrong passphrase (or the file was tampered with)", file=sys.stderr)
            return 1
        except ValueError as e:
            print(f"error: {e}", file=sys.stderr)
            return 1
        out.write_bytes(plain)
        print(f"decrypted -> {out}  ({len(plain)/1e6:.1f} MB)")
        return 0

    out = args.out or Path(str(args.path) + ".enc")
    blob = encrypt(data, phrase, args.iterations)
    out.write_bytes(blob)
    print(f"encrypted -> {out}  ({len(blob)/1e6:.1f} MB, PBKDF2 {args.iterations:,} iters)")
    print("Commit the .enc; keep the plaintext out of git. The viewer will ask for the phrase:")
    print(f"  ?src=./assets/{out.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
