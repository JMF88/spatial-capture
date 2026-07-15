"""Gated assets: public repo, private capture.

The security claim is only worth what the failure modes are worth, so these test the
refusals as hard as the round-trip: a wrong passphrase and a tampered byte must both
be rejected, not silently produce garbage that the renderer would then try to draw.

The format is also a contract with docs/viewer/decrypt.js -- Python writes it, WebCrypto
reads it. test_header_layout pins the byte layout so the two can't drift apart silently;
cross-language decryption itself was verified against Node's WebCrypto.
"""
import struct

import pytest

pytest.importorskip("cryptography")


@pytest.fixture
def enc(load_module):
    return load_module("tools/encrypt_asset.py", "encrypt_asset")


def test_round_trip(enc):
    payload = b"gaussian splat bytes" * 500
    blob = enc.encrypt(payload, "a long enough passphrase", iters=1000)
    assert enc.decrypt(blob, "a long enough passphrase") == payload


def test_wrong_passphrase_is_rejected(enc):
    from cryptography.exceptions import InvalidTag
    blob = enc.encrypt(b"secret room", "right passphrase", iters=1000)
    with pytest.raises(InvalidTag):
        enc.decrypt(blob, "wrong passphrase")


def test_tampered_ciphertext_is_rejected(enc):
    from cryptography.exceptions import InvalidTag
    blob = bytearray(enc.encrypt(b"secret room", "right passphrase", iters=1000))
    blob[-3] ^= 0x01
    with pytest.raises(InvalidTag):
        enc.decrypt(bytes(blob), "right passphrase")


def test_header_is_authenticated(enc):
    """The KDF cost is in the header, so it must be covered by the GCM tag.

    Otherwise an attacker could rewrite 600k iterations down to 1 and hand the victim
    a blob that still verifies -- turning an expensive offline guess into a cheap one.
    """
    from cryptography.exceptions import InvalidTag
    blob = bytearray(enc.encrypt(b"secret room", "right passphrase", iters=50_000))
    struct.pack_into("<I", blob, 5, 1)      # iterations: 50000 -> 1
    with pytest.raises(InvalidTag):
        enc.decrypt(bytes(blob), "right passphrase")


def test_header_layout_matches_the_js_reader(enc):
    """Pins the wire format that docs/viewer/decrypt.js parses by fixed offsets."""
    blob = enc.encrypt(b"x", "passphrase here", iters=600_000)
    assert blob[:5] == b"SCAP1"
    assert enc.HEADER.size == 37, "decrypt.js hard-codes HEADER_LEN = 37"
    magic, iters, salt, iv = enc.HEADER.unpack(blob[:37])
    assert (magic, iters, len(salt), len(iv)) == (b"SCAP1", 600_000, 16, 12)


def test_not_encrypted_input_raises(enc):
    with pytest.raises(ValueError, match="SCAP1"):
        enc.decrypt(b"just a plain ply file, honest", "phrase")


def test_each_encryption_is_unique(enc):
    """Fresh salt and IV per call: identical plaintext must not yield identical bytes,
    or re-publishing a scene would leak that nothing changed."""
    a = enc.encrypt(b"same", "same phrase", iters=1000)
    b = enc.encrypt(b"same", "same phrase", iters=1000)
    assert a != b
    assert enc.decrypt(a, "same phrase") == enc.decrypt(b, "same phrase") == b"same"
