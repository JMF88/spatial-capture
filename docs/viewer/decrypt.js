// SPDX-License-Identifier: MIT
//
// Client-side decryption for gated scene assets. Pairs with tools/encrypt_asset.py.
//
// Why this exists: GitHub Pages serves a public repo publicly, with no access control.
// Keeping the pipeline open while keeping a scan of a real room private means putting
// the boundary in the file, not the host -- so the asset ships as AES-256-GCM
// ciphertext and is decrypted here, in the browser, with a passphrase. No server, no
// auth provider, nothing to host. WebCrypto is built in, so this adds no dependency
// and no external request, which keeps the viewer a self-contained static page.
//
// GCM is authenticated: a wrong passphrase throws instead of handing the renderer
// garbage, and the header is bound as additional data so nobody can strip the KDF
// iteration count down and have the blob still verify.
//
// Format (little-endian) -- must stay in lockstep with tools/encrypt_asset.py:
//   magic "SCAP1" (5) | iterations u32 (4) | salt (16) | iv (12) | ciphertext+tag

const MAGIC = "SCAP1";
const HEADER_LEN = 5 + 4 + 16 + 12; // 37

// The KDF cost is read from the file, and binding it as AAD only proves it wasn't
// altered -- the proof arrives AFTER the derivation has already run. So the count is
// attacker-controlled work: a 101-byte blob claiming 2^32-1 iterations wedges the tab
// for hours before GCM ever gets to reject it. Bound it before deriving, not after.
// 4M is ~7x the 600k default -- headroom to re-encrypt harder later, still ~seconds.
const MAX_ITERS = 4_000_000;

export function isEncrypted(buf) {
  if (!buf || buf.byteLength < HEADER_LEN) return false;
  const head = new Uint8Array(buf, 0, 5);
  return String.fromCharCode(...head) === MAGIC;
}

export async function decryptAsset(buf, passphrase) {
  if (!isEncrypted(buf)) throw new Error("not a SCAP1 encrypted asset");
  const u8 = new Uint8Array(buf);
  const dv = new DataView(buf);
  const iterations = dv.getUint32(5, true);
  if (iterations < 1 || iterations > MAX_ITERS) {
    // .malformed marks "no passphrase can fix this" -- the caller must not re-prompt.
    const e = new Error(`implausible PBKDF2 iteration count (${iterations})`);
    e.malformed = true;
    throw e;
  }
  const salt = u8.slice(9, 25);
  const iv = u8.slice(25, 37);
  const aad = u8.slice(0, HEADER_LEN);
  const body = u8.slice(HEADER_LEN);

  const base = await crypto.subtle.importKey(
    "raw", new TextEncoder().encode(passphrase), "PBKDF2", false, ["deriveKey"],
  );
  // 600k PBKDF2 iterations is ~a second on a phone. That is the point: it is the only
  // thing standing between a public ciphertext and an offline guessing attack, so the
  // cost is deliberate and belongs on the attacker's side too.
  const key = await crypto.subtle.deriveKey(
    { name: "PBKDF2", salt, iterations, hash: "SHA-256" },
    base, { name: "AES-GCM", length: 256 }, false, ["decrypt"],
  );
  try {
    return await crypto.subtle.decrypt({ name: "AES-GCM", iv, additionalData: aad }, key, body);
  } catch (_) {
    throw new Error("wrong passphrase");
  }
}
