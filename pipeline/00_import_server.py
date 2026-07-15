#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""
Stage 0 - LAN import.

Get the capture off the phone and onto the workstation over WiFi, with no cable,
no cloud, and no compression pass. Serves the Trove app to your phone and accepts
the video straight into `data/<scene>/`.

Why this stage exists: the phone is where the capture happens and the GPU is where
it gets reconstructed, and the gap between them is the least glamorous, most
reliably annoying part of the pipeline. AirDrop re-encodes, cloud round-trips cost
minutes and quality, and cables mean finding a cable. This is one command and a URL.

Why it can't just be the GitHub Pages app: that page is HTTPS, and a secure page
may not POST to a plain-HTTP LAN address (mixed content). So the same app is served
here over HTTP -- phone and server share an origin, and the upload is same-origin.

The upload is streamed to disk in chunks: a 4 GB capture never lands in RAM.

Usage:
    python pipeline/00_import_server.py                 # serves docs/, writes to data/
    python pipeline/00_import_server.py --port 8099 --data-root data
    # then, on the phone (same WiFi):  http://<your-lan-ip>:8099/app/

Security: a deliberately small dev tool for a *trusted* network. It accepts writes
from anyone who can reach the port, so it binds your LAN, not the internet. Don't
run it on cafe/airport WiFi, and stop it when the import is done. Filenames are
sanitised and confined to --data-root; nothing else is writable.

Dependencies: stdlib only.
"""
from __future__ import annotations

import argparse
import json
import re
import socket
import sys
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

# Video containers a phone can plausibly hand us. iPhone records HEVC in .mov;
# ffmpeg reads it, so we accept and never transcode here.
ALLOWED_SUFFIXES = {".mp4", ".mov", ".m4v", ".insv", ".avi", ".mkv"}
CHUNK = 1 << 20  # 1 MiB
SAFE = re.compile(r"[^A-Za-z0-9._-]")


def safe_name(raw: str, fallback: str) -> str:
    """Reduce an untrusted client string to a bare, harmless filename.

    Takes the basename only, strips anything outside [A-Za-z0-9._-], and refuses
    empties and dot-only names -- so no traversal, no absolute paths, no surprises.
    """
    base = raw.replace("\\", "/").split("/")[-1]
    cleaned = SAFE.sub("_", base).lstrip(".")
    return cleaned if cleaned and cleaned.strip(".") else fallback


def lan_ip() -> str:
    """Best-effort primary LAN IPv4.

    Opens a UDP socket toward a public address and asks which local interface the
    kernel would route it through. No packet is sent; it just resolves the route,
    which beats guessing among WSL/Hyper-V/VPN adapters that all look plausible.
    """
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0]
    except OSError:
        return "127.0.0.1"
    finally:
        s.close()


class ImportHandler(SimpleHTTPRequestHandler):
    """Static file server for docs/ plus a streaming upload endpoint."""

    data_root: Path = Path("data")
    # Quieten the default per-request logging; we print what matters ourselves.
    def log_message(self, fmt: str, *args) -> None:  # noqa: A003 - stdlib name
        if "/api/" in self.path:
            sys.stderr.write("  %s\n" % (fmt % args))

    def _json(self, code: int, payload: dict) -> None:
        body = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802 - stdlib name
        if urlparse(self.path).path == "/api/health":
            scenes = sorted(p.name for p in self.data_root.glob("*") if p.is_dir())
            return self._json(200, {"ok": True, "dataRoot": str(self.data_root), "scenes": scenes})
        return super().do_GET()

    def do_POST(self) -> None:  # noqa: N802 - stdlib name
        url = urlparse(self.path)
        if url.path != "/api/upload":
            return self._json(404, {"ok": False, "error": "unknown endpoint"})

        q = parse_qs(url.query)
        scene = safe_name(q.get("scene", ["shelf"])[0], "shelf")
        name = safe_name(q.get("name", ["capture.mp4"])[0], "capture.mp4")

        suffix = Path(name).suffix.lower()
        if suffix not in ALLOWED_SUFFIXES:
            return self._json(
                415, {"ok": False, "error": f"{suffix or '(none)'} not accepted; "
                      f"expected one of {sorted(ALLOWED_SUFFIXES)}"}
            )

        try:
            total = int(self.headers.get("Content-Length", ""))
        except ValueError:
            return self._json(411, {"ok": False, "error": "Content-Length required"})

        dest_dir = self.data_root / scene
        dest_dir.mkdir(parents=True, exist_ok=True)
        final = dest_dir / name
        partial = dest_dir / (name + ".part")

        mb = total / (1 << 20)
        print(f"\n<- receiving {name} ({mb:,.0f} MB) -> {final}", flush=True)

        got = 0
        step = max(total // 20, 1)  # progress line every ~5%
        next_mark = step
        try:
            with partial.open("wb") as fh:
                while got < total:
                    chunk = self.rfile.read(min(CHUNK, total - got))
                    if not chunk:
                        raise ConnectionError("client disconnected mid-upload")
                    fh.write(chunk)
                    got += len(chunk)
                    if got >= next_mark:
                        pct = 100 * got / total
                        print(f"   {pct:5.1f}%  ({got / (1 << 20):,.0f} MB)", flush=True)
                        next_mark += step
        except (ConnectionError, OSError) as exc:
            partial.unlink(missing_ok=True)
            print(f"!! upload failed: {exc}", flush=True)
            return self._json(500, {"ok": False, "error": str(exc)})

        # Only becomes the real filename once every byte is on disk, so a killed
        # upload can never look like a complete capture to the next stage.
        partial.replace(final)
        print(f"OK saved {final}  ({mb:,.0f} MB)", flush=True)
        print("   next:  python pipeline/01_extract_frames.py "
              f"--input {final.as_posix()} --out {(dest_dir / 'frames').as_posix()} "
              "--fps 3 --long-edge 1600 --keep 0.85", flush=True)
        return self._json(200, {"ok": True, "saved": str(final), "bytes": got, "scene": scene})


def main() -> int:
    ap = argparse.ArgumentParser(description="Serve Trove to your phone and import captures over WiFi.")
    ap.add_argument("--port", type=int, default=8099)
    ap.add_argument("--host", default="0.0.0.0", help="bind address (default: all interfaces)")
    ap.add_argument("--serve-root", type=Path, default=Path("docs"), help="static dir served to the phone")
    ap.add_argument("--data-root", type=Path, default=Path("data"), help="where captures are written")
    args = ap.parse_args()

    serve_root = args.serve_root.resolve()
    data_root = args.data_root.resolve()
    if not serve_root.is_dir():
        print(f"error: --serve-root {serve_root} does not exist (run me from the repo root)", file=sys.stderr)
        return 2
    data_root.mkdir(parents=True, exist_ok=True)

    ImportHandler.data_root = data_root
    handler = lambda *a, **kw: ImportHandler(*a, directory=str(serve_root), **kw)  # noqa: E731

    ip = lan_ip()
    srv = ThreadingHTTPServer((args.host, args.port), handler)
    print("spatial-capture - LAN import server")
    print(f"  serving : {serve_root}")
    print(f"  writing : {data_root}")
    print("")
    print("  On your phone (same WiFi):")
    print(f"    Trove   http://{ip}:{args.port}/app/")
    print(f"    Viewer  http://{ip}:{args.port}/viewer/")
    print("")
    print("  First run may raise a Windows Firewall prompt -- allow it on Private networks.")
    print("  Ctrl-C to stop.\n")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped.")
    finally:
        srv.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
