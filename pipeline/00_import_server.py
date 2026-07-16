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
from anyone who can reach the port, unauthenticated and with no size cap, and --host
defaults to 0.0.0.0 -- every interface, not just the LAN one. What keeps it off the
internet is your router, not this process. So don't run it on cafe/airport WiFi, and
stop it when the import is done. Filenames are sanitised and confined to --data-root;
nothing else is writable.

Dependencies: stdlib only.
"""
from __future__ import annotations

import argparse
import json
import re
import socket
import subprocess
import sys
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

# Video containers a phone can plausibly hand us. iPhone records HEVC in .mov;
# ffmpeg reads it, so we accept and never transcode here.
VIDEO_SUFFIXES = {".mp4", ".mov", ".m4v", ".insv", ".avi", ".mkv"}
# Stills: iPhone shoots HEIC by default. DNG for anyone shooting raw.
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".heic", ".heif", ".dng", ".tif", ".tiff"}
ALLOWED_SUFFIXES = VIDEO_SUFFIXES | IMAGE_SUFFIXES
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


def probe_video(path: Path) -> dict:
    """Report what actually landed, and whether iOS quietly re-encoded it on the way.

    This is not a nicety. Uploading through the iOS Photos picker hands the page a
    "compatible" *representation*, not the file on disk: a 4K60 HEVC capture arrives
    as 4K30 H.264 at roughly the HEVC bitrate -- half the frames gone and a weaker
    codec at the same data rate. Measured, not theorised (see _private notes). The
    only way to know is to look at the bytes that arrived, so we look and say so.
    """
    try:
        r = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
             "stream=codec_name,width,height,r_frame_rate,bit_rate", "-show_entries",
             "format=duration,bit_rate", "-of", "json", str(path)],
            capture_output=True, text=True, timeout=30,
        )
        if r.returncode != 0:
            return {}
        d = json.loads(r.stdout)
        st = (d.get("streams") or [{}])[0]
        fm = d.get("format") or {}
        num, _, den = (st.get("r_frame_rate") or "0/1").partition("/")
        fps = round(float(num) / float(den or 1), 2) if float(den or 1) else 0.0
        rate = int(fm.get("bit_rate") or st.get("bit_rate") or 0)
        info = {
            "codec": st.get("codec_name"),
            "width": st.get("width"),
            "height": st.get("height"),
            "fps": fps,
            "mbps": round(rate / 1e6, 1) if rate else None,
            "seconds": round(float(fm.get("duration") or 0), 1),
        }
        # An iPhone original is HEVC unless "Most Compatible" is set; H.264 at ~25 Mbps
        # is what the Photos picker's export produces, and it caps at 30fps.
        if info["codec"] == "h264" and info["mbps"] and info["mbps"] < 30 and (info["width"] or 0) >= 3000:
            info["warning"] = (
                f"H.264 @ {info['mbps']} Mbps {info['fps']}fps - looks re-encoded by the Photos picker, "
                f"not your original. If you shot HEVC or 60fps, this is not what you shot. "
                f"Use Share -> Save to Files on the phone, then import from Files."
            )
        return info
    except (OSError, subprocess.SubprocessError, ValueError, KeyError):
        return {}


def unique_dest(dest_dir: Path, name: str) -> Path:
    """A destination path that never silently replaces an existing capture.

    Camera apps love to reuse filenames -- a master and its generated proxy commonly
    share one, differing only by folder. Uploading both to the same scene would have
    the second land on the first and destroy it with no error and no way to know which
    survived. Given the whole point of this server is that what you shot is what arrives,
    clobbering is the one failure it must not have. Suffix instead: never overwrite.
    """
    target = dest_dir / name
    if not target.exists():
        return target
    stem, suffix = target.stem, target.suffix
    for n in range(2, 1000):
        alt = dest_dir / f"{stem} ({n}){suffix}"
        if not alt.exists():
            return alt
    raise OSError(f"too many files named like {name}")


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

        # Stills go in their own subdir: stage 1 globs the scene dir for the video,
        # and 200 reference photos sitting next to it would just be in the way.
        is_video = suffix in VIDEO_SUFFIXES
        dest_dir = self.data_root / scene if is_video else self.data_root / scene / "photos"
        dest_dir.mkdir(parents=True, exist_ok=True)
        final = unique_dest(dest_dir, name)
        partial = dest_dir / (final.name + ".part")

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

        media = probe_video(final) if is_video else {}
        if media:
            print(f"   {media.get('codec')} {media.get('width')}x{media.get('height')} "
                  f"{media.get('fps')}fps {media.get('mbps')}Mbps {media.get('seconds')}s", flush=True)
        if media.get("warning"):
            print(f"   !! {media['warning']}", flush=True)
        if is_video:
            print("   next:  python pipeline/01_extract_frames.py "
                  f"--input {final.as_posix()} --out {(dest_dir / ('frames_' + final.stem)).as_posix()} "
                  "--fps 3 --long-edge 1600 --keep 0.85", flush=True)
        return self._json(200, {"ok": True, "saved": str(final), "bytes": got,
                                "scene": scene, "kind": "video" if is_video else "photo",
                                "renamed": final.name if final.name != name else None,
                                "media": media})


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
