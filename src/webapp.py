"""Local-only HTTP server for the Soniq voice review studio."""
from __future__ import annotations

import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from src.detector import analyze_wav_bytes, build_demo_wav

ROOT = Path(__file__).resolve().parents[1]
WEB_ROOT = ROOT / "web"
MAX_UPLOAD_BYTES = 12 * 1024 * 1024
STATIC_FILES = {
    "/web/styles.css": ("styles.css", "text/css; charset=utf-8"),
    "/web/app.js": ("app.js", "text/javascript; charset=utf-8"),
    "/web/mark.svg": ("mark.svg", "image/svg+xml"),
}


class VoiceReviewHandler(BaseHTTPRequestHandler):
    """Serve the UI, demo tones, and in-memory WAV analysis endpoint."""

    server_version = "SoniqLocal/1.0"

    def _send(self, status: int, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("X-Frame-Options", "DENY")
        self.send_header("Cross-Origin-Resource-Policy", "same-origin")
        self.send_header(
            "Content-Security-Policy",
            "default-src 'self'; script-src 'self'; style-src 'self'; "
            "img-src 'self' data:; media-src 'self' blob:; connect-src 'self'; "
            "base-uri 'none'; form-action 'none'; frame-ancestors 'none'",
        )
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _send_json(self, status: int, payload: dict) -> None:
        body = json.dumps(payload, separators=(",", ":"), allow_nan=False).encode("utf-8")
        self._send(status, body, "application/json; charset=utf-8")

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/":
            self._send(200, (WEB_ROOT / "index.html").read_bytes(), "text/html; charset=utf-8")
            return
        if parsed.path in STATIC_FILES:
            filename, content_type = STATIC_FILES[parsed.path]
            self._send(200, (WEB_ROOT / filename).read_bytes(), content_type)
            return
        if parsed.path == "/api/health":
            self._send_json(200, {"status": "ok", "mode": "local-only"})
            return
        if parsed.path in {"/api/demo", "/api/demo.wav"}:
            kind = parse_qs(parsed.query).get("kind", ["reference"])[0]
            try:
                sample = build_demo_wav(kind)
                if parsed.path.endswith(".wav"):
                    self._send(200, sample, "audio/wav")
                else:
                    payload = analyze_wav_bytes(sample)
                    payload["demo_kind"] = kind
                    payload["demo_notice"] = "A generated test tone, not a recording of human speech."
                    self._send_json(200, payload)
            except ValueError as error:
                self._send_json(400, {"error": str(error)})
            return
        self._send_json(404, {"error": "Not found"})

    def do_POST(self) -> None:
        if urlparse(self.path).path != "/api/analyze":
            self._send_json(404, {"error": "Not found"})
            return
        try:
            content_length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            self._send_json(400, {"error": "Invalid upload size."})
            return
        if content_length <= 0:
            self._send_json(400, {"error": "Choose a WAV file first."})
            return
        if content_length > MAX_UPLOAD_BYTES:
            self._send_json(413, {"error": "WAV uploads are limited to 12 MB."})
            return
        content_type = self.headers.get("Content-Type", "").split(";", 1)[0].lower().strip()
        if content_type not in {"audio/wav", "audio/x-wav", "application/octet-stream"}:
            self._send_json(415, {"error": "Upload an uncompressed PCM WAV file."})
            return
        raw = self.rfile.read(content_length)
        if len(raw) != content_length:
            self._send_json(400, {"error": "The upload ended unexpectedly. Please try again."})
            return
        try:
            result = analyze_wav_bytes(raw)
            result["source"] = "uploaded"
            self._send_json(200, result)
        except ValueError as error:
            self._send_json(400, {"error": str(error)})
        except Exception:
            self._send_json(500, {"error": "Audio analysis failed. Check the WAV format and try again."})

    def log_message(self, format_string: str, *args: object) -> None:
        """Log status information without writing filenames, bytes, or audio features."""
        print(f"[Soniq] {self.address_string()} - {format_string % args}")


def main() -> None:
    """Launch on loopback, not on the local network."""
    try:
        port = int(os.environ.get("PORT", "8501"))
    except ValueError as error:
        raise SystemExit("PORT must be an integer.") from error
    if not 1 <= port <= 65535:
        raise SystemExit("PORT must be between 1 and 65535.")
    server = ThreadingHTTPServer(("127.0.0.1", port), VoiceReviewHandler)
    print(f"Soniq is ready at http://127.0.0.1:{port} (local-only)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping Soniq.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
