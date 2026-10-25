"""Serve the demo page on this laptop and connect its AI coach chat to Ollama.

The published page cannot reach a local Ollama (the hosting sandbox blocks it),
so run the page from here instead:

    python -m skilltwin.serve            then open http://localhost:8765

Routes:
    GET  /                  the demo page (out/demo/skill_twin.html)
    GET  /api/coach-status  is Ollama running, which local models are installed
    POST /api/coach-chat    {"model", "messages"} -> Ollama /api/chat, streamed back as NDJSON

Listens on 127.0.0.1 only, so nothing on the network can reach it.
"""

import argparse
import json
import urllib.error
import urllib.request
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

from skilltwin.config import OUT_DIR

DEMO_DIR = OUT_DIR / "demo"


class Handler(SimpleHTTPRequestHandler):
    ollama = "http://localhost:11434"
    # without an explicit charset, browsers decode the page as Latin-1 and garble Japanese/Hindi
    extensions_map = {**SimpleHTTPRequestHandler.extensions_map,
                      ".html": "text/html; charset=utf-8", ".json": "application/json; charset=utf-8",
                      ".webm": "video/webm"}

    def do_GET(self):
        if self.path == "/api/coach-status":
            return self._status()
        if self.path in ("/", "/index.html"):
            self.path = "/skill_twin.html"
        return super().do_GET()

    def do_POST(self):
        if self.path != "/api/coach-chat":
            return self.send_error(404)
        try:
            req = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
            payload = {"model": req.get("model") or "gemma3", "messages": req["messages"],
                       "stream": True, "options": {"temperature": 0.2, "top_p": 0.8, "num_ctx": 8192}}
        except (ValueError, KeyError):
            return self._json(400, {"error": "Expected JSON with a messages list."})
        try:
            upstream = urllib.request.urlopen(urllib.request.Request(
                self.ollama + "/api/chat", data=json.dumps(payload).encode(),
                headers={"Content-Type": "application/json"}), timeout=300)
        except (urllib.error.URLError, OSError) as e:
            return self._json(502, {"error": f"Ollama is not reachable at {self.ollama} ({e}). "
                                             "Start it with: ollama serve"})
        self.send_response(200)
        self.send_header("Content-Type", "application/x-ndjson")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        with upstream:
            for line in upstream:                  # one JSON object per token chunk
                self.wfile.write(line)
                self.wfile.flush()

    def _status(self):
        try:
            with urllib.request.urlopen(self.ollama + "/api/tags", timeout=3) as r:
                models = [m["name"] for m in json.load(r).get("models", [])]
            self._json(200, {"ok": True, "models": models})
        except (urllib.error.URLError, OSError, ValueError) as e:
            self._json(200, {"ok": False, "error": f"Ollama is not running ({e}). Start it with: ollama serve"})

    def _json(self, code, obj):
        data = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, fmt, *args):
        pass


def main():
    ap = argparse.ArgumentParser(description="Serve the Skill Twin demo with a local AI coach")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--ollama", default="http://localhost:11434")
    args = ap.parse_args()
    Handler.ollama = args.ollama.rstrip("/")
    server = ThreadingHTTPServer(("127.0.0.1", args.port), partial(Handler, directory=str(DEMO_DIR)))
    print(f"Skill Twin is running: open http://localhost:{args.port}   (Ctrl+C to stop)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
