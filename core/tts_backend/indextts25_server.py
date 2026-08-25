"""Small HTTP adapter for an official IndexTTS 2.5 checkout.

This file is launched with the target checkout's uv/Python environment while
the working directory is the IndexTTS repository, so its local ``indextts``
package and checkpoints are used without modifying that repository.
"""

import argparse
import json
import threading
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="VideoLingo IndexTTS 2.5 adapter")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9872)
    parser.add_argument("--model_dir", default="checkpoints")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--cuda_kernel", action="store_true")
    parser.add_argument("--deepspeed", action="store_true")
    parser.add_argument("--accel", action="store_true")
    parser.add_argument("--torch_compile", action="store_true")
    parser.add_argument("--use_qwen_emo", action="store_true")
    return parser.parse_args()


class Service:
    def __init__(self, args):
        from indextts.infer_v2_5 import IndexTTS2

        model_dir = Path(args.model_dir).resolve()
        self.tts = IndexTTS2(
            cfg_path=str(model_dir / "config.yaml"),
            model_dir=str(model_dir),
            use_bf16=args.bf16,
            use_cuda_kernel=args.cuda_kernel,
            use_deepspeed=args.deepspeed,
            use_accel=args.accel,
            use_torch_compile=args.torch_compile,
            use_qwen_emo=args.use_qwen_emo,
        )
        self.lock = threading.Lock()

    def infer(self, request):
        output_path = Path(request["output_path"]).resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with self.lock:
            self.tts.infer(
                spk_audio_prompt=request["spk_audio_prompt"],
                text=request["text"],
                output_path=str(output_path),
                lang=request.get("lang", "ZH"),
                emo_alpha=float(request.get("emo_alpha", 1.0)),
                use_emo_text=bool(request.get("use_emo_text", False)),
                emo_text=request.get("emo_text"),
                use_random=bool(request.get("use_random", False)),
                interval_silence=int(request.get("interval_silence", 200)),
                max_text_tokens_per_segment=int(request.get("max_text_tokens_per_segment", 120)),
                duration_factor=float(request.get("duration_factor", 1.0)),
                text_normalization=bool(request.get("text_normalization", True)),
            )
        return {"ok": True, "output_path": str(output_path), "version": "2.5"}


def handler_factory(service):
    class Handler(BaseHTTPRequestHandler):
        def _json(self, status, body):
            data = json.dumps(body, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.path == "/ping":
                self._json(200, {"ok": True, "version": "2.5"})
            else:
                self._json(404, {"ok": False, "error": "not found"})

        def do_POST(self):
            if self.path != "/tts":
                self._json(404, {"ok": False, "error": "not found"})
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                request = json.loads(self.rfile.read(length).decode("utf-8"))
                self._json(200, service.infer(request))
            except Exception as exc:
                traceback.print_exc()
                self._json(500, {"ok": False, "error": str(exc)})

        def log_message(self, fmt, *args):
            print("IndexTTS2.5 HTTP:", fmt % args, flush=True)

    return Handler


def main():
    args = parse_args()
    service = Service(args)
    server = ThreadingHTTPServer((args.host, args.port), handler_factory(service))
    print(f"IndexTTS 2.5 adapter ready at http://{args.host}:{args.port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
