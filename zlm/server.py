"""A local web playground for Z: type a prompt, watch the model continue it token by token.

Built on Python's standard-library http.server, so it needs no web framework. Three routes:

    GET  /               the page itself (zlm/playground.html)
    GET  /api/models     the checkpoints found under checkpoints/, with their size and scores
    POST /api/generate   {"checkpoint", "prompt", "temperature", "top_k", "max_new_tokens", "seed",
                          "mode": "complete" | "chat", "history": [{"user", "z"}, ...]}
                         -> a stream of JSON lines: {"text": "..."} per new piece of text,
                            then {"done": true, "tokens": n, "seconds": t, "stop": why}

Two modes:
  complete  continue the prompt as plain text (what a base model does).
  chat      wrap the message in the format the chat fine-tuning used (scripts/prepare_chat.py):
                <|endoftext|>User: earlier message\nZ: earlier reply<|endoftext|>
                User: your message\nZ:
            and stop at <|endoftext|>, which fine-tuning taught the model to write when its
            reply is done. A base model never learned that, so in chat mode it usually rambles
            on until the length limit, or starts writing the user's next line ("User:"), where
            we stop it too.

Why a stream of JSON lines (NDJSON) and not one JSON reply: the browser can show each token
the moment it is sampled, the way chat UIs do, instead of waiting for the whole answer. The
browser reads the response body incrementally with fetch() + a stream reader. When the user
presses Stop, the browser closes the connection; our next write then fails, and we stop
generating.

This is a local tool: it listens on 127.0.0.1 (this machine only) by default, and only
loads checkpoints from the list it found itself, never a path sent by the browser.
"""

from __future__ import annotations

import json
import threading
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import torch

from zlm.checkpoint import load_model
from zlm.config import Config
from zlm.device import DeviceSetup
from zlm.model import Z
from zlm.tokenizer import EOT, BPETokenizer

PAGE = Path(__file__).with_name("playground.html")
MAX_NEW_TOKENS = 1000  # per request: keeps one request from occupying the CPU for minutes


class TextStreamer:
    """Turns generated token ids into text pieces that are safe to display one by one.

    Decoding each token on its own is not enough: a BPE token can hold *part* of a UTF-8
    character (é is 2 bytes, many emoji are 4), so the first half would show as the
    replacement character "�". Instead we decode everything generated so far and send only
    the new part, holding it back while the text ends in an unfinished character.
    """

    def __init__(self, tok: BPETokenizer) -> None:
        self.tok = tok
        self.ids: list[int] = []
        self.sent = 0  # characters already handed out

    def push(self, token_id: int) -> str:
        self.ids.append(token_id)
        text = self.tok.decode(self.ids)
        if text.endswith("�"):
            return ""  # probably the first bytes of a multi-byte character: wait for the rest
        return self._take(text)

    def flush(self) -> str:
        """Whatever is left at the end (an incomplete character really is shown as "�")."""
        return self._take(self.tok.decode(self.ids))

    def _take(self, text: str) -> str:
        new, self.sent = text[self.sent:], len(text)
        return new


class ModelStore:
    """Finds checkpoints and loads each (model + tokenizer) once, on first use."""

    def __init__(self, checkpoints_dir: Path, setup: DeviceSetup) -> None:
        self.dir = checkpoints_dir
        self.setup = setup
        self.loaded: dict[str, tuple[Z, Config, BPETokenizer]] = {}
        self._info: dict[tuple[str, float], dict[str, Any]] = {}  # (name, mtime) -> info
        # One generation at a time: on a 2-core CPU, two at once would each run at half speed.
        # The same lock guards loading and the global torch RNG (seeded per request).
        self.lock = threading.Lock()

    def names(self) -> list[str]:
        """Checkpoint paths relative to the folder, newest first, e.g. "tiny/best.pt"."""
        paths = sorted(self.dir.glob("**/*.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
        return [p.relative_to(self.dir).as_posix() for p in paths]

    def get(self, name: str) -> tuple[Z, Config, BPETokenizer]:
        if name not in self.names():  # only files we listed: never an arbitrary path
            raise KeyError(name)
        if name not in self.loaded:
            model, cfg = load_model(self.dir / name, self.setup.device)
            data_dir = Path(cfg.data.data_dir)
            meta = json.loads((data_dir / "meta.json").read_text())
            tok = BPETokenizer.load(data_dir / meta["tokenizer"])
            self.loaded[name] = (model, cfg, tok)
        return self.loaded[name]

    def info(self, name: str) -> dict[str, Any]:
        # Cached per file version: training rewrites ckpt.pt / best.pt while the server runs,
        # and a new modification time means the step and val loss have changed.
        key = (name, (self.dir / name).stat().st_mtime)
        if key not in self._info:
            self._info[key] = self._read_info(name)
        return self._info[key]

    def _read_info(self, name: str) -> dict[str, Any]:
        # map_location="cpu", weights_only: same safe loading as zlm/checkpoint.py.
        ckpt = torch.load(self.dir / name, map_location="cpu", weights_only=True)
        m = ckpt["config"]["model"]
        # Tied weights (embedding = lm_head) appear under two names but share storage: count once.
        unique = {t.data_ptr(): t.numel() for t in ckpt["model"].values()}
        return {
            "name": name,
            "step": ckpt["step"],
            "best_val": ckpt["extra"].get("best_val"),
            "params": sum(unique.values()),
            "pos_encoding": m["pos_encoding"],
            "block_size": m["block_size"],
            "vocab_size": m["vocab_size"],
            "data_dir": ckpt["config"]["data"]["data_dir"],
        }


def chat_prompt_ids(tok: BPETokenizer, history: list[dict[str, str]], message: str) -> list[int]:
    """The token ids for a chat turn, built exactly like the fine-tuning examples."""
    if EOT not in tok.special_tokens:
        raise ValueError("chat mode needs a tokenizer with <|endoftext|>")
    # Start with <|endoftext|>: in the training stream every "User:" follows one.
    ids = [tok.eot_id]
    for turn in history:
        ids += tok.encode(f"User: {turn['user']}\nZ: {turn['z']}") + [tok.eot_id]
    return ids + tok.encode(f"User: {message}\nZ:")


def _parse_request(body: dict[str, Any]) -> dict[str, Any]:
    """Validate the browser's settings. Raises ValueError with a message the page can show."""
    prompt = body.get("prompt", "")
    if not isinstance(prompt, str) or not prompt:
        raise ValueError("the prompt is empty: the model needs some text to continue")
    temperature = float(body.get("temperature", 0.8))
    if not 0 <= temperature <= 5:
        raise ValueError("temperature must be between 0 and 5")
    top_k = int(body.get("top_k", 50))
    max_new = int(body.get("max_new_tokens", 200))
    if not 1 <= max_new <= MAX_NEW_TOKENS:
        raise ValueError(f"max_new_tokens must be between 1 and {MAX_NEW_TOKENS}")
    seed = body.get("seed")
    mode = body.get("mode", "complete")
    if mode not in ("complete", "chat"):
        raise ValueError("mode must be 'complete' or 'chat'")
    history = body.get("history", [])
    if not isinstance(history, list) or not all(
        isinstance(t, dict) and isinstance(t.get("user"), str) and isinstance(t.get("z"), str)
        for t in history
    ):
        raise ValueError("history must be a list of {user, z} messages")
    return {
        "mode": mode,
        "history": history,
        "checkpoint": str(body.get("checkpoint", "")),
        "prompt": prompt,
        "temperature": temperature,
        "top_k": top_k if top_k > 0 else None,  # 0 = off
        "max_new_tokens": max_new,
        "seed": int(seed) if seed not in (None, "") else None,
    }


def make_handler(store: ModelStore) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: Any) -> None:
            pass  # keep the console for our own messages, not one line per request

        def _send_json(self, data: Any, status: HTTPStatus = HTTPStatus.OK) -> None:
            body = json.dumps(data).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:
            if self.path == "/":
                body = PAGE.read_bytes()  # read per request: edit the HTML, just reload the page
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            elif self.path == "/api/models":
                self._send_json({"models": [store.info(n) for n in store.names()],
                                 "checkpoints_dir": str(store.dir)})
            else:
                self._send_json({"error": "not found"}, HTTPStatus.NOT_FOUND)

        def do_POST(self) -> None:
            if self.path != "/api/generate":
                self._send_json({"error": "not found"}, HTTPStatus.NOT_FOUND)
                return
            try:
                length = int(self.headers.get("Content-Length", 0))
                req = _parse_request(json.loads(self.rfile.read(length) or b"{}"))
                model, cfg, tok = store.get(req["checkpoint"])
            except KeyError:
                self._send_json({"error": "unknown checkpoint"}, HTTPStatus.BAD_REQUEST)
                return
            except (ValueError, TypeError) as e:  # bad JSON or bad settings
                self._send_json({"error": str(e)}, HTTPStatus.BAD_REQUEST)
                return

            # No Content-Length: the length isn't known in advance. With HTTP/1.0 (the
            # http.server default) the end of the body is simply the closed connection.
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "application/x-ndjson")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()

            def send(obj: dict[str, Any]) -> None:
                self.wfile.write((json.dumps(obj) + "\n").encode())
                self.wfile.flush()  # push it to the browser now, not when a buffer fills up

            with store.lock:
                if req["seed"] is not None:
                    torch.manual_seed(req["seed"])
                chat = req["mode"] == "chat"
                try:
                    ids = (chat_prompt_ids(tok, req["history"], req["prompt"]) if chat
                           else tok.encode(req["prompt"]))
                except ValueError as e:
                    send({"error": str(e)})
                    return
                idx = torch.tensor([ids], dtype=torch.long, device=store.setup.device)
                streamer, count, t0 = TextStreamer(tok), 0, time.perf_counter()
                text, stop = "", "length"
                try:
                    with store.setup.autocast():
                        for next_id in model.generate_stream(
                            idx, req["max_new_tokens"], req["temperature"], req["top_k"]
                        ):
                            count += 1
                            if chat and int(next_id) == tok.eot_id:
                                stop = "end"  # the model says its reply is finished
                                break
                            piece = streamer.push(int(next_id))
                            if not piece:
                                continue
                            text += piece
                            if chat and "\nUser:" in text:
                                # The model started writing *your* next message: cut that off.
                                # "cut" tells the page how much of what it got to keep.
                                send({"text": piece, "cut": text.index("\nUser:")})
                                stop = "user-turn"
                                break
                            send({"text": piece})
                    if stop == "length" and (tail := streamer.flush()):
                        send({"text": tail})
                    send({"done": True, "tokens": count, "seconds": time.perf_counter() - t0,
                          "stop": stop, "prompt_tokens": len(ids),
                          "context_limit": cfg.model.block_size})
                except (BrokenPipeError, ConnectionResetError):
                    pass  # the user pressed Stop (or closed the tab): stop generating

    return Handler


def make_server(host: str, port: int, store: ModelStore) -> ThreadingHTTPServer:
    # Threading: the page can load model info while another request is still generating.
    return ThreadingHTTPServer((host, port), make_handler(store))
