import json
import threading
import urllib.error
import urllib.request

import pytest
import torch

from zlm.checkpoint import save_checkpoint
from zlm.config import config_from_dict
from zlm.device import setup_device
from zlm.model import Z
from zlm.server import ModelStore, TextStreamer, make_server
from zlm.tokenizer import BPETokenizer

# --- TextStreamer ---------------------------------------------------------------------


def test_streamer_holds_back_half_a_character():
    tok = BPETokenizer()  # untrained: one token per byte, so "é" is two tokens
    e1, e2 = "é".encode()
    s = TextStreamer(tok)
    assert s.push(ord("a")) == "a"
    assert s.push(e1) == ""      # first byte of é alone would show as "�"
    assert s.push(e2) == "é"     # now complete
    assert s.push(ord("!")) == "!"
    assert s.flush() == ""


def test_streamer_pieces_add_up_to_the_full_decode():
    tok = BPETokenizer()
    ids = list("héllo 日本 ok".encode())
    s = TextStreamer(tok)
    assert "".join(s.push(i) for i in ids) + s.flush() == "héllo 日本 ok"


def test_streamer_flush_shows_an_unfinished_character():
    s = TextStreamer(BPETokenizer())
    s.push("é".encode()[0])
    assert s.flush() == "�"


# --- the HTTP server, end to end ---------------------------------------------------------


@pytest.fixture(scope="module")
def server(tmp_path_factory):
    """A real server on a free port, with one tiny untrained checkpoint."""
    root = tmp_path_factory.mktemp("serve")
    data_dir = root / "data"
    data_dir.mkdir()
    tok = BPETokenizer()
    tok.train("hello world, hello there. " * 20, vocab_size=280)
    tok.save(data_dir / "tokenizer.model")
    (data_dir / "meta.json").write_text(json.dumps({"tokenizer": "tokenizer.model"}))

    cfg = config_from_dict({
        "model": dict(vocab_size=tok.vocab_size, block_size=16, n_layer=1, n_head=2, n_embd=16),
        "data": {"data_dir": str(data_dir)},
    })
    model = Z(cfg.model)
    save_checkpoint(root / "ckpts" / "run" / "best.pt", model,
                    torch.optim.AdamW(model.parameters()), step=5, cfg=cfg, extra={"best_val": 4.2})

    store = ModelStore(root / "ckpts", setup_device("cpu"))
    srv = make_server("127.0.0.1", 0, store)  # port 0: the OS picks a free port
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_address[1]}", model
    srv.shutdown()
    srv.server_close()


def post(url, body):
    req = urllib.request.Request(url, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    return urllib.request.urlopen(req)


def test_page_is_served(server):
    base, _ = server
    with urllib.request.urlopen(base + "/") as r:
        assert r.headers["Content-Type"].startswith("text/html")
        assert b"Z playground" in r.read().replace(b"<span>", b"").replace(b"</span>", b"")


def test_models_lists_the_checkpoint(server):
    base, model = server
    with urllib.request.urlopen(base + "/api/models") as r:
        models = json.load(r)["models"]
    assert [m["name"] for m in models] == ["run/best.pt"]
    assert models[0]["step"] == 5 and models[0]["best_val"] == 4.2
    assert models[0]["params"] == model.num_params()  # tied weights counted once


def test_generate_streams_json_lines(server):
    base, _ = server
    body = {"checkpoint": "run/best.pt", "prompt": "hello", "max_new_tokens": 20,
            "temperature": 1.0, "top_k": 10, "seed": 1}
    with post(base + "/api/generate", body) as r:
        assert r.headers["Content-Type"] == "application/x-ndjson"
        lines = [json.loads(line) for line in r.read().decode().splitlines()]
    *pieces, done = lines
    assert done["done"] and done["tokens"] == 20 and done["prompt_tokens"] > 0
    assert all("text" in p for p in pieces)


def test_same_seed_same_text(server):
    base, _ = server
    body = {"checkpoint": "run/best.pt", "prompt": "hello", "max_new_tokens": 15, "seed": 7}

    def text():
        with post(base + "/api/generate", body) as r:
            return "".join(json.loads(line).get("text", "") for line in r.read().decode().splitlines())

    assert text() == text()


@pytest.mark.parametrize("body, message", [
    ({"checkpoint": "../../etc/passwd", "prompt": "hi"}, "unknown checkpoint"),
    ({"checkpoint": "run/best.pt", "prompt": ""}, "prompt is empty"),
    ({"checkpoint": "run/best.pt", "prompt": "hi", "max_new_tokens": 5000}, "max_new_tokens"),
    ({"checkpoint": "run/best.pt", "prompt": "hi", "temperature": -1}, "temperature"),
])
def test_bad_requests_are_rejected(server, body, message):
    base, _ = server
    with pytest.raises(urllib.error.HTTPError) as e:
        post(base + "/api/generate", body)
    assert e.value.code == 400
    assert message in json.load(e.value)["error"]


# --- chat mode -----------------------------------------------------------------------------


def test_chat_prompt_matches_the_fine_tuning_format():
    from zlm.server import chat_prompt_ids

    tok = BPETokenizer()
    tok.train("User: hi\nZ: hello there " * 10, vocab_size=280)
    ids = chat_prompt_ids(tok, [{"user": "hi", "z": "hello"}], "who are you?")
    assert ids[0] == tok.eot_id  # every "User:" follows <|endoftext|>, as in training
    assert tok.decode(ids[1:]) == "User: hi\nZ: hello<|endoftext|>User: who are you?\nZ:"


def test_chat_mode_streams_and_reports_why_it_stopped(server):
    base, _ = server
    body = {"checkpoint": "run/best.pt", "prompt": "hello", "mode": "chat", "seed": 3,
            "history": [{"user": "hi", "z": "hello"}], "max_new_tokens": 30}
    with post(base + "/api/generate", body) as r:
        lines = [json.loads(line) for line in r.read().decode().splitlines()]
    done = lines[-1]
    assert done["done"] and done["stop"] in ("end", "length", "user-turn")
    assert done["prompt_tokens"] > len("User: hello\nZ:") // 4


@pytest.mark.parametrize("body, message", [
    ({"checkpoint": "run/best.pt", "prompt": "hi", "mode": "poem"}, "mode"),
    ({"checkpoint": "run/best.pt", "prompt": "hi", "mode": "chat", "history": [{"user": 1}]}, "history"),
])
def test_bad_chat_requests_are_rejected(server, body, message):
    base, _ = server
    with pytest.raises(urllib.error.HTTPError) as e:
        post(base + "/api/generate", body)
    assert message in json.load(e.value)["error"]
