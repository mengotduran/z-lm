"""Open the Z playground in your browser: type a prompt, watch the model continue it.

Usage:
    python scripts/serve.py                  # then open http://localhost:8000
    python scripts/serve.py --port 8080 --checkpoints checkpoints

Every .pt file under --checkpoints shows up in the page's model menu (ckpt.pt and best.pt of
each run). New checkpoints appear when you reload the page, so you can keep this running
while training, and try the model as it improves.
The server logic lives in zlm/server.py and the page in zlm/playground.html.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from zlm.device import setup_device
from zlm.server import ModelStore, make_server


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoints", type=Path, default=Path("checkpoints"))
    parser.add_argument("--host", default="127.0.0.1",
                        help="127.0.0.1 = this computer only; 0.0.0.0 = also other devices on your network")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    setup = setup_device(args.device)
    store = ModelStore(args.checkpoints, setup)
    names = store.names()
    print(f"device={setup.device} | {len(names)} checkpoint(s) in {args.checkpoints}/"
          + (f": {', '.join(names)}" if names else " (train one first, then reload the page)"), flush=True)

    server = make_server(args.host, args.port, store)
    print(f"Z playground running at http://localhost:{args.port}  (Ctrl+C to stop)", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
