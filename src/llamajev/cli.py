import argparse
import json
import logging
import sys
import time

from .config import Settings


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="llamajev", description="Jev-compatible /v1/systemone over llama-server")
    sub = parser.add_subparsers(dest="command", required=True)

    serve = sub.add_parser("serve", help="Run the API (launch llama-server, or --connect to one)")
    serve.add_argument("--connect", metavar="URL", help="Use a running llama-server instead of launching")
    serve.add_argument("--model", help="GGUF to launch llama-server with")
    serve.add_argument("--mmproj", help="Multimodal projector GGUF (enables image_url parts)")
    serve.add_argument("--llama-server", help="Path to the llama-server binary")
    serve.add_argument("--slots", type=int, help="llama-server -np (parallel slots)")
    serve.add_argument("--ctx", type=int, help="llama-server -c (context size)")
    serve.add_argument("--host")
    serve.add_argument("--port", type=int)
    serve.add_argument("--top-n", type=int, help="n_probs readout depth per branch")

    smoke = sub.add_parser("smoke", help="Send one three-question request and print timings")
    smoke.add_argument("url", nargs="?", default="http://127.0.0.1:8000")

    sub.add_parser("schema", help="Print the OpenAPI schema")

    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)

    if args.command == "serve":
        overrides = {
            k: v
            for k, v in {
                "backend_url": args.connect,
                "model": args.model,
                "mmproj": args.mmproj,
                "llama_server": args.llama_server,
                "n_slots": args.slots,
                "n_ctx": args.ctx,
                "host": args.host,
                "port": args.port,
                "top_n": args.top_n,
            }.items()
            if v is not None
        }
        settings = Settings(**overrides)
        if not args.connect:
            settings = settings.model_copy(update={"backend_url": f"http://127.0.0.1:{settings.backend_port}"})
        import uvicorn

        from .api import create_app

        uvicorn.run(create_app(settings, launch_backend=not args.connect), host=settings.host, port=settings.port)
    elif args.command == "schema":
        from .api import create_app

        print(json.dumps(create_app(Settings()).openapi(), indent=2))
    elif args.command == "smoke":
        run_smoke(args.url)


def run_smoke(url: str) -> None:
    import httpx

    payload = {
        "model": "jev-latest",
        "state": [
            {"role": "system", "content": "You are a support assistant."},
            {"role": "user", "content": "I was charged twice. Please refund the duplicate."},
        ],
        "questions": {
            "refund": {"type": "noul", "instructions": "Does the user request a refund?"},
            "department": {
                "type": "choice",
                "instructions": "Which department should handle this?",
                "criteria": {"billing": "Payments and refunds", "technical": "Software bugs", "shipping": "Delivery"},
            },
            "urgency": {"type": "score", "instructions": "How urgent is the request?", "criteria": ["Routine", "Urgent", "Emergency"]},
        },
    }
    with httpx.Client(base_url=url, timeout=120) as client:
        for attempt in range(60):
            health = client.get("/health")
            if health.status_code == 200:
                break
            time.sleep(1)
        else:
            print("server not healthy:", health.text, file=sys.stderr)
            sys.exit(1)
        print("health:", health.json())
        for i in range(3):
            t0 = time.perf_counter()
            r = client.post("/v1/systemone", json=payload)
            dt = (time.perf_counter() - t0) * 1000
            print(f"run {i}: HTTP {r.status_code} in {dt:.0f} ms | {r.headers.get('server-timing')} | cached={r.headers.get('x-llamajev-cached-tokens')} retries={r.headers.get('x-llamajev-readout-retries')}")
        print(json.dumps(r.json(), indent=2))
        if r.status_code != 200:
            sys.exit(1)
