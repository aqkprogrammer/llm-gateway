"""``llm-gateway`` console entry point."""

from __future__ import annotations

import argparse
from pathlib import Path

import uvicorn
from dotenv import load_dotenv

from llm_gateway.config import Settings


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the LLM gateway.")
    parser.add_argument("--host", default=None, help="bind host (default: GATEWAY_HOST)")
    parser.add_argument("--port", type=int, default=None, help="bind port (default: GATEWAY_PORT)")
    parser.add_argument("--reload", action="store_true", help="auto-reload on code changes")
    args = parser.parse_args()

    if Path(".env").exists():
        load_dotenv(".env", override=False)
    settings = Settings()
    uvicorn.run(
        "llm_gateway.main:create_app",
        factory=True,
        host=args.host or settings.host,
        port=args.port or settings.port,
        reload=args.reload,
        log_config=None,
        access_log=False,
        proxy_headers=True,
        timeout_graceful_shutdown=20,
    )


if __name__ == "__main__":
    main()
