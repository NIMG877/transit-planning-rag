"""Web UI 启动入口。"""

import argparse

import uvicorn


def main() -> None:
    parser = argparse.ArgumentParser(description="启动本地 Web UI 服务")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="监听地址，默认 127.0.0.1")
    parser.add_argument("--port", type=int, default=8000, help="端口，默认 8000")
    parser.add_argument("--reload", action="store_true", help="开发模式热更新")
    args = parser.parse_args()

    uvicorn.run(
        "webapp.api:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )


if __name__ == "__main__":
    main()
