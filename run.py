"""One-click launcher for AI Trading Bot Web Dashboard.

Usage:
    python run.py              → Start on http://localhost:8000
    python run.py --port 9000  → Custom port
    python run.py --public     → Accessible from other devices on your network
"""

import argparse
import os
import socket
import sys
import webbrowser
from pathlib import Path


def get_local_ip():
    """Get the machine's local IP for network access."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"


def check_dependencies():
    """Check and install missing dependencies."""
    required = ["fastapi", "uvicorn", "pandas", "numpy", "plotly", "pyyaml", "sqlalchemy", "yfinance"]
    missing = []
    for pkg in required:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)

    if missing:
        print(f"Installing missing packages: {', '.join(missing)}")
        os.system(f"{sys.executable} -m pip install {' '.join(missing)}")


def main():
    parser = argparse.ArgumentParser(description="AI Trading Bot — Web Dashboard")
    parser.add_argument("--port", "-p", type=int, default=8000, help="Port number (default: 8000)")
    parser.add_argument("--public", action="store_true", help="Allow access from other devices")
    parser.add_argument("--no-browser", action="store_true", help="Don't open browser automatically")
    args = parser.parse_args()

    # Ensure we're in the project root
    os.chdir(Path(__file__).parent)

    # Check dependencies
    check_dependencies()

    # Create required directories
    for d in ["data", "models", "logs", "charts"]:
        Path(d).mkdir(exist_ok=True)

    host = "0.0.0.0" if args.public else "127.0.0.1"
    local_ip = get_local_ip()

    print()
    print("=" * 55)
    print("  AI Trading Bot — Web Dashboard")
    print("=" * 55)
    print(f"  Local:   http://localhost:{args.port}")
    if args.public:
        print(f"  Network: http://{local_ip}:{args.port}")
        print(f"  (accessible from phone/other devices on same WiFi)")
    print()
    print("  Features:")
    print("    - Live NSE stock charts with indicators")
    print("    - Trading signals (Trend, MeanReversion, Momentum)")
    print("    - Strategy backtesting with equity curve")
    print("    - Paper trading portfolio")
    print("  API docs: http://localhost:{}/docs".format(args.port))
    print("=" * 55)
    print()

    # Open browser
    if not args.no_browser:
        webbrowser.open(f"http://localhost:{args.port}")

    # Start server
    import uvicorn
    uvicorn.run("web.server:app", host=host, port=args.port, reload=False)


if __name__ == "__main__":
    main()
