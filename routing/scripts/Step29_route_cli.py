"""CLI for the approved Step29 JSON tools."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from routing.src.route_interface import RouteToolInterface


def main() -> int:
    parser = argparse.ArgumentParser(description="Validated local thermal-route tools")
    parser.add_argument(
        "tool",
        choices=(
            "geocode_place",
            "snap_origin_destination",
            "route",
            "compare_routes",
            "summarize_route",
            "export_route_map",
        ),
    )
    parser.add_argument("--json", type=Path, help="UTF-8 JSON payload; otherwise read stdin")
    args = parser.parse_args()
    payload = json.loads(
        args.json.read_text(encoding="utf-8") if args.json else sys.stdin.read()
    )
    result = RouteToolInterface(ROOT).call(args.tool, payload)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
