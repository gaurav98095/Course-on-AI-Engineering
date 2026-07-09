"""Call the deployed RAG endpoint like any user would.

Run:  python client.py "Why does an aircraft stall at the critical angle of attack?"
      python client.py "What does this instrument do?" --image data/sample-query-instrument.jpg
      python client.py "..." --url https://<your-exposed-url>
"""

import argparse
import base64
import time

import requests


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("question")
    ap.add_argument("--image", help="optional query image path")
    ap.add_argument("--url", default="http://localhost:8000", help="server base URL")
    args = ap.parse_args()

    payload = {"question": args.question}
    if args.image:
        payload["image_b64"] = base64.b64encode(open(args.image, "rb").read()).decode()

    t0 = time.perf_counter()
    r = requests.post(f"{args.url.rstrip('/')}/predict", json=payload, timeout=180)
    wall = time.perf_counter() - t0
    r.raise_for_status()
    out = r.json()

    print(out["answer"])
    print(f"\nsources: {', '.join(out['sources'])}")
    print(f"server time: {out['seconds']}s | wall time incl. network: {wall:.2f}s "
          f"| overhead: {wall - out['seconds']:.2f}s")


if __name__ == "__main__":
    main()
