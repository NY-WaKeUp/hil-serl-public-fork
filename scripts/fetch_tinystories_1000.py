"""Download the first 1000 TinyStories from Hugging Face into TinyStories-1000.txt."""

import argparse
import urllib.request
from pathlib import Path

DEFAULT_URL = (
    "https://huggingface.co/datasets/roneneldan/TinyStories/"
    "resolve/main/TinyStories-train.txt"
)
DELIM = "<|endoftext|>"


def fetch_stories(url: str, target: int) -> tuple[list[str], int]:
    buf = ""
    stories: list[str] = []
    bytes_read = 0
    print(f"streaming from {url}")
    print(
        "note: only the beginning of the 1.79 GB file is fetched, not the full dataset"
    )
    with urllib.request.urlopen(url, timeout=120) as resp:
        while len(stories) < target:
            chunk = resp.read(1024 * 1024)
            if not chunk:
                break
            bytes_read += len(chunk)
            buf += chunk.decode("utf-8", errors="replace")
            parts = buf.split(DELIM)
            buf = parts[-1]
            for part in parts[:-1]:
                story = part.strip()
                if story:
                    stories.append(story)
                    if len(stories) % 100 == 0:
                        print(
                            f"  downloaded {bytes_read / 1024 / 1024:.2f} MB, got {len(stories)} stories"
                        )
                    if len(stories) >= target:
                        break
    if len(stories) < target:
        raise RuntimeError(f"only got {len(stories)} stories from {url}")
    return stories, bytes_read


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("TinyStories-1000.txt"),
        help="output txt path",
    )
    parser.add_argument("--count", type=int, default=1000, help="number of stories")
    parser.add_argument("--url", type=str, default=DEFAULT_URL, help="source txt url")
    args = parser.parse_args()

    stories, bytes_read = fetch_stories(args.url, args.count)
    args.output.write_text(DELIM.join(stories) + DELIM, encoding="utf-8")
    size_mb = args.output.stat().st_size / 1024 / 1024
    print(f"streamed {bytes_read / 1024 / 1024:.2f} MB from Hugging Face")
    print(f"wrote {args.output.resolve()} ({size_mb:.2f} MB, {len(stories)} stories)")
    print(f"first story ({len(stories[0])} chars):")
    print(stories[0])


if __name__ == "__main__":
    main()
