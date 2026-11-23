"""Generate a local TinyStories-style placeholder file when offline."""

import argparse
import random
from pathlib import Path

DELIM = "<|endoftext|>"
NAMES = ["Tim", "Lily", "Sam", "Anna", "Ben", "Mia", "Leo", "Zoe", "Max", "Eva"]
ANIMALS = ["cat", "dog", "bird", "rabbit", "fox", "bear", "duck", "frog"]
PLACES = ["park", "garden", "kitchen", "yard", "forest", "room", "hill", "lake"]
OBJECTS = ["ball", "book", "toy", "hat", "box", "map", "kite", "cup"]
COLORS = ["red", "blue", "green", "yellow", "pink", "white"]


def make_story(i: int, rng: random.Random) -> str:
    name = NAMES[i % len(NAMES)]
    friend = NAMES[(i + 3) % len(NAMES)]
    animal = ANIMALS[(i + 1) % len(ANIMALS)]
    place = PLACES[(i + 2) % len(PLACES)]
    obj = OBJECTS[(i + 4) % len(OBJECTS)]
    color = COLORS[(i + 5) % len(COLORS)]
    template = i % 4

    if template == 0:
        return (
            f"Once upon a time, there was a child named {name}. "
            f"{name} went to the {place} with a {color} {obj}. "
            f"A small {animal} came close and looked at the {obj}. "
            f"{name} smiled and shared the {obj} with the {animal}. "
            f"They played until the sun went down, and then {name} went home happy."
        )
    if template == 1:
        return (
            f"{name} and {friend} liked to play in the {place}. "
            f"One day they found a lost {animal} under a tree. "
            f"The {animal} was scared, so they spoke in a soft voice. "
            f"They gave it water and walked it to a safe spot. "
            f"The {animal} wagged its tail, and both kids felt proud."
        )
    if template == 2:
        return (
            f"In the {place}, {name} wanted to build something new. "
            f"{name} used a {obj}, a box, and some string from home. "
            f"It was hard work, but {name} did not give up. "
            f"When the {color} project was done, {friend} clapped and said it was great. "
            f"{name} learned that trying again can make good things happen."
        )
    return (
        f"On a quiet day, {name} read a book about a brave {animal}. "
        f"After reading, {name} went outside to the {place}. "
        f"There was a little problem: a {obj} was stuck high on a branch. "
        f"{name} asked {friend} for help, and together they solved it. "
        f"They laughed, cleaned up, and promised to help each other again."
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("TinyStories-1000.txt"),
        help="output txt path",
    )
    parser.add_argument("--count", type=int, default=1000, help="number of stories")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    stories = [make_story(i, rng) for i in range(args.count)]
    args.output.write_text(DELIM.join(stories) + DELIM, encoding="utf-8")
    size_mb = args.output.stat().st_size / 1024 / 1024
    print(
        f"wrote placeholder {args.output.resolve()} ({size_mb:.2f} MB, {len(stories)} stories)"
    )


if __name__ == "__main__":
    main()
