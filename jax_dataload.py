from grain._src.python.load import data_loader
import jax
import jax.numpy as jnp

import grain.python as grain

from numpy.core import numeric
import tiktoken
from pathlib import Path


def load_stories_from_file(path, max_stories=None):
    text = Path(path).read_text(encoding="utf-8")
    stories = text.split("<|endoftext|>")
    stories = [story + "<|endoftext|>" for story in stories if story.strip()]
    if max_stories is not None:
        stories = stories[:max_stories]
    return stories


class StoryDataset:
    def __init__(self, stories, max_len, tokenizer):
        self.stories = stories
        self.max_len = max_len
        self.tokenizer = tokenizer
        self.end_token = tokenizer.encode(
            "<|endoftext|>", allowed_special={"<|endoftext|>"}
        )[0]

    def __len__(self):
        return len(self.stories)

    def __getitem__(self, idx):
        story = self.stories[idx]
        tokens = self.tokenizer.encode(story, allowed_special={"<|endoftext|>"})
        # 太长截断
        if len(tokens) > self.max_len:
            tokens = tokens[: self.max_len]
        # 太短补齐
        tokens.extend([0] * (self.max_len - len(tokens)))
        # 返回固定长度的
        return jnp.array(tokens)


shuffled_sampler = grain.IndexSampler(
    num_records=10,
    shuffle=True,
    seed=42,
    shard_options=grain.NoSharding(),
    num_epochs=1,
)


def print_sampler_example(sampler, name):
    print(f"\n{name}")
    for i, idx in enumerate(sampler):
        print(f"  {i}: {idx}")


print_sampler_example(shuffled_sampler, "Shuffled Sampler")

batch_op_keep = grain.Batch(batch_size=32, drop_remainder=False)


def create_dataloader(
    stories,
    tokenizer,
    max_len,
    batch_size,
    shuffle=False,
    num_epochs=1,
    seed=42,
    worker_count=0,
):
    dataset = StoryDataset(stories, max_len, tokenizer)
    esimated_batches = len(dataset) // batch_size
    sampler = grain.IndexSampler(
        num_records=len(dataset),  # 1,000 stories for this dataset
        shuffle=shuffle,
        seed=seed,
        shard_options=grain.NoSharding(),
        num_epochs=num_epochs,
    )
    dataloader = grain.DataLoader(
        data_source=dataset,
        sampler=sampler,
        operations=[
            grain.Batch(batch_size=batch_size, drop_remainder=True),
        ],
        worker_count=worker_count,
    )
    return dataloader, esimated_batches


def load_and_preprocess_data(
    file_path,
    max_stories=None,
    max_len=128,
    batch_size=32,
    shuffle=False,
    num_epochs=1,
    seed=42,
    worker_count=0,
    tokenizer=None,
):
    if tokenizer is None:
        tokenizer = tiktoken.get_encoding("gpt2")
    stories = load_stories_from_file(file_path, max_stories=max_stories)
    return create_dataloader(
        stories=stories,
        tokenizer=tokenizer,
        max_len=max_len,
        batch_size=batch_size,
        shuffle=shuffle,
        num_epochs=num_epochs,
        seed=seed,
        worker_count=worker_count,
    )


if __name__ == "__main__":
    tokenizer = tiktoken.get_encoding("gpt2")
    stories = load_stories_from_file("TinyStories-1000.txt", max_stories=100)
    dataloader, batches_per_batch = create_dataloader(
        stories=stories,
        tokenizer=tokenizer,
        max_len=128,
        batch_size=32,
        shuffle=True,
        num_epochs=1,
        seed=42,
        worker_count=0,
    )
    print(batches_per_batch)
    print(next(iter(dataloader)))
