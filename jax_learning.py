from typing import Any
import os
from pathlib import Path

os.environ["DISPLAY"] = ":52.0"
os.environ["TIKTOKEN_CACHE_DIR"] = str(
    Path(__file__).resolve().parent / "tiktoken_cache"
)

import matplotlib.pyplot as plt
import seaborn as sns
import jax
import jax.numpy as jnp
from flax import nnx


class TokenAndPosEmbed(nnx.Module):
    def __init__(self, maxlen, vocab_size, embed_dim, *, rngs) -> None:
        super().__init__()
        self.token_emb = nnx.Embed(vocab_size, embed_dim, rngs=rngs)
        self.pos_emb = nnx.Embed(maxlen, embed_dim, rngs=rngs)

    def __call__(
        self,
        x,
    ) -> Any:
        seq_len = x.shape[1]
        pos = jnp.arange(seq_len)[None, :]
        return self.token_emb(x) + self.pos_emb(pos)


def causal_attn_mask(seq_len):
    return jnp.tril(jnp.ones((seq_len, seq_len)))


mask = causal_attn_mask(8)

plt.figure(figsize=(6, 5))
sns.heatmap(
    mask,
    cmap="Blues",
    vmin=0,
    vmax=1,
    square=True,
    linewidths=0.5,
    linecolor="white",
    cbar_kws={"label": "allowed"},
)
plt.xlabel("key position j")
plt.ylabel("query position i")
plt.title("Causal mask (j ≤ i)")
plt.tight_layout()

out = Path("causal_mask.png")
plt.savefig(out, dpi=150)
print(f"saved and opened: {out.resolve()}")


class TransformerBlock(nnx.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, *, rngs) -> None:
        super().__init__()
        self.attn = nnx.MultiHeadAttention(
            num_heads=num_heads,
            in_features=embed_dim,
            qkv_features=embed_dim,
            out_features=embed_dim,
            decode=False,
            rngs=rngs,
        )

    def __call__(self, x, mask=None):
        attn_out = self.attn(x, mask=mask)
        x = x + attn_out
        return x


class MiniGPT(nnx.Module):
    def __init__(
        self,
        max_len,
        vocab_size,
        embed_dim,
        num_heads,
        feed_forward_dim,
        num_transformer_blocks,
        *,
        rngs,
    ):
        self.maxlen = max_len
        self.embedding = TokenAndPosEmbed(
            maxlen=max_len, vocab_size=vocab_size, embed_dim=embed_dim, rngs=rngs
        )
        self.transformer_blocks = [
            TransformerBlock(embed_dim, num_heads, feed_forward_dim, rngs=rngs)
            for _ in range(num_transformer_blocks)
        ]
        self.output_layer = nnx.Linear(embed_dim, vocab_size, rngs=rngs)

    def causal_attn_mask(self, seq_len):
        return jnp.tril(jnp.ones((seq_len, seq_len)))

    def __call__(self, token_ids):
        seq_len = token_ids.shape[1]
        mask = self.causal_attn_mask(seq_len)
        x = self.embedding(token_ids)
        for block in self.transformer_blocks:
            x = block(x, mask=mask)
        return self.output_layer(x)


import tiktoken

tokenizer = tiktoken.get_encoding("gpt2")
print(tokenizer.n_vocab)
print(tokenizer.special_tokens_set)

model = MiniGPT(
    max_len=128,
    vocab_size=tokenizer.n_vocab,
    embed_dim=192,
    num_heads=6,
    feed_forward_dim=512,
    num_transformer_blocks=6,
    rngs=nnx.Rngs(0),
)
# print(model)
dummy_ids = jnp.arange(8, dtype=jnp.int32)[None, :]
# out_shape = nnx.eval_shape(lambda m, x: m(x), model, dummy_ids)
# n_params = sum(v.value.size for _, v in nnx.state(model, nnx.Param).flat_state())
# print(f"output shape: {out_shape.shape}")
# print(f"total parameters: {n_params:,}")
# print(nnx.tabulate(model.embedding, dummy_ids))
