import os
import logging
from rich.logging import RichHandler
import time

# 🚨 关键修改：在 import jax 之前，先布下“结界”
os.environ["JAX_PLATFORMS"] = "cuda"
# 如果你还想屏蔽更底层的 C++ 编译警告，可以加上这句
# os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# 配置现代化的日志系统
logging.basicConfig(
    level="INFO",
    format="%(name)s : %(message)s",
    handlers=[RichHandler(rich_tracebacks=True)],
)

log = logging.getLogger("serl_trainer")
log.info("Starting learner loop")
log.warning("Grasp penalty detected!")
log.info(
    "[bold green]Loaded previous buffer data.[/bold green]", extra={"markup": True}
)

# 👇 现在结界布好了，再把 JAX 请进来，它就会乖乖听话了
import jax

# 生成一个初始的种子 (key)
rng = jax.random.PRNGKey(42)
print("分裂前类型:", type(rng))

rng, key = jax.random.split(rng)

print("分裂后 rng 类型:", type(rng))
print("分裂后 key 类型:", type(key))
