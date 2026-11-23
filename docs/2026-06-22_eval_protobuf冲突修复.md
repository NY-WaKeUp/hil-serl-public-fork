# 2026-06-22 · eval protobuf 冲突修复（RTLD_DEEPBIND）

## 问题

```
F0000 descriptor.cc:2519] Check failed: GeneratedDatabase()->Add(...)
File already exists in database: google/protobuf/descriptor.proto
```

## 根因

`libpyauboi5.so` 静态链接了 protobuf（自带 `InternalAddGeneratedFile` 符号），TF/JAX 也静态链接了 protobuf。两个 .so 各有一份 protobuf descriptor pool。

但默认 `RTLD_GLOBAL` 导致符号被**第一个加载者独占**。当 TF 先加载 → libpyauboi5 的 constructor 调用 `InternalAddGeneratedFile` 时走 PLT → 解析到 TF 的版本 → 写入 TF 的 descriptor pool → 重复注册 → crash。

用符号地址验证：

```
# libpyauboi5 有自己的一份 protobuf
$ nm -D python3.x/libpyauboi5.so | grep InternalAddGeneratedFile
000000000027b55e T _ZN6google8protobuf14DescriptorPool24InternalAddGeneratedFileEPKvi

# 没有动态链接系统 libprotobuf
$ ldd python3.x/libpyauboi5.so | grep protobuf
(空)
```

## 之前的方案（失败）

| 方案 | 失败原因 |
|------|----------|
| `usercustomize.py` + `ctypes.CDLL(RTLD_GLOBAL)` | `ctypes.CDLL` 和 Python `import` 是不同的 dlopen，constructor 只运行一次但 RTLD_GLOBAL 仍然导致符号抢夺 |
| `sitecustomize.py` + `import libpyauboi5` | 即使先加载 libpyauboi5，TF 后加载时 TF 的 constructor 会调用到 libpyauboi5 的 protobuf → 同样 crash（谁后加载谁死） |
| `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python` | 只影响 Python 层的 `google.protobuf`，不影响 C++ 静态 constructor |

## 修复（本次）

编译一个 30 行的 `.so` shim，用 `LD_PRELOAD` 拦截 `dlopen`，当加载 `libpyauboi5.so` 时自动添加 `RTLD_DEEPBIND`：

```c
// python3.x/dlopen_deepbind_pyaubo.c
void *dlopen(const char *filename, int flags) {
    static orig_dlopen_t real = NULL;
    if (!real) real = dlsym(RTLD_NEXT, "dlopen");
    if (filename && strstr(filename, "libpyauboi5"))
        flags |= RTLD_DEEPBIND;
    return real(filename, flags);
}
```

`RTLD_DEEPBIND` 强制 libpyauboi5 内部符号查找**优先使用自身**的 protobuf，而不是全局符号表中的 TF 版本。

| 无 RTLD_DEEPBIND | 有 RTLD_DEEPBIND |
|---|---|
| libpyauboi5 的 PLT → TF 的 `InternalAddGeneratedFile` → 写入 TF 的 pool → crash | libpyauboi5 的 PLT → **自己的** `InternalAddGeneratedFile` → 写入自己的 pool → 各自独立 |

编译产物：`python3.x/protobuf_deepbind.so`

## 脚本修改

两个脚本去掉了旧的 `sitecustomize.py` / `PYTHONPATH` / `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION`，改为单行：

```bash
export LD_PRELOAD="$REPO_ROOT/python3.x/protobuf_deepbind.so:${LD_PRELOAD:-}"
```

- `run_eval.sh`
- `run_actor.sh`

## 验证

```bash
# 冒烟测试通过
cd ~/hil-serl
LD_PRELOAD="$PWD/python3.x/protobuf_deepbind.so" PYTHONPATH="$PWD/python3.x" \
  python3 -c "import libpyauboi5; import jax; print('SUCCESS')"
```

输出：
```
[dlopen_deepbind_pyaubo] adding RTLD_DEEPBIND for .../libpyauboi5.so
SUCCESS
```

## 运行

```bash
cd ~/hil-serl/examples/experiments/aubo_pick_cube_real
source ~/hil-serl/.venv/bin/activate
export DISPLAY=:52.0
export CHECKPOINT_PATH=~/hil-serl/checkpoints/aubo_pick_cube_sim_rsa
export EVAL_STEP=175000 EVAL_N_TRAJS=1
bash run_eval.sh
```

- 不应再出现 `descriptor.proto` / `Aborted (core dumped)`
- 若 checkpoint 或机器人未就绪，会在更晚阶段报错（属预期）

| 优势 | 风险 |
|------|------|
| 不改 Python/robot SDK 任何代码 | `RTLD_DEEPBIND` 对 exception handling 可能有影响（实测 libpyauboi5 无 C++ exception 跨边界） |
| LD_PRELOAD 即插即用 | pragma-weak 的动态链接行为理论上被影响，但 libpyauboi5 不依赖 |
| 零进程开销 | 需要 gcc 编译一次 shim .so |

**置信度**：根因 —— **高**（nm/ldd 验证两个 .so 各有独立 protobuf，符号抢夺通过 PLT 发生）。修复 —— **高**（冒烟测试通过，二者在同一进程共存）。
