# JIT and Compilation in JAX

## 1. The problem

- **Per-op dispatch in eager mode.** Each JAX primitive is wrapped in a one-op `jit`, compiled (cached per op and input signature), and launched on its own. Python runs between every op. Bottleneck: dispatch latency, which dominates when ops are small.
- **No fusion across ops.** XLA sees one op at a time, so each intermediate is written to memory and read back by the next op. `sin(x) * 2 + 1` does 3 reads and 3 writes of the array eagerly, 1 and 1 fused. Bottleneck: memory bandwidth.
- **No whole-program optimization.** Layout, scheduling, and sharding or communication decisions are made per op, not across the function. Overlapping communication with compute across ops is not possible.

## 2. The idea

`jit` does not switch compilation on. It changes the **unit** compiled: one XLA program for the whole function instead of one per op.

First call with a new signature runs three stages, then execution:

| Stage | Input → output | Runs in |
|---|---|---|
| Trace | Python fn + abstract values (shape, dtype only) → jaxpr | Python, once |
| Lower | jaxpr → StableHLO | Python/XLA |
| Compile | HLO → machine code | XLA, usually the slowest |
| Execute | executable + data → output | every call |

"Compile time" in casual use means trace + lower + compile.

- **In a profiler trace:** setup runs on the Python thread before any execution starts. In an XProf trace of a first call, a `pjit cache_miss` box contained `pxla from_hlo` → `_cached_compile` → `compiler compile` (the compile stage), and execution boxes (`ExecuteReplicated`) came after. Frame names vary by JAX version, and trace and lower are not reliably separable in a trace. To time them separately, wrap `f.lower(x)` and `.compile()` in a timer.
- **Cache key:** shapes, dtypes, shardings, static args. A hit skips trace, lower and compile. A miss shows as `pjit cache_miss`.
- **Inspecting each stage:**

  ```python
  jax.make_jaxpr(f)(x)        # trace: the jaxpr
  lowered = f.lower(x)        # trace + lower
  lowered.as_text()           # StableHLO
  compiled = lowered.compile()  # XLA compile
  compiled.as_text()          # optimized HLO that actually runs
  compiled(x)                 # execution only
  ```

- **Ahead-of-time (AOT):** `f.lower(jax.ShapeDtypeStruct(shape, dtype, sharding=...)).compile()` builds the executable without data. Include the sharding, because the executable depends on layout. Call the returned object directly.
- **Two kinds of first-call cost.** Compilation is cached in the executable. Runtime state is not: thread pools, buffer allocation, cold caches, collective setup (on GPU, NCCL communicator creation). The second kind appears on the first execution even after `compile()`.

## 3. The cost

- **First-call cost:** trace, lower and compile run once per cache key before any execution (stages in section 2). A timing of the first call measures this, not the computation.
- **Recompilation:** any change to a cache-key component repeats all three stages. Varying input shapes, or passing varying Python values as static args, compiles repeatedly.
- **Trace-time side effects:** `print` and Python `if` on Python values run once during tracing, not on every call. Branching on traced values fails or needs `lax.cond`.
- **Baked-in constants:** Python values read during tracing become constants in the program. Changing them later has no effect on an already-compiled function.
- **Compile time grows with program size.** A larger jit unit trades a higher one-time cost for faster steady-state execution. Python loops are unrolled during tracing and inflate the program; `lax.scan` / `fori_loop` keep it small.
- **AOT is strict:** the compiled object raises an error on a shape, dtype or sharding mismatch instead of recompiling. Whether a later plain `f(x)` reuses an AOT compilation depends on the JAX version.
- **AOT does not remove warmup.** `compile()` removes the compilation cost (trace, lower, compile). It does not remove the runtime-state cost (thread pools, buffer allocation, cold caches, collective setup), which is paid on the first execution. To check whether that cost matters on your backend, time each of the first ~10 executions separately. If run 0 is slower than runs 1 to 9, there is a one-time runtime cost and a warmup execution is needed before timing. If all runs are about equal, warmup can be skipped.

### Caveats

| Issue | Effect | Fix |
|---|---|---|
| First call included in performance timing | Measures trace + lower + compile | Compile ahead of time or warm up |
| Runtime state not cached by `compile()` | First execution slower than later ones | 1 to 2 untimed executions |
| Shape or static-arg churn | Repeated compilation | Fix shapes, pad or bucket inputs |
| Side effects in jitted code | Run once, at trace time | Use `jax.debug.print` for per-call output |
| Eager mode in hot paths | Per-op dispatch and no fusion | Wrap the whole step in `jit` |