import jax
import jax.numpy as jnp
import jax.sharding as shd
import time

jax.config.update("jax_platform_name", "cpu")
jax.config.update("jax_num_cpu_devices", 8)
assert len(jax.devices()) == 8

mesh = jax.make_mesh(axis_sizes=(8,), axis_names=('X'))

def P(*args):
  return shd.NamedSharding(mesh, shd.PartitionSpec(*args))

@jax.jit
@jax.shard_map(out_specs=shd.PartitionSpec(None), mesh=mesh, check_vma=False)
def all_gather(array: jax.Array):
  return jax.lax.all_gather(array, 'X', axis=0, tiled=True)

@jax.jit
@jax.shard_map(out_specs=shd.PartitionSpec(None), mesh=mesh, check_vma=False)
def all_reduce(array: jax.Array):
  reduced: jax.Array = jax.lax.psum(array, axis_name='X')
  assert reduced.shape == array.shape
  return reduced

@jax.jit
@jax.shard_map(out_specs=shd.PartitionSpec('X'), mesh=mesh, check_vma=False)
def reduce_scatter(array: jax.Array):
  reduced: jax.Array = jax.lax.psum_scatter(array, axis_name='X', tiled=True)
  assert reduced.shape == (256, 8192)
  return reduced

@jax.jit
@jax.shard_map(out_specs=shd.PartitionSpec('X'), mesh=mesh, check_vma=False)
def all_to_all(array: jax.Array):
  return jax.lax.all_to_all(array, axis_name='X', split_axis=1, concat_axis=0, tiled=True)

A = jnp.zeros((2048, 8192), dtype=jnp.bfloat16)
sharded_A = jax.device_put(A, P('X', None))

# use perf count for, compile, first execution time, and average run-time
def profile(name: str, fn: jax.stages.Wrapped, array: jax.Array):
  print(name)
  start = time.perf_counter()
  compiled_gather = fn.lower(array).compile()
  print(f"compile time: {(time.perf_counter() - start) * 1000:.2f}ms")

  start = time.perf_counter()
  gathered_A = compiled_gather(array)
  gathered_A.block_until_ready()
  print(f"first run: {(time.perf_counter() - start) * 1000:.2f}ms")

  total = 0
  for _ in range(10):
    start = time.perf_counter()
    gathered_A = compiled_gather(array)
    gathered_A.block_until_ready()
    total += (time.perf_counter() - start) * 1000

  average = round(total / 10, 2)
  print(f"average run time: {average}ms\n")
  return average

ag_average = profile("AllGather", all_gather, sharded_A)
ar_average = profile("AllReduce", all_reduce, A)
rs_average = profile("ReduceScatter", reduce_scatter, A)
aa_average = profile("AllToAll", all_to_all, sharded_A)

print("baseline: AllGather")
print(f"AllReduce ratio: {ar_average / ag_average:.2f}")
print(f"ReduceScatter ratio: {rs_average / ag_average:.2f}")
print(f"AllToAll ratio: {aa_average / ag_average:.2f}")