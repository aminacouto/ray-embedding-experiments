import ray

print("Ray:", ray.__version__)

ray.init(
    include_dashboard=False,
    num_cpus=2
)

@ray.remote
def teste(x):
    return x * 2

print(ray.get([teste.remote(i) for i in range(5)]))

ray.shutdown()