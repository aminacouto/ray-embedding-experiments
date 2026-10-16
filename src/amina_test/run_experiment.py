"""Executa baseline ou uma varredura e salva métricas por repetição."""
import argparse
import csv
from datetime import datetime, timezone
from itertools import product
from pathlib import Path
import platform
import random
import time

from amina_test.generate_dataset import positive


def configurations(args):
    batches, workers, blocks, sizes = [args.batch_size], [args.workers], [args.blocks], [args.rows]
    if args.experiment in ("batch", "matrix"):
        batches = [16, 32, 64, 128, 256, 512]
    if args.experiment == "workers":
        workers = [1, 2, 4, 8]
    if args.experiment in ("blocks", "matrix"):
        blocks = [1, 2, 4, 8, 16, 32]
    if args.experiment == "size":
        sizes = [1000, 10000, 50000, 100000]
    configs = list(product(batches, workers, blocks, sizes))
    random.Random(42).shuffle(configs)
    return configs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", choices=["baseline", "batch", "workers", "blocks", "matrix", "size"], default="baseline")
    parser.add_argument("--data", type=Path, default=Path("data/texts.csv"))
    parser.add_argument("--rows", type=positive, default=10000)
    parser.add_argument("--batch-size", type=positive, default=64)
    parser.add_argument("--workers", type=positive, default=1)
    parser.add_argument("--blocks", type=positive, default=16)
    parser.add_argument("--repeats", type=positive, default=3)
    parser.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--address", default=None, help="auto para um cluster iniciado; omitir para Ray local")
    parser.add_argument("--label", default="local", help="Identificação do ambiente/rodada")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    # Importação adiada: --help funciona sem instalar dependências.
    import ray
    from amina_test.embedder import Embedder

    with args.data.open(newline="", encoding="utf-8") as file:
        records = list(csv.DictReader(file))
    if not records or not {"id", "text"}.issubset(records[0]):
        parser.error("O CSV deve conter id e text e pelo menos uma linha.")
    configs = configurations(args)
    if max(c[3] for c in configs) > len(records):
        parser.error("CSV insuficiente: gere mais textos com generate_dataset.py.")
    if any(blocks > size for _, _, blocks, size in configs):
        parser.error("O número de blocks não pode exceder o número de textos.")

    ray.init(address=args.address)
    try:
        resources = ray.cluster_resources()
        if max(c[1] for c in configs) > resources.get("CPU", 0):
            parser.error("Workers excedem as CPUs disponíveis no Ray. Ajuste o ambiente/configuração.")
        # Download fora das medições no processo coordenador.
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(args.model, device="cpu")
        del model
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        output = args.output or Path(f"results/{args.experiment}_{stamp}.csv")
        if output.exists():
            parser.error("O arquivo de saída já existe; escolha outro nome.")
        output.parent.mkdir(parents=True, exist_ok=True)
        fields = ["utc", "label", "experiment", "repeat", "batch_size", "workers", "requested_blocks", "input_blocks", "rows", "time_s", "embeddings_s", "setup_s", "model", "ray_version", "python_version", "ray_cpus", "nodes"]
        with output.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=fields)
            writer.writeheader()
            for batch, workers, blocks, size in configs:
                setup_start = time.perf_counter()
                ds = ray.data.from_items(records[:size]).repartition(blocks).materialize()
                setup_s = time.perf_counter() - setup_start
                for repetition in range(1, args.repeats + 1):
                    start = time.perf_counter()
                    result = ds.map_batches(
                        Embedder, batch_size=batch, batch_format="numpy",
                        compute=ray.data.ActorPoolStrategy(size=workers),
                        num_cpus=1, fn_constructor_kwargs={"model": args.model},
                    ).materialize()
                    elapsed = time.perf_counter() - start
                    if result.count() != size:
                        raise RuntimeError("Quantidade de embeddings diferente da entrada.")
                    writer.writerow(dict(
                        utc=datetime.now(timezone.utc).isoformat(), label=args.label,
                        experiment=args.experiment, repeat=repetition, batch_size=batch,
                        workers=workers, requested_blocks=blocks, input_blocks=ds.num_blocks(),
                        rows=size, time_s=elapsed, embeddings_s=size / elapsed,
                        setup_s=setup_s, model=args.model, ray_version=ray.__version__,
                        python_version=platform.python_version(), ray_cpus=resources.get("CPU", 0),
                        nodes=sum(node["Alive"] for node in ray.nodes()),
                    ))
                    file.flush()
                    print(f"batch={batch} workers={workers} blocks={blocks} rows={size}: {elapsed:.2f}s / {size / elapsed:.2f} embeddings/s")
                    del result
                del ds
        print(f"Resultados: {output}")
    finally:
        ray.shutdown()


if __name__ == "__main__":
    main()
