#!/bin/bash
#SBATCH --job-name=ray_exp3
#SBATCH --partition=LAD_geral
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --exclusive
#SBATCH --time=04:00:00
#SBATCH --output=/home/%u/ray-exp/exp3_%j.log
#SBATCH --error=/home/%u/ray-exp/exp3_%j.err

export PATH="$HOME/.local/bin:$PATH"
cd ~/ray-exp

# ── Identifica os nós alocados ────────────────────────────────────
NODES=($(scontrol show hostnames "$SLURM_JOB_NODELIST"))
HEAD=${NODES[0]}
WORKER=${NODES[1]}
HEAD_IP=$(getent hosts "$HEAD" | awk '{print $1}')
RAY_PORT=6379

echo "============================================================"
echo "  Job ID  : $SLURM_JOB_ID"
echo "  Head    : $HEAD ($HEAD_IP)"
echo "  Worker  : $WORKER"
echo "  Nós     : $SLURM_JOB_NUM_NODES"
echo "============================================================"

# ── PARTE 1: inicia Ray head no nó 0 via srun ────────────────────
srun --nodes=1 --ntasks=1 --nodelist="$HEAD" \
    ray start --head \
        --node-ip-address="$HEAD_IP" \
        --port=$RAY_PORT \
        --num-cpus=48 \
        --dashboard-host=0.0.0.0 \
        --block &

HEAD_PID=$!
echo "Ray head iniciado (PID $HEAD_PID), aguardando..."
sleep 20

# ── PARTE 2: conecta worker no nó 1 via srun ────────────────────
srun --nodes=1 --ntasks=1 --nodelist="$WORKER" \
    ray start \
        --address="${HEAD_IP}:${RAY_PORT}" \
        --num-cpus=48 \
        --block &

WORKER_PID=$!
echo "Ray worker iniciado (PID $WORKER_PID), aguardando conexão..."
sleep 25

# ── Verifica cluster ─────────────────────────────────────────────
python3 - <<PYEOF
import ray
ray.init(address="auto", ignore_reinit_error=True)
nodes = [n for n in ray.nodes() if n["Alive"]]
print(f"Nós ativos no cluster Ray: {len(nodes)}")
for n in nodes:
    print(f"  {n['NodeManagerHostname']} — {int(n['Resources'].get('CPU',0))} CPUs")
ray.shutdown()
PYEOF

# ── Baseline (1 worker — necessário pro speedup) ─────────────────
echo ""
echo "Rodando baseline..."
python3 - <<PYEOF
import ray, time, json

ray.init(address="auto", ignore_reinit_error=True)

DATASET    = "dataset.jsonl"
BATCH_SIZE = 64

class EmbeddingModel:
    def __init__(self):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer("intfloat/multilingual-e5-small")
    def __call__(self, batch):
        batch["embedding"] = self.model.encode(
            batch["text"].tolist(), batch_size=BATCH_SIZE,
            show_progress_bar=False, normalize_embeddings=True)
        return batch

ds = ray.data.read_json(DATASET).repartition(8)
t0 = time.time()
result = ds.map_batches(EmbeddingModel, batch_size=BATCH_SIZE,
                         concurrency=1, batch_format="numpy").take_all()
baseline_time = time.time() - t0

rec = {
    "baseline_time_s"  : round(baseline_time, 4),
    "n_docs"           : len(result),
    "throughput_emb_s" : round(len(result) / baseline_time, 2),
}
with open("baseline_result.json", "w") as f:
    json.dump(rec, f)

print(f"Baseline: {baseline_time:.2f}s | {rec['throughput_emb_s']} emb/s")
ray.shutdown()
PYEOF

# ── Experimento 3: varia concurrency ─────────────────────────────
echo ""
echo "Iniciando Experimento 3..."
python3 - <<PYEOF
import ray, time, json, statistics, psutil, socket

ray.init(address="auto", ignore_reinit_error=True)

DATASET    = "dataset.jsonl"
N_BLOCKS   = 8
BATCH_SIZE = 64
N_RUNS     = 5
OUT        = "results_exp3_paralelismo.jsonl"

with open("baseline_result.json") as f:
    baseline_time = json.load(f)["baseline_time_s"]

nodes      = [n for n in ray.nodes() if n["Alive"]]
total_cpus = int(sum(n["Resources"].get("CPU", 0) for n in nodes))
concurrency_values = sorted({1, max(1, total_cpus // 4),
                              max(1, total_cpus // 2), total_cpus})

print(f"Cluster: {len(nodes)} nos, {total_cpus} CPUs totais")
print(f"Testando concurrency: {concurrency_values}")
print(f"Baseline de referencia: {baseline_time:.2f}s")

class EmbeddingModel:
    def __init__(self):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer("intfloat/multilingual-e5-small")
        self.hostname = socket.gethostname()
    def __call__(self, batch):
        batch["embedding"] = self.model.encode(
            batch["text"].tolist(), batch_size=BATCH_SIZE,
            show_progress_bar=False, normalize_embeddings=True)
        batch["worker_host"] = [self.hostname] * len(batch["text"])
        return batch

all_results = []
with open(OUT, "w") as out:
    for n_workers in concurrency_values:
        times, throughputs, mem_peaks = [], [], []
        print(f"\n-- concurrency={n_workers} --")

        for run in range(N_RUNS):
            ds = ray.data.read_json(DATASET).repartition(N_BLOCKS)
            t0 = time.time()
            result = ds.map_batches(
                EmbeddingModel, batch_size=BATCH_SIZE,
                concurrency=n_workers, batch_format="numpy"
            ).take_all()
            elapsed = time.time() - t0
            hosts   = {r.get("worker_host", "?") for r in result}
            times.append(elapsed)
            throughputs.append(len(result) / elapsed)
            mem_peaks.append(psutil.virtual_memory().percent)
            print(f"  run {run+1}/{N_RUNS}: {elapsed:.2f}s | "
                  f"{len(result)/elapsed:.1f} emb/s | hosts: {hosts}")

        mean_t        = statistics.mean(times)
        std_t         = statistics.stdev(times) if len(times) > 1 else 0.0
        speedup_real  = baseline_time / mean_t
        speedup_ideal = float(n_workers)
        efficiency    = speedup_real / speedup_ideal * 100

        rec = {
            "n_workers"             : n_workers,
            "n_blocks"              : N_BLOCKS,
            "batch_size"            : BATCH_SIZE,
            "cluster_nodes"         : len(nodes),
            "cluster_total_cpus"    : total_cpus,
            "mean_time_s"           : round(mean_t, 4),
            "std_time_s"            : round(std_t, 4),
            "mean_throughput_emb_s" : round(statistics.mean(throughputs), 2),
            "std_throughput_emb_s"  : round(statistics.stdev(throughputs) if len(throughputs) > 1 else 0, 2),
            "speedup_real"          : round(speedup_real, 4),
            "speedup_ideal"         : round(speedup_ideal, 4),
            "efficiency_pct"        : round(efficiency, 2),
            "mean_ram_peak_pct"     : round(statistics.mean(mem_peaks), 2),
            "raw_times_s"           : [round(t, 4) for t in times],
        }
        out.write(json.dumps(rec) + "\n")
        out.flush()
        all_results.append(rec)

        print(f"  >> {mean_t:.2f}s | speedup {speedup_real:.2f}x "
              f"(ideal {speedup_ideal:.0f}x) | eficiencia {efficiency:.0f}%")

ray.shutdown()
print(f"\nResultados salvos em: {OUT}")
PYEOF

# ── Gráficos ─────────────────────────────────────────────────────
echo ""
echo "Gerando graficos..."
python3 ~/ray-exp/6_analise_exp3.py

# ── Encerra Ray em todos os nós ───────────────────────────────────
echo ""
echo "Encerrando Ray..."
srun --nodes=1 --ntasks=1 --nodelist="$HEAD"   ray stop --force
srun --nodes=1 --ntasks=1 --nodelist="$WORKER" ray stop --force

echo ""
echo "============================================================"
echo "  Job finalizado!"
echo "  Resultados: ~/ray-exp/results_exp3_paralelismo.jsonl"
echo "  Graficos  : ~/ray-exp/grafico_*.png"
echo "============================================================"