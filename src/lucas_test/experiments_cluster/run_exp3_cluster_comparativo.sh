#!/bin/bash
#SBATCH --job-name=ray_exp3_qp3
#SBATCH --partition=LAD_geral
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --exclusive
#SBATCH --time=04:00:00
#SBATCH --output=/home/%u/ray-exp/exp3c_%j.log
#SBATCH --error=/home/%u/ray-exp/exp3c_%j.err

# ============================================================
# Experimento 3 — QP3: Paralelismo (versão comparativa)
# Variável : concurrency = [1, 2, 4]  (igual ao local)
# Fixos    : n_blocks=8, batch_size=64, dataset=500 docs
# Replica os EXATOS parâmetros do notebook local para comparativo
# ============================================================

export PATH="$HOME/.local/bin:/usr/bin:$PATH"
export PYTHONPATH="$HOME/.local/lib/python3.8/site-packages"
cd ~/ray-exp

NODES=($(scontrol show hostnames "$SLURM_JOB_NODELIST"))
HEAD=${NODES[0]}
WORKER=${NODES[1]}

HEAD_IP=$(getent hosts "$HEAD" 2>/dev/null | awk '{print $1}' | grep -v ':' | head -1)
if [ -z "$HEAD_IP" ]; then
    HEAD_IP=$(nslookup "$HEAD" 2>/dev/null | awk '/Address/ && !/#/{print $2}' | grep -v ':' | head -1)
fi
RAY_PORT=6379

echo "============================================================"
echo "  Job    : $SLURM_JOB_ID  (QP3 — Paralelismo comparativo)"
echo "  Head   : $HEAD ($HEAD_IP)"
echo "  Worker : $WORKER"
echo "============================================================"
[ -z "$HEAD_IP" ] && { echo "ERRO: IP nao resolvido"; exit 1; }

RAY_BIN="$HOME/.local/bin/ray"

srun --nodes=1 --ntasks=1 --nodelist="$HEAD" \
    "$RAY_BIN" start --head \
        --node-ip-address="$HEAD_IP" \
        --port=$RAY_PORT \
        --num-cpus=48 \
        --dashboard-host=0.0.0.0 \
        --block &
sleep 20

srun --nodes=1 --ntasks=1 --nodelist="$WORKER" \
    "$RAY_BIN" start \
        --address="${HEAD_IP}:${RAY_PORT}" \
        --num-cpus=48 \
        --block &
sleep 20

/usr/bin/python3 - <<PYEOF
import ray
ray.init(address="${HEAD_IP}:${RAY_PORT}", ignore_reinit_error=True)
nodes = [n for n in ray.nodes() if n["Alive"]]
print(f"Nos ativos: {len(nodes)} | CPUs totais: {int(sum(n['Resources'].get('CPU',0) for n in nodes))}")
ray.shutdown()
PYEOF

# Garante dataset (500 docs, mesmo seed do local)
/usr/bin/python3 - <<PYEOF
import json, random, os
OUTPUT = "dataset.jsonl"
if os.path.exists(OUTPUT):
    import subprocess
    n = int(subprocess.check_output(["wc", "-l", OUTPUT]).split()[0])
    if n >= 500:
        print(f"Dataset ja existe: {n} docs"); exit()
TEMPLATES = [
    "O processamento distribuido permite escalar cargas de trabalho de IA",
    "Ray Data organiza datasets em blocks armazenados no object store",
    "O modelo multilingual-e5-small gera vetores de 384 dimensoes",
    "Embeddings sao representacoes vetoriais densas de textos e documentos",
    "O particionamento de dados melhora o throughput em pipelines RAG",
    "Sistemas distribuidos exigem balanceamento de carga entre workers",
    "A busca vetorial aproximada permite recuperacao eficiente de documentos",
    "O overhead de comunicacao limita o ganho linear com mais workers",
]
random.seed(42)
def make_chunk(idx):
    return {"id": idx, "text": " ".join(random.choice(TEMPLATES) for _ in range(200))}
with open(OUTPUT, "w") as f:
    for i in range(500):
        f.write(json.dumps(make_chunk(i), ensure_ascii=False) + "\n")
print("Dataset gerado: 500 docs")
PYEOF

# ============================================================
# EXP 3 — QP3 Comparativo (mesmos params do local)
# ============================================================
echo ""
echo "Iniciando QP3 — Paralelismo (comparativo com local)..."
/usr/bin/python3 - <<PYEOF
import ray, time, psutil, json, statistics, socket
import numpy as np

ray.init(address="${HEAD_IP}:${RAY_PORT}", ignore_reinit_error=True)

# === EXATOS mesmos parâmetros do notebook local ===
CONCURRENCY_VALUES = [1, 2, 4]
N_BLOCKS   = 8
BATCH_SIZE = 64
N_RUNS     = 5
OUT        = "results_exp3_cluster_comparativo.jsonl"

nodes      = [n for n in ray.nodes() if n["Alive"]]
total_cpus = int(sum(n["Resources"].get("CPU", 0) for n in nodes))
print(f"Cluster: {len(nodes)} nos, {total_cpus} CPUs")
print(f"Parametros: N_BLOCKS={N_BLOCKS}, BATCH_SIZE={BATCH_SIZE}, N_RUNS={N_RUNS}")
print(f"Testando concurrency: {CONCURRENCY_VALUES}")

class EmbeddingModel:
    def __init__(self):
        import torch
        from transformers import AutoTokenizer, AutoModel
        self.tokenizer = AutoTokenizer.from_pretrained("intfloat/multilingual-e5-small")
        self.model     = AutoModel.from_pretrained("intfloat/multilingual-e5-small")
        self.model.eval(); self.model.to("cpu")
        self.hostname  = socket.gethostname()

    def __call__(self, batch):
        import torch
        texts = batch["text"].tolist()
        enc = self.tokenizer(texts, padding=True, truncation=True,
                             max_length=128, return_tensors="pt")
        with torch.no_grad():
            out = self.model(**enc)
        emb = out.last_hidden_state[:, 0, :].cpu().numpy()
        norms = np.linalg.norm(emb, axis=1, keepdims=True)
        batch["embedding"]   = emb / np.where(norms == 0, 1, norms)
        batch["worker_host"] = [self.hostname] * len(texts)
        return batch

baseline_time = None

with open(OUT, "w") as out:
    for n_workers in CONCURRENCY_VALUES:
        times, mem_peaks = [], []
        print(f"\n-- concurrency={n_workers} --")

        for run in range(N_RUNS):
            ds = ray.data.read_json("dataset.jsonl").repartition(N_BLOCKS)
            t0 = time.time()
            result = ds.map_batches(
                EmbeddingModel, batch_size=BATCH_SIZE,
                concurrency=n_workers, batch_format="numpy"
            ).take_all()
            elapsed = time.time() - t0
            hosts = {r.get("worker_host", "?") for r in result}
            times.append(elapsed)
            mem_peaks.append(psutil.virtual_memory().percent)
            print(f"  run {run+1}/{N_RUNS}: {elapsed:.2f}s | hosts: {hosts}")

        mean_t = statistics.mean(times)
        std_t  = statistics.stdev(times) if len(times) > 1 else 0.0

        if n_workers == 1:
            baseline_time = mean_t

        speedup      = baseline_time / mean_t
        ideal_speedup = float(n_workers)
        efficiency   = speedup / ideal_speedup * 100

        rec = {
            "experiment"        : "exp3_paralelismo",
            "environment"       : "cluster",
            "cluster_nodes"     : len(nodes),
            "cluster_total_cpus": total_cpus,
            "n_workers"         : n_workers,
            "n_blocks"          : N_BLOCKS,
            "batch_size"        : BATCH_SIZE,
            "mean_time_s"       : round(mean_t, 4),
            "std_time_s"        : round(std_t, 4),
            "throughput_emb_s"  : round(len(result) / mean_t, 4),
            "speedup"           : round(speedup, 4),
            "ideal_speedup"     : round(ideal_speedup, 4),
            "efficiency_pct"    : round(efficiency, 2),
            "mean_memory_pct"   : round(statistics.mean(mem_peaks), 2),
            "raw_times_s"       : [round(t, 4) for t in times],
        }
        out.write(json.dumps(rec) + "\n")
        out.flush()
        print(f"  >> {mean_t:.2f}s ± {std_t:.2f} | speedup {speedup:.2f}x "
              f"(ideal {ideal_speedup:.0f}x) | efic={efficiency:.0f}%")

ray.shutdown()
print(f"\nResultados: {OUT}")
PYEOF

echo ""
echo "Encerrando Ray..."
srun --nodes=1 --ntasks=1 --nodelist="$HEAD"   "$RAY_BIN" stop --force
srun --nodes=1 --ntasks=1 --nodelist="$WORKER" "$RAY_BIN" stop --force

echo "============================================================"
echo "  QP3 comparativo finalizado!"
echo "  ~/ray-exp/results_exp3_cluster_comparativo.jsonl"
echo "============================================================"