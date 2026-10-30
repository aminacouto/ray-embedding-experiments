#!/bin/bash
#SBATCH --job-name=ray_exp1_qp1
#SBATCH --partition=LAD_geral
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --exclusive
#SBATCH --time=04:00:00
#SBATCH --output=/home/%u/ray-exp/exp1_%j.log
#SBATCH --error=/home/%u/ray-exp/exp1_%j.err

# ============================================================
# Experimento 1 — QP1: Particionamento
# Variável : n_blocks = [4, 8, 16, 32]
# Fixos    : batch_size=64, concurrency=2, dataset=500 docs
# Replica os mesmos parâmetros do notebook local
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
echo "  Job    : $SLURM_JOB_ID  (QP1 — Particionamento)"
echo "  Head   : $HEAD ($HEAD_IP)"
echo "  Worker : $WORKER"
echo "============================================================"
[ -z "$HEAD_IP" ] && { echo "ERRO: IP nao resolvido"; exit 1; }

RAY_BIN="$HOME/.local/bin/ray"

# Inicia Ray head
srun --nodes=1 --ntasks=1 --nodelist="$HEAD" \
    "$RAY_BIN" start --head \
        --node-ip-address="$HEAD_IP" \
        --port=$RAY_PORT \
        --num-cpus=48 \
        --dashboard-host=0.0.0.0 \
        --block &
sleep 20

# Inicia Ray worker
srun --nodes=1 --ntasks=1 --nodelist="$WORKER" \
    "$RAY_BIN" start \
        --address="${HEAD_IP}:${RAY_PORT}" \
        --num-cpus=48 \
        --block &
sleep 20

# Verifica cluster
/usr/bin/python3 - <<PYEOF
import ray
ray.init(address="${HEAD_IP}:${RAY_PORT}", ignore_reinit_error=True)
nodes = [n for n in ray.nodes() if n["Alive"]]
print(f"Nos ativos: {len(nodes)} | CPUs totais: {int(sum(n['Resources'].get('CPU',0) for n in nodes))}")
ray.shutdown()
PYEOF

# Gera dataset igual ao local (500 docs, mesmo seed e templates)
echo ""
echo "Gerando dataset (500 docs — igual ao local)..."
/usr/bin/python3 - <<PYEOF
import json, random

NUM_DOCS   = 500
CHUNK_SIZE = 200
OUTPUT     = "dataset.jsonl"

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
    words = []
    for _ in range(CHUNK_SIZE):
        words.append(random.choice(TEMPLATES))
    return {"id": idx, "text": " ".join(words)}

with open(OUTPUT, "w") as f:
    for i in range(NUM_DOCS):
        f.write(json.dumps(make_chunk(i), ensure_ascii=False) + "\n")

print(f"Dataset gerado: {NUM_DOCS} docs -> {OUTPUT}")
PYEOF

# ============================================================
# EXP 1 — QP1: Particionamento
# ============================================================
echo ""
echo "Iniciando QP1 — Particionamento..."
/usr/bin/python3 - <<PYEOF
import ray, time, psutil, json, statistics
import numpy as np

ray.init(address="${HEAD_IP}:${RAY_PORT}", ignore_reinit_error=True)

BLOCKS_VALUES = [4, 8, 16, 32]   # mesmos do notebook local
BATCH_SIZE    = 64
CONCURRENCY   = 2
N_RUNS        = 5
OUT           = "results_exp1_cluster.jsonl"

class EmbeddingModel:
    def __init__(self):
        import torch
        from transformers import AutoTokenizer, AutoModel
        self.tokenizer = AutoTokenizer.from_pretrained("intfloat/multilingual-e5-small")
        self.model     = AutoModel.from_pretrained("intfloat/multilingual-e5-small")
        self.model.eval(); self.model.to("cpu")

    def __call__(self, batch):
        import torch
        texts = batch["text"].tolist()
        enc = self.tokenizer(texts, padding=True, truncation=True,
                             max_length=128, return_tensors="pt")
        with torch.no_grad():
            out = self.model(**enc)
        emb = out.last_hidden_state[:, 0, :].cpu().numpy()
        norms = np.linalg.norm(emb, axis=1, keepdims=True)
        batch["embedding"] = emb / np.where(norms == 0, 1, norms)
        return batch

nodes      = [n for n in ray.nodes() if n["Alive"]]
total_cpus = int(sum(n["Resources"].get("CPU", 0) for n in nodes))
print(f"Cluster: {len(nodes)} nos, {total_cpus} CPUs")
print(f"Parametros: BATCH_SIZE={BATCH_SIZE}, CONCURRENCY={CONCURRENCY}, N_RUNS={N_RUNS}")

with open(OUT, "w") as out:
    for n_blocks in BLOCKS_VALUES:
        times, mem_peaks = [], []
        print(f"\n-- n_blocks={n_blocks} --")

        for run in range(N_RUNS):
            ds = ray.data.read_json("dataset.jsonl").repartition(n_blocks)
            t0 = time.time()
            result = ds.map_batches(
                EmbeddingModel, batch_size=BATCH_SIZE,
                concurrency=CONCURRENCY, batch_format="numpy"
            ).take_all()
            elapsed = time.time() - t0
            times.append(elapsed)
            mem_peaks.append(psutil.virtual_memory().percent)
            print(f"  run {run+1}/{N_RUNS}: {elapsed:.2f}s | mem={mem_peaks[-1]:.1f}%")

        mean_t = statistics.mean(times)
        std_t  = statistics.stdev(times) if len(times) > 1 else 0.0
        rec = {
            "experiment"        : "exp1_particionamento",
            "environment"       : "cluster",
            "cluster_nodes"     : len(nodes),
            "cluster_total_cpus": total_cpus,
            "n_blocks"          : n_blocks,
            "batch_size"        : BATCH_SIZE,
            "concurrency"       : CONCURRENCY,
            "mean_time_s"       : round(mean_t, 4),
            "std_time_s"        : round(std_t, 4),
            "throughput_emb_s"  : round(len(result) / mean_t, 4),
            "memory_pct"        : round(statistics.mean(mem_peaks), 2),
            "raw_times_s"       : [round(t, 4) for t in times],
        }
        out.write(json.dumps(rec) + "\n")
        out.flush()
        print(f"  >> {mean_t:.2f}s ± {std_t:.2f} | {rec['throughput_emb_s']:.2f} emb/s")

ray.shutdown()
print(f"\nResultados: {OUT}")
PYEOF

echo ""
echo "Encerrando Ray..."
srun --nodes=1 --ntasks=1 --nodelist="$HEAD"   "$RAY_BIN" stop --force
srun --nodes=1 --ntasks=1 --nodelist="$WORKER" "$RAY_BIN" stop --force

echo "============================================================"
echo "  QP1 finalizado! ~/ray-exp/results_exp1_cluster.jsonl"
echo "============================================================"