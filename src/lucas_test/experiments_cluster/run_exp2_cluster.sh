#!/bin/bash
#SBATCH --job-name=ray_exp2_qp2
#SBATCH --partition=LAD_geral
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --exclusive
#SBATCH --time=04:00:00
#SBATCH --output=/home/%u/ray-exp/exp2_%j.log
#SBATCH --error=/home/%u/ray-exp/exp2_%j.err

# ============================================================
# Experimento 2 — QP2: Batch Size
# Variável : batch_size = [32, 64, 128, 256]
# Fixos    : n_blocks=8, concurrency=2, dataset=500 docs
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
echo "  Job    : $SLURM_JOB_ID  (QP2 — Batch Size)"
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

# Garante que o dataset existe (gera se necessário)
/usr/bin/python3 - <<PYEOF
import json, random, os

OUTPUT = "dataset.jsonl"
if os.path.exists(OUTPUT):
    import subprocess
    n = int(subprocess.check_output(["wc", "-l", OUTPUT]).split()[0])
    if n >= 500:
        print(f"Dataset ja existe: {n} docs")
        exit()

NUM_DOCS = 500; CHUNK_SIZE = 200
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
    return {"id": idx, "text": " ".join(random.choice(TEMPLATES) for _ in range(CHUNK_SIZE))}
with open(OUTPUT, "w") as f:
    for i in range(NUM_DOCS):
        f.write(json.dumps(make_chunk(i), ensure_ascii=False) + "\n")
print(f"Dataset gerado: {NUM_DOCS} docs")
PYEOF

# ============================================================
# EXP 2 — QP2: Batch Size
# ============================================================
echo ""
echo "Iniciando QP2 — Batch Size..."
/usr/bin/python3 - <<PYEOF
import ray, time, psutil, json, statistics
import numpy as np

ray.init(address="${HEAD_IP}:${RAY_PORT}", ignore_reinit_error=True)

BATCH_SIZES = [32, 64, 128, 256]   # mesmos do notebook local
N_BLOCKS    = 8
CONCURRENCY = 2
N_RUNS      = 5
OUT         = "results_exp2_cluster.jsonl"

nodes      = [n for n in ray.nodes() if n["Alive"]]
total_cpus = int(sum(n["Resources"].get("CPU", 0) for n in nodes))
print(f"Cluster: {len(nodes)} nos, {total_cpus} CPUs")
print(f"Parametros: N_BLOCKS={N_BLOCKS}, CONCURRENCY={CONCURRENCY}, N_RUNS={N_RUNS}")

def make_model_class(batch_sz):
    class EmbeddingModel:
        def __init__(self):
            import torch
            from transformers import AutoTokenizer, AutoModel
            self.tokenizer = AutoTokenizer.from_pretrained("intfloat/multilingual-e5-small")
            self.model     = AutoModel.from_pretrained("intfloat/multilingual-e5-small")
            self.model.eval(); self.model.to("cpu")
            self._bs = batch_sz

        def __call__(self, batch):
            import torch
            texts = batch["text"].tolist()
            mem_before = psutil.virtual_memory().used / (1024**2)
            enc = self.tokenizer(texts, padding=True, truncation=True,
                                 max_length=128, return_tensors="pt")
            with torch.no_grad():
                out = self.model(**enc)
            emb = out.last_hidden_state[:, 0, :].cpu().numpy()
            norms = np.linalg.norm(emb, axis=1, keepdims=True)
            batch["embedding"] = emb / np.where(norms == 0, 1, norms)
            mem_after = psutil.virtual_memory().used / (1024**2)
            batch["mem_delta_mb"] = [mem_after - mem_before] * len(texts)
            return batch
    return EmbeddingModel

with open(OUT, "w") as out:
    for bs in BATCH_SIZES:
        times, mem_peaks = [], []
        print(f"\n-- batch_size={bs} --")

        for run in range(N_RUNS):
            ds = ray.data.read_json("dataset.jsonl").repartition(N_BLOCKS)
            t0 = time.time()
            result = ds.map_batches(
                make_model_class(bs), batch_size=bs,
                concurrency=CONCURRENCY, batch_format="numpy"
            ).take_all()
            elapsed = time.time() - t0
            times.append(elapsed)
            mem_peaks.append(psutil.virtual_memory().percent)
            print(f"  run {run+1}/{N_RUNS}: {elapsed:.2f}s | mem={mem_peaks[-1]:.1f}%")

        mean_t = statistics.mean(times)
        std_t  = statistics.stdev(times) if len(times) > 1 else 0.0
        rec = {
            "experiment"        : "exp2_batching",
            "environment"       : "cluster",
            "cluster_nodes"     : len(nodes),
            "cluster_total_cpus": total_cpus,
            "batch_size"        : bs,
            "n_blocks"          : N_BLOCKS,
            "concurrency"       : CONCURRENCY,
            "mean_time_s"       : round(mean_t, 4),
            "std_time_s"        : round(std_t, 4),
            "throughput_emb_s"  : round(len(result) / mean_t, 4),
            "mean_memory_pct"   : round(statistics.mean(mem_peaks), 2),
            "raw_times_s"       : [round(t, 4) for t in times],
        }
        out.write(json.dumps(rec) + "\n")
        out.flush()
        print(f"  >> {mean_t:.2f}s ± {std_t:.2f} | {rec['throughput_emb_s']:.2f} emb/s | mem={rec['mean_memory_pct']:.1f}%")

ray.shutdown()
print(f"\nResultados: {OUT}")
PYEOF

echo ""
echo "Encerrando Ray..."
srun --nodes=1 --ntasks=1 --nodelist="$HEAD"   "$RAY_BIN" stop --force
srun --nodes=1 --ntasks=1 --nodelist="$WORKER" "$RAY_BIN" stop --force

echo "============================================================"
echo "  QP2 finalizado! ~/ray-exp/results_exp2_cluster.jsonl"
echo "============================================================"