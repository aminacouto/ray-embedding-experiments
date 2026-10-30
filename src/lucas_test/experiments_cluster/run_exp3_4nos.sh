#!/bin/bash
#SBATCH --job-name=ray_exp3_4nos
#SBATCH --partition=LAD_geral
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --exclusive
#SBATCH --time=04:00:00
#SBATCH --output=/home/%u/ray-exp/exp3_%j.log
#SBATCH --error=/home/%u/ray-exp/exp3_%j.err

export PATH="$HOME/.local/bin:/usr/bin:$PATH"
export PYTHONPATH="$HOME/.local/lib/python3.8/site-packages"
cd ~/ray-exp

NODES=($(scontrol show hostnames "$SLURM_JOB_NODELIST"))
HEAD=${NODES[0]}
WORKER1=${NODES[1]}
WORKER2=${NODES[2]}
WORKER3=${NODES[3]}

HEAD_IP=$(getent hosts "$HEAD" 2>/dev/null | awk '{print $1}' | grep -v ':' | head -1)
if [ -z "$HEAD_IP" ]; then
    HEAD_IP=$(nslookup "$HEAD" 2>/dev/null | awk '/Address/ && !/#/{print $2}' | grep -v ':' | head -1)
fi
RAY_PORT=6379

echo "============================================================"
echo "  Job ID  : $SLURM_JOB_ID"
echo "  Head    : $HEAD ($HEAD_IP)"
echo "  Workers : $WORKER1 $WORKER2 $WORKER3"
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
sleep 25

for WORKER in $WORKER1 $WORKER2 $WORKER3; do
    srun --nodes=1 --ntasks=1 --nodelist="$WORKER" \
        "$RAY_BIN" start \
            --address="${HEAD_IP}:${RAY_PORT}" \
            --num-cpus=48 \
            --block &
    sleep 8
done
sleep 20

/usr/bin/python3 - <<PYEOF
import ray
ray.init(address="${HEAD_IP}:${RAY_PORT}", ignore_reinit_error=True)
nodes = [n for n in ray.nodes() if n["Alive"]]
print(f"Nos ativos: {len(nodes)}")
for n in nodes:
    print(f"  {n['NodeManagerHostname']} — {int(n['Resources'].get('CPU',0))} CPUs")
print(f"Total CPUs: {int(sum(n['Resources'].get('CPU',0) for n in nodes))}")
ray.shutdown()
PYEOF

echo "Gerando dataset..."
/usr/bin/python3 - <<PYEOF
import json, random, string
random.seed(42)
def rand_text(n=200):
    words = [''.join(random.choices(string.ascii_lowercase, k=random.randint(3,10))) for _ in range(n)]
    return ' '.join(words)
with open("dataset_small.jsonl", "w") as f:
    for i in range(500):
        f.write(json.dumps({"id": i, "text": rand_text()}) + "\n")
print("500 docs gerados")
PYEOF

echo "Rodando baseline (N_BLOCKS=16)..."
/usr/bin/python3 - <<PYEOF
import ray, time, json
import numpy as np
ray.init(address="${HEAD_IP}:${RAY_PORT}", ignore_reinit_error=True)
DATASET="dataset_small.jsonl"; BATCH_SIZE=32; N_BLOCKS=16
class EmbeddingModel:
    def __init__(self):
        import torch
        from transformers import AutoTokenizer, AutoModel
        self.tokenizer = AutoTokenizer.from_pretrained("intfloat/multilingual-e5-small")
        self.model = AutoModel.from_pretrained("intfloat/multilingual-e5-small")
        self.model.eval(); self.model.to("cpu")
    def __call__(self, batch):
        import torch
        texts = batch["text"].tolist()
        enc = self.tokenizer(texts, padding=True, truncation=True, max_length=128, return_tensors="pt")
        with torch.no_grad(): out = self.model(**enc)
        emb = out.last_hidden_state[:, 0, :].cpu().numpy()
        norms = np.linalg.norm(emb, axis=1, keepdims=True)
        batch["embedding"] = emb / np.where(norms==0,1,norms)
        return batch
ds = ray.data.read_json(DATASET).repartition(N_BLOCKS)
t0 = time.time()
result = ds.map_batches(EmbeddingModel, batch_size=BATCH_SIZE, concurrency=1, batch_format="numpy").take_all()
baseline_time = time.time() - t0
with open("baseline_result.json","w") as f:
    json.dump({"baseline_time_s": round(baseline_time,4), "n_docs": len(result)}, f)
print(f"Baseline: {baseline_time:.2f}s | {len(result)} docs")
ray.shutdown()
PYEOF

echo "Iniciando experimento 4 nos..."
/usr/bin/python3 - <<PYEOF
import ray, time, json, statistics, psutil, socket
import numpy as np
ray.init(address="${HEAD_IP}:${RAY_PORT}", ignore_reinit_error=True)
DATASET="dataset_small.jsonl"; N_BLOCKS=16; BATCH_SIZE=32; N_RUNS=3; OUT="results_exp3_4nos.jsonl"
with open("baseline_result.json") as f:
    baseline_time = json.load(f)["baseline_time_s"]
nodes = [n for n in ray.nodes() if n["Alive"]]
total_cpus = int(sum(n["Resources"].get("CPU",0) for n in nodes))
concurrency_values = [1, 2, 4, 8, 16]
print(f"Cluster: {len(nodes)} nos, {total_cpus} CPUs | N_BLOCKS={N_BLOCKS} | Baseline={baseline_time:.2f}s")
class EmbeddingModel:
    def __init__(self):
        import torch
        from transformers import AutoTokenizer, AutoModel
        self.tokenizer = AutoTokenizer.from_pretrained("intfloat/multilingual-e5-small")
        self.model = AutoModel.from_pretrained("intfloat/multilingual-e5-small")
        self.model.eval(); self.model.to("cpu")
        self.hostname = socket.gethostname()
    def __call__(self, batch):
        import torch
        texts = batch["text"].tolist()
        enc = self.tokenizer(texts, padding=True, truncation=True, max_length=128, return_tensors="pt")
        with torch.no_grad(): out = self.model(**enc)
        emb = out.last_hidden_state[:, 0, :].cpu().numpy()
        norms = np.linalg.norm(emb, axis=1, keepdims=True)
        batch["embedding"] = emb / np.where(norms==0,1,norms)
        batch["worker_host"] = [self.hostname]*len(texts)
        return batch
with open(OUT,"w") as out:
    for n_workers in concurrency_values:
        times=[]; throughputs=[]; mem_peaks=[]
        print(f"\n-- concurrency={n_workers} --")
        for run in range(N_RUNS):
            ds = ray.data.read_json(DATASET).repartition(N_BLOCKS)
            t0 = time.time()
            result = ds.map_batches(EmbeddingModel, batch_size=BATCH_SIZE, concurrency=n_workers, batch_format="numpy").take_all()
            elapsed = time.time()-t0
            hosts = {r.get("worker_host","?") for r in result}
            times.append(elapsed); throughputs.append(len(result)/elapsed); mem_peaks.append(psutil.virtual_memory().percent)
            print(f"  run {run+1}/{N_RUNS}: {elapsed:.2f}s | hosts: {hosts}")
        mean_t=statistics.mean(times); std_t=statistics.stdev(times) if len(times)>1 else 0.0
        speedup_real=baseline_time/mean_t; speedup_ideal=float(n_workers)
        efficiency=speedup_real/speedup_ideal*100
        rec={"n_workers":n_workers,"n_blocks":N_BLOCKS,"cluster_nodes":len(nodes),"cluster_total_cpus":total_cpus,
             "mean_time_s":round(mean_t,4),"std_time_s":round(std_t,4),
             "speedup_real":round(speedup_real,4),"speedup_ideal":round(speedup_ideal,4),
             "efficiency_pct":round(efficiency,2),"raw_times_s":[round(t,4) for t in times]}
        out.write(json.dumps(rec)+"\n"); out.flush()
        print(f"  >> {mean_t:.2f}s | speedup {speedup_real:.2f}x (ideal {speedup_ideal:.0f}x) | eficiencia {efficiency:.0f}%")
ray.shutdown()
print(f"Resultados: {OUT}")
PYEOF

echo "Gerando graficos..."
/usr/bin/python3 ~/ray-exp/6_analise_exp3_4nos.py

RAY_BIN="$HOME/.local/bin/ray"
srun --nodes=1 --ntasks=1 --nodelist="$HEAD"    "$RAY_BIN" stop --force
srun --nodes=1 --ntasks=1 --nodelist="$WORKER1" "$RAY_BIN" stop --force
srun --nodes=1 --ntasks=1 --nodelist="$WORKER2" "$RAY_BIN" stop --force
srun --nodes=1 --ntasks=1 --nodelist="$WORKER3" "$RAY_BIN" stop --force

echo "============================================================"
echo "  Job finalizado! Resultados: ~/ray-exp/results_exp3_4nos.jsonl"
echo "============================================================"
