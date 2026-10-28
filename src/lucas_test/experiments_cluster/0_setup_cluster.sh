#!/bin/bash
# =============================================================================
# SCRIPT 0 — SETUP DO CLUSTER RAY VIA SSH
# Execute este script UMA VEZ na máquina HEAD NODE (nó principal)
#
# Pré-requisito: acesso SSH sem senha configurado para todos os workers
# Uso: bash 0_setup_cluster.sh
# =============================================================================

set -e

# ─── CONFIGURAÇÃO: edite estas variáveis ────────────────────────────────────

HEAD_NODE_IP="192.168.1.10"          # IP da máquina head node (onde você está)
RAY_PORT=6379                         # Porta do Ray GCS
DASHBOARD_PORT=8265                   # Porta do dashboard

# IPs dos worker nodes (adicione ou remova conforme seu cluster)
WORKER_NODES=(
    "192.168.1.11"
    "192.168.1.12"
    "192.168.1.13"
)

SSH_USER="ubuntu"                     # Usuário SSH nos worker nodes
SSH_KEY="~/.ssh/id_rsa"              # Chave SSH privada
VENV_PATH="~/ray-exp/.venv"          # Caminho do venv em TODOS os nós
PROJECT_DIR="~/ray-exp"              # Diretório do projeto em todos os nós

# ─── FIM DA CONFIGURAÇÃO ────────────────────────────────────────────────────

echo "=========================================="
echo "  RAY CLUSTER SETUP"
echo "  Head node : $HEAD_NODE_IP"
echo "  Workers   : ${#WORKER_NODES[@]} nós"
echo "=========================================="

# ── Função auxiliar para rodar comandos nos workers ──────────────────────────
run_on_worker() {
    local node=$1
    local cmd=$2
    echo "[worker $node] $cmd"
    ssh -i "$SSH_KEY" -o StrictHostKeyChecking=no "$SSH_USER@$node" "$cmd"
}

# ── PASSO 1: Instalar dependências no HEAD NODE ──────────────────────────────
echo ""
echo "── [PASSO 1] Configurando HEAD NODE ──"

python3 -m venv "$VENV_PATH" --clear
source "$VENV_PATH/bin/activate"

pip install --quiet --upgrade pip
pip install --quiet \
    "numpy==1.26.4" \
    "pyarrow==14.0.2" \
    "ray[data]==2.9.3" \
    "pandas==2.1.4" \
    "sentence-transformers==2.6.1" \
    "transformers==4.39.3" \
    "torch==2.2.2" \
    "matplotlib==3.8.4" \
    "psutil==5.9.8"

echo "✓ Dependências instaladas no head node"

# ── PASSO 2: Instalar dependências em todos os WORKER NODES ─────────────────
echo ""
echo "── [PASSO 2] Configurando WORKER NODES ──"

for node in "${WORKER_NODES[@]}"; do
    echo ""
    echo "→ Configurando worker: $node"

    # Cria diretório e venv
    run_on_worker "$node" "mkdir -p $PROJECT_DIR"
    run_on_worker "$node" "python3 -m venv $VENV_PATH --clear"
    run_on_worker "$node" "
        source $VENV_PATH/bin/activate && \
        pip install --quiet --upgrade pip && \
        pip install --quiet \
            'numpy==1.26.4' \
            'pyarrow==14.0.2' \
            'ray[data]==2.9.3' \
            'pandas==2.1.4' \
            'sentence-transformers==2.6.1' \
            'transformers==4.39.3' \
            'torch==2.2.2' \
            'psutil==5.9.8'
    "
    echo "✓ Worker $node configurado"
done

# ── PASSO 3: Copiar scripts para todos os workers ────────────────────────────
echo ""
echo "── [PASSO 3] Copiando scripts para workers ──"

SCRIPTS_DIR="$(dirname "$0")"
for node in "${WORKER_NODES[@]}"; do
    scp -i "$SSH_KEY" -r "$SCRIPTS_DIR"/*.py "$SSH_USER@$node:$PROJECT_DIR/"
    echo "✓ Scripts copiados para $node"
done

# ── PASSO 4: Iniciar o HEAD NODE do Ray ─────────────────────────────────────
echo ""
echo "── [PASSO 4] Iniciando Ray HEAD NODE ──"

source "$VENV_PATH/bin/activate"

# Para qualquer instância anterior
ray stop --force 2>/dev/null || true
sleep 2

ray start \
    --head \
    --node-ip-address="$HEAD_NODE_IP" \
    --port="$RAY_PORT" \
    --dashboard-host="0.0.0.0" \
    --dashboard-port="$DASHBOARD_PORT" \
    --num-cpus=$(nproc) \
    --block &

sleep 5
echo "✓ Head node Ray iniciado em $HEAD_NODE_IP:$RAY_PORT"

# ── PASSO 5: Iniciar WORKER NODES ────────────────────────────────────────────
echo ""
echo "── [PASSO 5] Iniciando Ray WORKER NODES ──"

for node in "${WORKER_NODES[@]}"; do
    run_on_worker "$node" "
        source $VENV_PATH/bin/activate
        ray stop --force 2>/dev/null || true
        sleep 1
        ray start \
            --address='$HEAD_NODE_IP:$RAY_PORT' \
            --num-cpus=\$(nproc) \
            --block &
        sleep 3
        echo 'Worker Ray iniciado'
    "
    echo "✓ Worker $node conectado ao cluster"
done

# ── PASSO 6: Verificar cluster ────────────────────────────────────────────────
echo ""
echo "── [PASSO 6] Verificando cluster ──"
sleep 5

source "$VENV_PATH/bin/activate"
python3 -c "
import ray
ray.init(address='$HEAD_NODE_IP:$RAY_PORT', ignore_reinit_error=True)
nodes = ray.nodes()
alive = [n for n in nodes if n['Alive']]
print(f'Nós no cluster  : {len(nodes)}')
print(f'Nós ativos      : {len(alive)}')
for n in alive:
    resources = n['Resources']
    print(f'  {n[\"NodeManagerAddress\"]} → CPUs: {resources.get(\"CPU\",0):.0f}')
print(f'Dashboard       : http://$HEAD_NODE_IP:$DASHBOARD_PORT')
ray.shutdown()
"

echo ""
echo "=========================================="
echo "  CLUSTER PRONTO!"
echo "  Dashboard: http://$HEAD_NODE_IP:$DASHBOARD_PORT"
echo "  Próximo passo: python3 1_gerar_dataset.py"
echo "=========================================="