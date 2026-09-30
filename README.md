# Ray: experimentos de geração de embeddings

Objetivo: medir como batch size, workers e particionamento influenciam a geração de embeddings com `Ray Data.map_batches()`.

O ambiente de testes ainda será definido. A base roda em CPU, em uma máquina/VM ou em um cluster Ray.

## Preparação

Use Python 3.11 ou 3.12. Após baixar e extrair o projeto:

```bash
cd ray-embedding-experiments
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip freeze > results/environment.txt
```

No Windows, crie o venv com `python -m venv .venv` e ative no PowerShell com `.venv\Scripts\Activate.ps1`.

A primeira execução precisa de internet para baixar o modelo `sentence-transformers/all-MiniLM-L6-v2`. Em cluster, instale as mesmas dependências e disponibilize o modelo em todos os nós antes de medir.

## Primeiro teste

Gere 1.000 textos e valide o pipeline com um worker:

```bash
python src/generate_dataset.py --rows 1000
python src/run_experiment.py --rows 1000 --repeats 1
```

O script gera embeddings reais, verifica a quantidade de saídas e salva os resultados em `results/`. Os textos são sintéticos e curtos: servem para começar; depois repetiremos os testes com um corpus real e comprimentos documentados.

## Testes planejados

| Teste | O que variar | O que manter fixo |
|---|---|---|
| Baseline | Uma configuração inicial | 1 worker, batch 64, 16 blocks |
| Batch size | 16, 32, 64, 128, 256, 512 | Dataset, workers e blocks |
| Workers | 1, 2, 4, 8 | Dataset, batch e blocks |
| Particionamento | 1, 2, 4, 8, 16, 32 blocks | Dataset, batch e workers |
| Volume | 1.000, 10.000, 50.000, 100.000 textos | Batch, workers e blocks |
| Batch × blocks | Combinações dos dois parâmetros | Dataset e workers |
| Local × distribuído | Uma máquina/VM versus cluster | Dataset, modelo, configuração e recursos totais comparáveis |

Workers são os actors que carregam o modelo. Blocks são as divisões do dataset; batches são os grupos de textos entregues ao modelo. Poucos blocks podem limitar o paralelismo, mesmo com vários workers.

## Executar os experimentos

Prepare 100.000 textos (os testes menores usam as primeiras linhas):

```bash
python src/generate_dataset.py --rows 100000
python src/run_experiment.py --experiment baseline
python src/run_experiment.py --experiment batch --workers 1
python src/run_experiment.py --experiment workers
python src/run_experiment.py --experiment blocks --workers 4
python src/run_experiment.py --experiment size --workers 4
python src/run_experiment.py --experiment matrix --workers 4
```

O teste de workers exige pelo menos 8 CPUs disponíveis no Ray. Os comandos com 4 workers exigem pelo menos 4. O script verifica essa disponibilidade. Comece pelo baseline e ajuste conforme o ambiente escolhido.

Para executar uma configuração específica:

```bash
python src/run_experiment.py --batch-size 128 --workers 2 --blocks 8 --rows 10000 --repeats 3 --label vm-teste
```

## Quando houver um cluster

Antes: escolher os nós, configurar comunicação entre eles, instalar as mesmas versões e iniciar o Ray head e os workers. Executar o repositório no head; não é necessário iniciar este script em cada worker.

```bash
python src/run_experiment.py --address auto --label cluster-2-nos --workers 2 --blocks 8
```

Repita localmente com os mesmos parâmetros e outra label. Registre CPU, RAM, sistema, número de nós e recursos reservados para o head. Várias VMs no mesmo servidor permitem estudar distribuição e overhead, mas não demonstram ganho de capacidade entre servidores físicos.

## Medição

- CSV: tempo em segundos, embeddings/s, parâmetros, repetição e versões.
- Cada configuração roda 3 vezes; use a mediana para comparar.
- O tempo inclui criação dos actors, carregamento/aquecimento do modelo e geração dos embeddings. `materialize()` força a execução, pois o Ray é lazy.
- Leitura e reparticionamento ficam fora desse tempo; o CSV registra `setup_s` separadamente. Cada actor usa 1 CPU lógica e 1 thread do PyTorch.
- CPU/RAM são métricas planejadas para coleta no ambiente escolhido; este script ainda não as coleta. GPU e banco vetorial ficam para uma etapa posterior.
- Evite outras cargas durante a medição. Guarde `environment.txt`, descrição do hardware e CSVs de cada ambiente. Em cluster, aqueça o cache do modelo em todos os nós.

## Arquivos

- `src/generate_dataset.py`: cria o CSV de textos.
- `src/embedder.py`: carrega o modelo e gera embeddings por batch.
- `src/run_experiment.py`: configura os testes e salva métricas.
- `data/`: entradas; `results/`: resultados. Dados gerados, venv e resultados são ignorados pelo Git. Para versionar resultados selecionados, use `git add -f results/ARQUIVO.csv`.

## Publicar no GitHub depois

Crie um repositório vazio no GitHub. Na pasta do projeto:

```bash
git init -b main
git add .
git commit -m "Base dos experimentos Ray"
git remote add origin https://github.com/SEU_USUARIO/ray-embedding-experiments.git
git push -u origin main
```

## Referências

- [Ray: map_batches](https://docs.ray.io/en/latest/data/api/doc/ray.data.Dataset.map_batches.html)
- [Ray: ActorPoolStrategy](https://docs.ray.io/en/latest/data/api/doc/ray.data.ActorPoolStrategy.html)
- [Exemplo de origem](https://github.com/Anant/example-ray-vector-embedding)
