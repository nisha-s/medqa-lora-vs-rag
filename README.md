# medqa-lora-vs-rag

Symptom-to-disease prediction with small LLMs: does knowledge stored in the
weights (LoRA fine-tuning) beat knowledge retrieved at inference time (RAG over
past cases)?

This continues [nisha-s/lora-medical-qa](https://github.com/nisha-s/lora-medical-qa),
which fine-tuned TinyLlama with LoRA on the same task. That repo's
symptom-to-text conversion, prompt format and LoRA defaults are reused here.

## Research question

For symptom-to-disease prediction with ~1B parameter models, how does LoRA
fine-tuning compare with retrieval-augmented generation (RAG) over past cases
on:

1. **Medical accuracy** on held-out symptom sets.
2. **General-knowledge retention** on a fixed MMLU subset.
3. **Cost**: training time, inference latency and prompt length.

## Dataset

[Disease and symptoms dataset 2023](https://data.mendeley.com/) (Mendeley Data,
CC BY 4.0). 246,945 synthetic rows, 382 binary symptom columns, 773 diseases.
The CSV is not checked in; place it at `data/disease_symptoms.csv` or set
`data.csv_path` in the config.

### Data handling

- Each row becomes a set of active symptoms plus the disease label.
- Rows are deduplicated on (symptom set, disease).
- Symptom sets that map to more than one disease are kept but counted; that
  count sets an accuracy ceiling and is reported with the results.
- Splits are made by unique symptom set, so no symptom set appears in more than
  one split. Stratified by disease where possible.
- Splits: train pool, validation (500), test (1000). The LoRA training set and
  the RAG knowledge base are the same rows (default 4000, optionally 20000).
  Sizes count unique symptom sets; an ambiguous set keeps all of its rows in
  one split, so row counts run about 10% higher.
- Test and validation are drawn first, so they are identical for every
  training-set size.
- Splits are saved to disk with a fixed seed so every arm sees identical data.
  With seed 42: 246,945 raw rows, 189,647 after dedup, 12,634 ambiguous
  symptom sets, test ceiling 0.899.

## Experiment arms

| Arm | Weights | Inference prompt |
|-----|---------|------------------|
| 1. Base zero-shot | base | query only |
| 2. Base + RAG | base | top-k retrieved cases + query |
| 3. LoRA | base + adapter | query only |
| 4. LoRA + RAG | base + adapter | top-k retrieved cases + query |
| 5. Retrieval-only | none | majority disease among top-k neighbours |

Arm 5 is the control: it shows whether the LLM adds anything over retrieval.

LoRA training computes the loss on the disease tokens only, not on the symptom
prompt. The notebook's full-sequence loss is available with
`lora.loss_on_prompt: true`.

Models: `TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T` (default) and
`Qwen/Qwen2.5-1.5B-Instruct`. Retrievers: `all-MiniLM-L6-v2` embeddings with
cosine similarity, and Jaccard similarity over symptom sets.

## Metrics

- **Primary**: exact-match accuracy after normalisation (lowercase, strip
  punctuation and whitespace).
- **Secondary**: semantic similarity, ROUGE-L, retrieval recall@k.
- **Retention**: MMLU accuracy on 5 subjects x 50 questions (anatomy,
  clinical knowledge, college biology, high school mathematics, world
  religions), zero-shot, scored by comparing the log-likelihood of the four
  answer letters after `Answer:`. The subset is sampled once with a fixed
  seed and cached to `outputs/mmlu_subset.csv`, so every model scores the
  same questions. Run for base and LoRA weights only. RAG leaves the weights
  unchanged, so the base model's retention score applies to the RAG arms as
  well. Results go to `outputs/mmlu.csv` with per-subject accuracy in each
  run's JSON.
- **Cost**: training wall-clock time, peak GPU memory, inference latency per
  example, mean prompt length in tokens.

Results are reported as mean and standard deviation over seeds 42, 43 and 44.

## Running

### On Google Colab

Open `notebooks/colab_runner.ipynb` in Colab with a GPU runtime. It clones
the repo, installs `requirements.txt`, mounts Google Drive, reads the W&B key
from Colab secrets (or disables logging if there is none) and runs the
scripts with `configs/colab.yaml`, which puts data and outputs under
`MyDrive/medqa-lora-vs-rag/`. Put the CSV at
`MyDrive/medqa-lora-vs-rag/data/disease_symptoms.csv` first. The notebook
starts with a one-step smoke run (`configs/colab_debug.yaml`) before the
real one. After a disconnect, re-run from the top: finished runs are skipped
and training resumes from its last checkpoint.

`notebooks/analysis.ipynb` reads the aggregated results and produces the
tables and figures below.

### Locally (debugging only)

The `debug.yaml` config uses 50 train rows, 20 test rows and 1 training step,
so it runs on a laptop CPU.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
bash scripts/prepare_data.sh configs/debug.yaml
bash scripts/run_all.sh configs/debug.yaml
pytest tests/
```

Or with Docker:

```bash
docker build -t medqa .
docker run --rm -v "$PWD:/app" medqa bash scripts/run_all.sh configs/debug.yaml
```

Device is picked automatically: CUDA, then MPS, then CPU. No bitsandbytes.

### Training outputs and resuming

Each LoRA run writes its adapter and a `train_stats.json` (steps, losses,
wall-clock time, peak GPU memory) to `outputs/adapters/<model>_seed<seed>/`.
Checkpoints go to `outputs/checkpoints/` every `lora.save_steps` steps; if a
Colab session dies, re-running the same command resumes from the last one.
A run whose adapter already exists is skipped, so `run_all.sh` is safe to
re-run.

### Weights & Biases

Set `WANDB_API_KEY` in the environment (Colab secrets on Colab). Set
`wandb.enabled: false` in the config to run without it.

## Results

Test split: 1000 symptom sets, 1112 rows, accuracy ceiling 0.899 (seed 42).
LLM arms are placeholders until the Colab runs finish; mean ± std over 3 seeds,
TinyLlama, 4000 training symptom sets, k = 5.

| Arm | Exact match | Semantic sim | ROUGE-L | MMLU | Latency (ms) | Prompt tokens |
|-----|-------------|--------------|---------|------|--------------|---------------|
| Base zero-shot | – | – | – | – | – | – |
| Base + RAG (Jaccard, k=5) | – | – | – | same as base | – | – |
| LoRA | – | – | – | – | – | – |
| LoRA + RAG (Jaccard, k=5) | – | – | – | same as LoRA | – | – |
| Retrieval-only (Jaccard, k=5) | 0.727 | 0.832 | 0.745 | n/a | 0.03 | n/a |

Retrieval-only control over k, exact match / recall@k (deterministic, one run):

| k | Jaccard | MiniLM embedding |
|---|---------|------------------|
| 1 | 0.664 / 0.664 | 0.661 / 0.661 |
| 3 | 0.686 / 0.849 | 0.658 / 0.806 |
| 5 | 0.727 / 0.906 | 0.649 / 0.871 |
| 10 | 0.725 / 0.951 | 0.630 / 0.931 |

Jaccard is the default retriever: it beats MiniLM at every k, and majority
vote over MiniLM neighbours gets worse as k grows. Recall@k shows how often
the right disease is somewhere in the retrieved cases, which is what the
RAG arms can exploit.

## Limitations

- The dataset is synthetic; results may not transfer to real clinical notes.
- Models are small (about 1B parameters), chosen for Colab budgets.
- MMLU subset is 250 questions, so retention differences below a few points
  are within noise.
- Exact match penalises valid synonyms; the secondary metrics soften this.

## Layout

```
configs/        base.yaml, debug.yaml, colab.yaml, colab_debug.yaml
src/medqa/      data, prompts, retrieval, train_lora, generate, metrics, evaluate
scripts/        prepare_data.sh, run_all.sh
notebooks/      colab_runner.ipynb, analysis.ipynb
tests/          test_data.py, test_metrics.py
```
