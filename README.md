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
CC BY 4.0). About 246k synthetic rows, each with binary symptom columns and a
disease label. The CSV is not checked in; set its path in `configs/base.yaml`.

### Data handling

- Each row becomes a set of active symptoms plus the disease label.
- Rows are deduplicated on (symptom set, disease).
- Symptom sets that map to more than one disease are kept but counted; that
  count sets an accuracy ceiling and is reported with the results.
- Splits are made by unique symptom set, so no symptom set appears in more than
  one split. Stratified by disease where possible.
- Splits: train pool, validation (500), test (1000). The LoRA training set and
  the RAG knowledge base are the same rows (default 4000, optionally 20000).
- Splits are saved to disk with a fixed seed so every arm sees identical data.

## Experiment arms

| Arm | Weights | Inference prompt |
|-----|---------|------------------|
| 1. Base zero-shot | base | query only |
| 2. Base + RAG | base | top-k retrieved cases + query |
| 3. LoRA | base + adapter | query only |
| 4. LoRA + RAG | base + adapter | top-k retrieved cases + query |
| 5. Retrieval-only | none | majority disease among top-k neighbours |

Arm 5 is the control: it shows whether the LLM adds anything over retrieval.

Models: `TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T` (default) and
`Qwen/Qwen2.5-1.5B-Instruct`. Retrievers: `all-MiniLM-L6-v2` embeddings with
cosine similarity, and Jaccard similarity over symptom sets.

## Metrics

- **Primary**: exact-match accuracy after normalisation (lowercase, strip
  punctuation and whitespace).
- **Secondary**: semantic similarity, ROUGE-L, retrieval recall@k.
- **Retention**: MMLU accuracy on 5 subjects x 50 questions, scored by
  log-likelihood of the answer letters. Run for base and LoRA weights only.
  RAG leaves the weights unchanged, so the base model's retention score applies
  to the RAG arms as well.
- **Cost**: training wall-clock time, peak GPU memory, inference latency per
  example, mean prompt length in tokens.

Results are reported as mean and standard deviation over seeds 42, 43 and 44.

## Running

### On Google Colab

Open `notebooks/colab_runner.ipynb`. It clones the repo, installs
`requirements.txt`, mounts Google Drive for the output directory, reads the
W&B key from Colab secrets and runs the scripts below.

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

### Weights & Biases

Set `WANDB_API_KEY` in the environment (Colab secrets on Colab). Set
`wandb.enabled: false` in the config to run without it.

## Results

Placeholders until experiments are run. Mean ± std over 3 seeds, TinyLlama,
4000 training rows, k = 5.

| Arm | Exact match | Semantic sim | ROUGE-L | MMLU | Latency (ms) | Prompt tokens |
|-----|-------------|--------------|---------|------|--------------|---------------|
| Base zero-shot | – | – | – | – | – | – |
| Base + RAG | – | – | – | same as base | – | – |
| LoRA | – | – | – | – | – | – |
| LoRA + RAG | – | – | – | same as LoRA | – | – |
| Retrieval-only | – | – | – | n/a | – | n/a |

Accuracy ceiling from ambiguous symptom sets: –

## Limitations

- The dataset is synthetic; results may not transfer to real clinical notes.
- Models are small (about 1B parameters), chosen for Colab budgets.
- MMLU subset is 250 questions, so retention differences below a few points
  are within noise.
- Exact match penalises valid synonyms; the secondary metrics soften this.

## Layout

```
configs/        base.yaml, debug.yaml
src/medqa/      data, prompts, retrieval, train_lora, generate, metrics, evaluate
scripts/        prepare_data.sh, run_all.sh
notebooks/      colab_runner.ipynb, analysis.ipynb
tests/          test_data.py, test_metrics.py
```
