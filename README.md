# medqa-lora-vs-rag

Given a patient's symptoms, predict the disease. The question this repo asks is
where the knowledge should live for a small language model: baked into the
weights through LoRA fine-tuning, or pulled in at inference time with RAG over
past cases. There's also a third option people tend to skip past, which is to
drop the language model entirely and just look up similar cases, so that's in
here too as a control.

It builds on [nisha-s/lora-medical-qa](https://github.com/nisha-s/lora-medical-qa),
an earlier project that fine-tuned TinyLlama on the same data. The
symptom-to-text conversion, the `### Symptoms / ### Disease` prompt format, and
the LoRA hyperparameters all carry over from there.

## Research question

For symptom-to-disease prediction with roughly 1B-parameter models, how does
LoRA fine-tuning stack up against RAG over past cases on three things:

1. **Medical accuracy** on held-out symptom sets.
2. **General-knowledge retention**, measured on a fixed MMLU subset.
3. **Cost**: training time, inference latency, and prompt length.

## Dataset

[Disease and symptoms dataset 2023](https://data.mendeley.com/) (Mendeley Data,
CC BY 4.0): 246,945 synthetic rows, 382 binary symptom columns, and 773
diseases. The CSV isn't checked in. Drop it at `data/disease_symptoms.csv` or
point `data.csv_path` at wherever you keep it.

### How the data is handled

Each row turns into a set of active symptoms and a disease label. The raw file
has a lot of exact repeats, so the first step is to drop duplicate
(symptom set, disease) pairs. Beyond that, 12,634 symptom sets map to more than
one disease. Those stay in, but they get counted, because they put a hard cap on
how well any method can do: if the same symptoms point to two diseases,
exact-match accuracy can't reach 1.0. With seed 42 that ceiling works out to
0.899 on the test set.

Splitting is done by unique symptom set, not by row, so the same set never shows
up in two splits. That's the part the earlier notebook got wrong, and it's the
main reason the numbers here are lower (and more honest) than before. Splits are
stratified by disease where the counts allow it. Test (1000 sets) and validation
(500 sets) are carved off first, which means they stay fixed no matter how big
the training pool is. The LoRA training set and the RAG knowledge base are the
exact same rows (4000 by default, or 20000 if you set it), so neither method
gets to see data the other doesn't.

One wrinkle: split sizes count symptom sets, but an ambiguous set drags all of
its rows into whichever split it lands in, so row counts come out roughly 10%
above the set counts. After dedup there are 189,647 rows to work with.

## Experiment arms

| Arm | Weights | Inference prompt |
|-----|---------|------------------|
| 1. Base zero-shot | base | query only |
| 2. Base + RAG | base | top-k retrieved cases, then the query |
| 3. LoRA | base + adapter | query only |
| 4. LoRA + RAG | base + adapter | top-k retrieved cases, then the query |
| 5. Retrieval-only | none | majority disease among the k nearest cases |

Arm 5 is the point of comparison. If the fine-tuned or RAG-equipped model can't
beat a plain nearest-neighbour vote, the language model isn't earning its keep.

Training computes the loss on the disease tokens only, not on the symptom prompt.
If you want the earlier notebook's behaviour (loss over the whole sequence), set
`lora.loss_on_prompt: true`.

Two models are supported: `TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T`
(the default, for continuity with the earlier work) and
`Qwen/Qwen2.5-1.5B-Instruct`. Two retrievers as well: MiniLM
(`all-MiniLM-L6-v2`) embeddings with cosine similarity, and Jaccard overlap on
the raw symptom sets.

## Metrics

Exact-match accuracy is the headline number, taken after a light normalisation
(lowercase, strip punctuation and surrounding whitespace). Semantic similarity
and ROUGE-L come along as softer secondary metrics, both carried over from the
earlier notebook, and retrieval recall@k tells you how often the true disease
was even among the cases that got retrieved.

For retention, the models answer a fixed MMLU subset: five subjects (anatomy,
clinical knowledge, college biology, high-school mathematics, world religions),
50 questions each, zero-shot. Scoring compares the log-likelihood of the four
answer letters after `Answer:` rather than parsing generated text. The subset is
sampled once with a fixed seed and cached to `outputs/mmlu_subset.csv`, so every
model faces the same 250 questions. Only the base and LoRA weights get scored;
RAG doesn't touch the weights, so the base number stands in for the RAG arms.

Cost is tracked as training wall-clock time, peak GPU memory, per-example
inference latency, and mean prompt length in tokens. Everything is run across
seeds 42, 43, and 44 and reported as mean ± standard deviation.

## Running

### On Colab

Open `notebooks/colab_runner.ipynb` with a GPU runtime. It clones the repo,
installs the requirements, mounts Drive, picks up a `WANDB_API_KEY` from Colab
secrets if there is one (and quietly disables logging if not), and runs
everything with `configs/colab.yaml`, which keeps data and outputs under
`MyDrive/Projects/medqa-lora-vs-rag/`. Put the CSV at
`MyDrive/Projects/medqa-lora-vs-rag/data/disease_symptoms.csv` before you start.
There's a one-step smoke run (`configs/colab_debug.yaml`) up front to catch path
and install problems before committing to the real thing. If the session drops,
just run from the top again: finished adapters are skipped and a half-finished
one picks up from its last checkpoint.

`notebooks/analysis.ipynb` reads the aggregated CSVs and draws the tables and
figures. You don't need it to get the results, only to visualise them.

### Locally

The local path is really just for debugging the plumbing. `debug.yaml` shrinks
everything down (50 training rows, 20 test rows, a single training step) so it
finishes on a laptop CPU.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
bash scripts/prepare_data.sh configs/debug.yaml
bash scripts/run_all.sh configs/debug.yaml
pytest tests/
```

Or in Docker:

```bash
docker build -t medqa .
docker run --rm -v "$PWD:/app" medqa bash scripts/run_all.sh configs/debug.yaml
```

The device is chosen automatically, CUDA then MPS then CPU, and nothing depends
on bitsandbytes, so macOS is fine.

### Training outputs and resuming

Every LoRA run saves its adapter plus a `train_stats.json` (steps, losses,
wall-clock time, peak GPU memory) under
`outputs/adapters/<model>_seed<seed>/`. Checkpoints land in
`outputs/checkpoints/` every `lora.save_steps` steps and are what a resume reads
from after a disconnect. Once an adapter is saved its training is skipped on the
next run, so `run_all.sh` is safe to re-run as many times as you need.

### Weights & Biases

Set `WANDB_API_KEY` in the environment (a Colab secret works). To run without
it, set `wandb.enabled: false` in the config.

## Results

TinyLlama, 4000 training symptom sets, tested on 1112 rows. The accuracy ceiling
from ambiguous symptom sets is 0.899. RAG arms below use the Jaccard retriever at
k = 5; numbers are mean ± std over the three seeds where the arm was run for all
three.

| Arm | Exact match | Semantic | ROUGE-L | MMLU | Latency (ms) | Prompt tokens |
|-----|-------------|----------|---------|------|--------------|---------------|
| Base zero-shot | 0.001 | 0.32 | 0.03 | 0.224 | 117 | 38 |
| Base + RAG | 0.609 | 0.77 | 0.63 | same as base | 418 | 276 |
| LoRA | 0.727 ± 0.006 | 0.84 | 0.75 | 0.232 ± 0.014 | 95 | 38 |
| LoRA + RAG | 0.637 ± 0.011 | 0.77 | 0.66 | same as LoRA | 383 | 276 |
| Retrieval-only | 0.714 | 0.82 | 0.73 | n/a | 0.2 | n/a |

The short version: fine-tuning wins, but only just. LoRA lands at 0.727 and the
no-LLM retrieval baseline is right behind at 0.714. A plain majority vote over
the nearest cases gets within a point and a half of a fine-tuned 1.1B model,
with no training and sub-millisecond lookups.

RAG doesn't pay off here. It rescues the base model, which can't do the task at
all zero-shot (0.001), but even then reading the retrieved cases (0.609) trails
just voting on them (0.714). For the fine-tuned model RAG actively hurts: LoRA
alone beats LoRA + RAG at every k, and the gap widens as more cases are stuffed
into the prompt. The extra context seems to distract more than it helps.

Retention holds up. MMLU barely moves between base (0.224) and LoRA (0.232), so
fine-tuning on this narrow task didn't wipe out general knowledge. Both sit near
the 0.25 chance line because TinyLlama is weak on MMLU to begin with; the point
is that LoRA didn't make it worse.

### Effect of k

Exact match as k grows, Jaccard retriever:

| k | Retrieval-only | Base + RAG | LoRA + RAG |
|---|----------------|------------|------------|
| 1 | 0.674 | 0.611 | 0.718 |
| 3 | 0.690 | 0.577 | 0.679 |
| 5 | 0.714 | 0.609 | 0.637 |
| 10 | 0.710 | 0.601 | 0.615 |

Voting likes more neighbours up to a point. The language-model arms don't: LoRA +
RAG is actually best at k = 1 and falls off from there, which is the clearest
sign that the model isn't making good use of the retrieved context.

### Jaccard vs MiniLM

Jaccard beats the MiniLM embeddings at every k on this data, which makes sense
given the inputs are discrete symptom sets rather than free text. It's also far
cheaper, so it's the default. Recall@k (how often the right disease is somewhere
in the retrieved cases) climbs steadily with k for both, reaching 0.95 for
Jaccard at k = 10, so the ceiling on what RAG could do is high. The arms just
don't cash it in.

| k | Jaccard EM / recall | MiniLM EM / recall |
|---|---------------------|--------------------|
| 1 | 0.674 / 0.674 | 0.662 / 0.662 |
| 3 | 0.690 / 0.841 | 0.663 / 0.809 |
| 5 | 0.714 / 0.908 | 0.649 / 0.870 |
| 10 | 0.710 / 0.954 | 0.629 / 0.932 |

## Limitations

The dataset is synthetic, so none of this necessarily carries over to real
clinical notes. The models are small by design, chosen to fit Colab's budget,
and a stronger model might use the retrieved context better than TinyLlama does.
The MMLU subset is only 250 questions, so retention gaps smaller than a few
points are just noise. And exact match is harsh on valid synonyms, which is part
of why the semantic and ROUGE-L columns are there.

## Layout

```
configs/        base, debug, colab, colab_debug
src/medqa/      data, prompts, retrieval, train_lora, generate, metrics, evaluate
scripts/        prepare_data.sh, run_all.sh
notebooks/      colab_runner.ipynb, analysis.ipynb
tests/          test_data, test_config, test_metrics, test_generate, test_train_lora
```
