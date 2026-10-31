# RETHINK WebShop — Qwen2.5-1.5B, from base

> Shared cross-method evaluation: [exact common140 tasks and unchanged sampling](docs/SHARED_EVALUATION140.md).


An independently runnable **base → Route-SFT → GRPO+OPSD** recipe.
Start from public `Qwen/Qwen2.5-1.5B-Instruct`, train the **same frozen WebShop
SFT samples** as the [3B companion](https://github.com/leiye302/rethink-webshop-qwen25-3b),
then run public SDAR GRPO+OPSD beta5 with Teacher-only local hints at the Actor's
own `<RETHINK>` decisions. No SEED/M1/Analyzer warm-start is required.

This repository contains a recipe, not an already trained1.5B result. It does
not load the3B checkpoint by default and does not infer1.5B performance from3B.
No author computer, private server or paid data-generation service is required
to obtain the supplied SFT dataset. Stage2 hint usage requires your own provider.

## Fixed recipe

| Stage | Settings |
|---|---|
| Base | Qwen2.5-1.5B-Instruct, revision `989aa7980e4cf806f80c7fef2b1adb7bc71aa306` |
| SFT | 7500 Good +4500 Bad, 5:3/global8/micro1, one epoch1500 updates |
| Targets | Good: original sampled action; Bad: marker + accepted corrective next action |
| Loss | Sample-mean suffix/action/EOS CE +0.1 full-vocabulary marker BCE, no reference KL |
| Precision | Full FP32 trainable parameters +BF16 autocast, no LoRA/quantization |
| Stage2 | Native SDAR GRPO+OPSD, 150 updates, gate beta5, coefficient.01 |
| Hints | Static public skills everywhere; extra local hint only when Actor generates marker |
| Teacher | Current Actor no-gradient scoring view; Student never sees hints |
| WebShop | Default1000 products/synthetic goals, history2, max15 actions,4096/512 tokens |
| Monitoring | Same fixed128 reserved goals, validation every10, checkpoints every20/final150/latest2 |

The source trajectories were collected with3B; using them for1.5B is intentional
shared-data adaptation, not claimed1.5B on-policy collection.

## Install and prepare (Linux x86-64)

Prerequisites: Git, Conda, Python3.10/3.12 and a CUDA12.8-compatible driver/toolkit.
Use only GPUs actually allocated to the job. Both training stages target
**one node with 4×A100 80GB**: SFT uses four DDP ranks with2 samples/rank/update
(global8); stage2 uses FSDP4 and **TP2×DP2** rollout. This is not a single-24GB
memory-fit claim. Offline evaluation still uses one GPU.

```bash
git clone https://github.com/leiye302/rethink-webshop-qwen25-1.5b
cd rethink-webshop-qwen25-1.5b
conda create -y -p "$PWD/.venv-opsd" -c conda-forge python=3.10 openjdk=11
conda activate "$PWD/.venv-opsd"
python -m pip install --upgrade pip setuptools wheel packaging==25.0
python scripts/install_opsd.py
python -m pip install --no-build-isolation flash-attn==2.8.3
mkdir -p assets
export PYTHONPATH="$PWD/src:$PWD/.deps/sdar"
python -m rethink_webshop.java --prepare
python -m rethink_webshop.setup --framework --data
python -m pip install --no-deps -e .deps/sdar
python -m pip check
export JAVA_HOME="$PWD/assets/java"
export JVM_PATH="$JAVA_HOME/lib/server/libjvm.so"
bash scripts/prepare.sh
python -m rethink_webshop.setup --base
python -m pytest -q
python scripts/smoke_native.py

python3.12 -m venv .venv-sft
.venv-sft/bin/pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
.venv-sft/bin/pip install -r configs/requirements-sft.txt
.venv-sft/bin/python -m rethink_webshop.audit_data
```

Preparation pins [SDAR](https://github.com/ZJU-REAL/SDAR) at
`d511f043afe9199b8576fc99ea52e12d84841f32`, applies a3-file patch, verifies
unchanged loss/prompt/skills, extracts frozen data, downloads public WebShop
assets and builds native Lucene indexes. All downloads are explicitly described
and hash checked. Setup does not start paid API calls or training.

## Stage1: SFT from1.5B base

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/train_sft.sh
```

The script first checks sampler/tokenization, native AdamW numerical parity and
2 disposable real-model updates, then reloads the original base for formal SFT.
It saves complete resumable checkpoints750/1500, validation loss,200 truly free
generations and the fixed128 native episodes. The first supervised token is
directly `<RETHINK>` or `<action>`; H/R targets are masked, natural separator
and joint tokenization must match. No merged `><` boundary is silently accepted.

Resume only the same complete run with
`CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/train_sft.sh --resume runs/sft/checkpoint_000750`.
The recipe rejects different code, parent, data, batch order, world size or
optimizer state. Each SFT checkpoint stores the four per-rank RNG states.
Both750/1500 full checkpoints require room for weights plus FP32 AdamW states;
see [storage notes](docs/REPRODUCING.md).

## Stage2: Actor-routed local-hint GRPO+OPSD

```bash
export HINT_API_BASE="https://your-provider.example/v1"
export HINT_MODEL="glm-5.3-flash"
read -rsp 'Hint API key: ' HINT_API_KEY; echo
export HINT_API_KEY
CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/train_opsd.sh --allow-api-cost
```

Default Actor is **`runs/sft/checkpoint_001500/model`**. Configure your API provider
and key; the model is fixed to final-ALFWorld `glm-5.3-flash`. One episode request
groups Actor-triggered steps; the API does not judge routes. Hints only augment
Teacher context. Student responses, actions, rewards, GRPO participation and
native confidence gate remain unchanged. Teacher uses current Actor weights,
not a frozen initial model. The native stage2 reference-policy KL remains.01;
this is separate from the absent SFT reference KL.

The final ALFWorld model/protocol is pinned to `glm-5.3-flash`. Invalid returned
hints/illegal commands fall back to static-skill Teacher with coverage counters.
Transport/auth/quota failures retain the same batch and wait; truncated Teacher
hints fail visibly. See [the exact alignment](docs/ALFWORLD_ALIGNMENT.md).
Add `--resume` only to
resume the same native run. Do not reuse an output directory for a different arm.

## Baseline and evaluation

```bash
# Same public GRPO+OPSD beta5 from1.5B base, no local hint API:
CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/train_opsd.sh --baseline --run-dir runs/baseline
# Formal native shared140 benchmark, temperature0.4, never using hints:
CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/eval.sh --model assets/base --output runs/eval_base
CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/eval.sh --model runs/sft/checkpoint_001500/model --output runs/eval_sft1500
```

Formal evaluation and RL validation use the same shared140 native protocol:
temperature0.4, sampling, top-p1, unrestricted top-k, one rollout/task,
history2, prompt4096, response512 and15 environment steps. No hint API is used
at evaluation. The200-H greedy generation probe is a separate SFT diagnostic.
Old128/greedy scores are not relabeled. See the exact shared contract above.
Report native success and mean raw task score separately. Marker rate counts
marker-bearing active decisions, not a fraction of generated tokens. Source
Good/Bad groups are not labels for freshly generated reasoning.

## Details and validation boundaries

- [Method, source data, masks and exact objectives](docs/METHOD.md)
- [Four minimally patched upstream files and named configuration changes](docs/UPSTREAM_DIFF.md)
- [Resources, environment, recovery, API safety and evaluation](docs/REPRODUCING.md)
- [Executed checks / unverified GPU work](docs/VERIFICATION.md)
- [Data](data/README.md), [third-party/model terms](THIRD_PARTY.md)

CPU/clean-checkout checks are not a newly completed1.5B GPU training result.
No full resume states, private logs, credentials, machine addresses or user
computer paths are included in either public repository.
