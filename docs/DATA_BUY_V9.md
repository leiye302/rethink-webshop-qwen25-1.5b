# Active 1.5B Stage1 data: buy-v9

The default recipe is **public Qwen2.5-1.5B base → buy-v9 Route-SFT1500 →
GRPO+OPSD150**. Its active `data/frozen_sft.zip` and `configs/data.json` match
the 3B companion's buy-v9 database byte-for-byte. All files are bundled here;
another checkout, an author's machine or a private data server is not needed.

| Partition | Good | Bad |
|---|---:|---:|
| Train | 7,500 | 4,500 |
| SFT validation | 631 | 271 |
| Free-generation diagnostic | 100 | 100 |

The training set contains **268 Buy targets: 176 Good and 92 Bad**. The accepted
revision adds87 natural Good Buy and30 corrected Bad Buy rows relative to its
preceding library, replacing equal numbers of same-route non-Buy examples.
Good keeps its original sampled action, without a full-score gate. Bad targets
the marker plus an accepted corrective action. The30 added corrective actions
passed exact-current-state native execution checks. These data checks are not
perfect-label or trained-model performance claims. Validation and generation
files are unchanged. Original revision checks: `data/buy_v9/verification.json`.

- ZIP SHA256: `34b25a42c2717bdee5d3a3bee8a955b4cffcdcc60ebbbfa3ca62a9c6693692f9`.
- Train SHA256: `860a7b0ac73f91396089365947570c2f3052b2318d553ec1ac204964e78b9e59`.

## Start a fresh data-bound experiment

Clone the current repository into a new directory and follow the main README.
No manual ZIP/config copying is needed: buy-v9 is already the default.
Preparation checks the ZIP and all members and refuses to overwrite a different
extracted database. The training validator also compares the extracted manifest
to the active data lock, preventing a self-consistent stale cache from being
silently used after an update.

Keep any old checkout, checkpoints and records intact. Do not overlay the new
data on an old run, resume an old optimizer or relabel old weights as buy-v9.

After the documented installation and preparation:

```bash
.venv-sft/bin/python -m rethink_webshop.audit_data
CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/train_sft.sh
# Configure the Teacher-only hint API as documented, then:
CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/train_opsd.sh --allow-api-cost
```

Stage2 defaults to `runs/sft/checkpoint_001500/model`. Its Stage1 completion,
parent, recipe, data and model identities are verified before training. Baseline
still starts from the public1.5B base. Independent evaluation does not require
an SFT training receipt.

## Unchanged scientific recipe

SFT remains1500 updates,5 Good/3 Bad,global8,micro1,LR2e-6,masked H/R,
suffix CE plus0.1 independent full-vocabulary marker BCE,no reference KL,
max_length8192/error,and four DDP ranks. Stage2 remains150 updates,Actor-owned
RETHINK routing,current-Actor no-gradient Teacher,Teacher-only local hints,
SDAR beta5 and coefficient0.01. The shared140 task IDs,seeds,decoding and scores
are unchanged. Selecting or checking data does not start training or API calls.
