# Purchase-supplement SFT database: buy-v9

This is a separately versioned frozen database. The original default database,
its existing descriptions, model resources and training/evaluation algorithms
are not changed by this publication.

## Contents

`data/buy_v9/frozen_sft.zip` contains `train.jsonl`, `val.jsonl`,
`generation_val200.jsonl` and the manifest consumed by the existing SFT loader.
`data/buy_v9/data.json` locks the ZIP and each row file by SHA256 and byte count.
No download from an author's machine, private server or data-generation service
is required. Model/Java/WebShop dependencies remain those in the main README.

| Partition | Good | Bad |
| --- | ---: | ---: |
| Train | 7,500 | 4,500 |
| SFT validation | 631 | 271 |
| Free-generation diagnostic | 100 | 100 |

The 12,000-row training set contains **268 Buy targets: 176 Good and 92 Bad**.
This revision adds 87 natural Good Buy and 30 corrected Bad Buy targets to its
preceding accepted library, replacing only equal numbers of same-route non-Buy
rows. Good retains its original sampled action and has no full-score gate.
The 30 added corrective targets passed exact-current-state native execution
checks. These checks are not a claim of perfect semantic labels or a trained
model's success/trigger rate. The validation and generation files are unchanged.

Train SHA256:
`860a7b0ac73f91396089365947570c2f3052b2318d553ec1ac204964e78b9e59`.

## Select v9 before preparation

Use a **fresh checkout**, before any command extracts SFT data:

```bash
git clone https://github.com/leiye302/rethink-webshop-qwen25-1.5b
cd rethink-webshop-qwen25-1.5b
cp data/buy_v9/frozen_sft.zip data/frozen_sft.zip
cp data/buy_v9/data.json configs/data.json
```

Then follow the existing installation, preparation and SFT commands in the main
README. Once the documented dependencies are installed, the unchanged data
preparation command is:

```bash
export PYTHONPATH="$PWD/src:$PWD/.deps/sdar"
python -m rethink_webshop.setup --data
```

Both copied files are required: copying only the ZIP must fail the old checksum
guard. The loader refuses to overwrite a different previously extracted dataset.
Do not overlay v9 onto an existing run or resume an optimizer trained on another
database. An existing model's training provenance is not relabeled by this release.

The scientific recipe stays one epoch, 1,500 updates, global batch eight with
5 Good/3 Bad, LR 2e-6, suffix CE plus 0.1 full-vocabulary marker BCE, no SFT
reference KL, masked H/R and an 8,192-token error-on-overlength limit. Formal
evaluation remains the exact common140 manifest and decoding/scoring protocol.
Stage 2's Actor marker routing, Teacher-only local hints and beta5 gate are
unchanged. Selecting data does not launch a GPU job or an API request.

`verification.json` records the data checks; `tests/test_buy_v9_data.py` exercises
the original extractor and actual training-data validator with this bundle.
Those CPU checks are not a new GPU training or performance result.
