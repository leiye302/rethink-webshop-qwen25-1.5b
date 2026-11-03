# Frozen WebShop Route-SFT data

`frozen_sft.zip` is intentionally included (about9.8MB compressed). It contains
the exact current training12000 rows, validation902, fixed-generation200 and a
public manifest, not optimizer states or unfiltered private provenance logs.
The default revision is buy-v9, including268 Buy targets (176 Good,92 Bad).
Its active archive and data lock match the 3B companion byte-for-byte; no
manual profile copying is needed. See [revision and fresh-run rules](../docs/DATA_BUY_V9.md).
Both model-size repositories contain identical bytes; no paid reannotation is
needed to reproduce SFT. `setup --data` verifies ZIP SHA before extraction and
each row-file SHA afterward, and refuses to replace different existing files.

Only `prompt`, `assistant_prefix` and `response` determine model input/target.
Other fields are audit metadata and must not be concatenated into Actor input.
Good actions are natural original samples. Bad actions are accepted next-step
corrections. The SFT labels refer to visible reasoning, not the final rollout's
success or to a guarantee that every original action must change.

Original WebShop catalog/product text and environment assets remain subject to
their original source terms. This repository does not relicense third-party
product descriptions, public model weights or upstream WebShop datasets as
Apache software. Follow [WebShop's original repository](https://github.com/princeton-nlp/WebShop)
and [SDAR](https://github.com/ZJU-REAL/SDAR) for third-party terms.
