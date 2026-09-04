# ZipMap (main branch) — vendored into lingbot-map

Clone of the upstream ZipMap repository (model, demos, training) with `.git`
removed on purpose — see `CUT3R/UPSTREAM.md` for the rule. The quantitative
evaluation harness lives on upstream's separate `evaluation` branch, vendored
as the sibling directory `ZipMap_eval/`.

Checkpoints are symlinks into `/group/compact-3dmem/checkpoints/ZipMap/`
(downloaded from https://huggingface.co/coast01/ZipMap).

## Provenance

| | |
|---|---|
| upstream | https://github.com/haian-jin/ZipMap |
| commit | `e0f1f400852e8d4770ff1dbb6fd017a6048afbfd` |
| subject | "Correct license information" |
| author | Haian Jin \<haianjin0415@gmail.com\> |
| date | 2026-06-10 |
| branch | `main` |
| vendored | 2026-09-04 |
