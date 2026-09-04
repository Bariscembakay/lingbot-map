# ZipMap evaluation harness — vendored into lingbot-map

Upstream ZipMap's `evaluation` branch, which is not a variant of the main
branch but a separate tree: a fork of the Pi3 authors' recons_eval
(https://github.com/ZhouTimeMachine/recons_eval) with ZipMap interfaces added.
It carries the mv_recon / relpose / videodepth / monodepth suites we reproduce
the paper numbers with. Vendored with `.git` removed — see `CUT3R/UPSTREAM.md`
for the rule. Committed `__pycache__` directories from upstream were dropped.

## Provenance

| | |
|---|---|
| upstream | https://github.com/haian-jin/ZipMap (branch `evaluation`) |
| commit | `8b6b629d9ba39a36776d3c384df263421979131f` |
| subject | "Update README.md" |
| author | Haian Jin \<haianjin0415@gmail.com\> |
| date | 2026-04-16 |
| branch | `evaluation` |
| vendored | 2026-09-04 |
