# TTT3R — vendored into lingbot-map

Clone of the upstream TTT3R repository with `.git` removed on purpose, following
the same rule as `CUT3R/UPSTREAM.md`: code reaches other zones by lingbot-map's
own git history, never by copy, and an embedded repo is invisible to the
parent's commits.

TTT3R ships no weights of its own — it is an inference-time state-update rule
that runs CUT3R's released checkpoint. `src/cut3r_512_dpt_4_64.pth` is a
symlink to the same `/group` checkpoint the vendored CUT3R uses.

## Provenance

| | |
|---|---|
| upstream | https://github.com/Inception3D/TTT3R |
| commit | `edd6d8c000aaf2ef0f588403e1b3bd3300a54cc4` |
| subject | "Add conference badges to README" |
| author | Xingyu Chen \<rover.xingyu@gmail.com\> |
| date | 2026-05-11 |
| branch | `main` |
| vendored | 2026-09-04 |
