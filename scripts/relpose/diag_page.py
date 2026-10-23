#!/usr/bin/env python3
"""HTML report of traj_diagnostics.py output: aggregate tables per benchmark, per-scene ATE, the per-scene
figures and a glossary. Findings are passed in as a text file (one finding per line, HTML allowed), since
they are a reading of the numbers rather than numbers.
Usage: diag_page.py --diag DIR --out DIR [--findings FILE] [--notes FILE]
"""
import argparse
import html
import json
import shutil
from pathlib import Path

import numpy as np

COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f"]

COLUMNS = [  # (header, group, getter over one scene's result, format)
    ("ATE m", "global", lambda r: r["ate"], "{:.1f}"),
    ("rot @1", "local", lambda r: r["gap"]["1"]["rot_deg"], "{:.2f}"),
    ("trans @1 m", "local", lambda r: r["gap"]["1"]["trans_m"], "{:.2f}"),
    ("rot @16", "local", lambda r: r["gap"]["16"]["rot_deg"], "{:.2f}"),
    ("rot @256", "drift", lambda r: r["gap"]["256"]["rot_deg"], "{:.2f}"),
    ("rot @1024", "drift", lambda r: r["gap"].get("1024", {}).get("rot_deg"), "{:.1f}"),
    ("trans @256 %", "drift", lambda r: r["gap"]["256"]["trans_pct"], "{:.1f}"),
    ("t_rel %", "drift", lambda r: (r["kitti_odometry"] or {}).get("t_rel_pct"), "{:.1f}"),
    ("r_rel °/100 m", "drift", lambda r: (r["kitti_odometry"] or {}).get("r_rel_deg_per_100m"), "{:.2f}"),
    ("|bias| °/frame", "drift", lambda r: float(np.linalg.norm(r["bias_deg_per_frame_xyz"])), "{:.4f}"),
    ("scale p10", "scale", lambda r: (r["scale"]["log2_p10_p50_p90"] or [None])[0], "{:+.2f}"),
    ("scale p90", "scale", lambda r: (r["scale"]["log2_p10_p50_p90"] or [None, None, None])[2], "{:+.2f}"),
    ("late / early", "stream", lambda r: (lambda s: s[1] / s[0])(r["stream"]["gap10_rot_deg_first_last"]) if "stream" in r else None, "{:.2f}"),
    ("jitter pos ×", "smooth", lambda r: r["jitter"]["pos_rms_ratio"], "{:.1f}"),
    ("jitter rot ×", "smooth", lambda r: r["jitter"]["rot_rms_ratio"], "{:.1f}"),
    ("still m/frame", "smooth", lambda r: r["still"]["pred_m_per_frame"], "{:.3f}"),
    ("revisit err m", "revisit", lambda r: r["revisit"].get("dist_err_m_mean"), "{:.1f}"),
]
MEDIAN = {"jitter pos ×", "jitter rot ×"}   # heavy-tailed: one scene would set a mean

GLOSSARY = [
    ("ATE", "root-mean-square camera-centre error after one Sim3 alignment of the whole trajectory to GT, metres"),
    ("rot / trans @d", "mean error of the relative pose between frames i and i+d (rotation in degrees, translation in metres or % of the GT distance). Flat across d means jitter, growing like √d a random walk, growing like d a constant per-step bias"),
    ("t_rel, r_rel", "KITTI odometry metrics: relative error over 100–800 m stretches, in % of distance and degrees per 100 m"),
    ("|bias|", "length of the mean signed per-step rotation error (camera axes), degrees per frame; a constant bias adds up linearly over the stream"),
    ("scale p10 / p90", "log2 of predicted over GT path length on sliding 100-frame segments; 0 is right, +1 is twice too long. Jitter inflates it, scale drift spreads it"),
    ("late / early", "gap-10 rotation error over the last fifth of the frames divided by the first fifth; above 1 the method degrades as the stream grows"),
    ("jitter ×", "RMS second difference of the camera centres (pos) and of the step rotations (rot), relative to GT's; 1 is as smooth as GT. Median over scenes"),
    ("still", "predicted motion per frame while GT moves less than 1 cm per frame"),
    ("revisit err", "error in the distance between two frames more than 300 frames apart that GT places within 3 m of each other: what a loop closure has to fix"),
]

CSS = """
:root { --bg: #f3f6f8; --surface: #ffffff; --ink: #17202a; --muted: #566472; --rule: #d6dee5; --accent: #1b5b88;
  --head: "Newsreader", Georgia, serif; --body: "Public Sans", system-ui, sans-serif; --mono: "JetBrains Mono", ui-monospace, monospace; }
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) { --bg: #0e141a; --surface: #151d25; --ink: #e2e8ee; --muted: #93a1ae; --rule: #28343f; --accent: #84b6de; color-scheme: dark } }
:root[data-theme="dark"] { --bg: #0e141a; --surface: #151d25; --ink: #e2e8ee; --muted: #93a1ae; --rule: #28343f; --accent: #84b6de; color-scheme: dark }
body { background: var(--bg); color: var(--ink); font-family: var(--body); font-size: 15px; line-height: 1.55;
  padding-left: max(16px, env(safe-area-inset-left, 0px)); padding-right: max(16px, env(safe-area-inset-right, 0px)); padding-block: 28px 56px; }
main { max-width: 1180px; margin: 0 auto; display: grid; gap: 36px; }
h1, h2 { font-family: var(--head); font-weight: 600; text-wrap: balance; margin: 0; }
h1 { font-size: 2.1rem; letter-spacing: -0.01em; }
h2 { font-size: 1.45rem; }
h3 { font-size: 0.78rem; text-transform: uppercase; letter-spacing: 0.08em; color: var(--muted); margin: 0 0 8px; font-weight: 600; }
.lede { color: var(--muted); max-width: 72ch; margin: 8px 0 0; }
section { display: grid; gap: 14px; min-width: 0; }
.findings { margin: 0; padding-left: 1.2em; display: grid; gap: 8px; max-width: 86ch; }
.legend { display: flex; flex-wrap: wrap; gap: 8px 18px; font-size: 0.9rem; }
.legend span { display: inline-flex; align-items: center; gap: 7px; }
.sw { width: 18px; height: 4px; border-radius: 2px; display: inline-block; }
.tablewrap { overflow-x: auto; border: 1px solid var(--rule); border-radius: 6px; background: var(--surface); }
table { border-collapse: collapse; font-size: 0.86rem; width: 100%; }
th, td { padding: 6px 10px; text-align: right; white-space: nowrap; border-bottom: 1px solid var(--rule); }
th:first-child, td:first-child { text-align: left; position: sticky; left: 0; background: var(--surface); }
thead th { font-weight: 600; color: var(--muted); font-size: 0.78rem; }
thead tr.groups th { text-transform: uppercase; letter-spacing: 0.07em; font-size: 0.7rem; border-bottom: none; padding-bottom: 0; }
td { font-family: var(--mono); font-variant-numeric: tabular-nums; }
td.best { font-weight: 700; color: var(--accent); }
tbody tr:last-child td { border-bottom: none; }
.gallery { display: grid; gap: 22px; }
figure { margin: 0; display: grid; gap: 6px; min-width: 0; }
figure img { border: 1px solid var(--rule); border-radius: 6px; background: #fff; width: 100%; }
figcaption { font-size: 0.86rem; color: var(--muted); }
figcaption b { color: var(--ink); font-family: var(--mono); font-weight: 600; }
dl { display: grid; grid-template-columns: minmax(120px, 190px) 1fr; gap: 8px 18px; margin: 0; max-width: 100ch; }
dt { font-family: var(--mono); font-size: 0.85rem; font-weight: 600; }
dd { margin: 0; color: var(--muted); }
.toc { display: flex; flex-wrap: wrap; gap: 6px 12px; font-family: var(--mono); font-size: 0.82rem; }
a { color: var(--accent); }
a:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }
@media (max-width: 560px) { dl { grid-template-columns: 1fr; } h1 { font-size: 1.7rem; } }
"""


def agg(results, col):
    _, _, get, _ = col
    vals = []
    for r in results:
        try:
            v = get(r)
        except (KeyError, TypeError, ZeroDivisionError):
            v = None
        if v is not None and np.isfinite(v):
            vals.append(v)
    if not vals:
        return None
    return float(np.median(vals) if col[0] in MEDIAN else np.mean(vals))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--diag", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--findings", type=Path)
    ap.add_argument("--notes", type=Path, help="protocol / pending notes, one per line, HTML allowed")
    ap.add_argument("--title", default="LingBot vs ABot Diagnostics")
    args = ap.parse_args()
    d = json.loads((args.diag / "diagnostics.json").read_text())
    methods = list(dict.fromkeys(m for s in d.values() for m in s))
    color = dict(zip(methods, COLORS))
    (args.out / "fig").mkdir(parents=True, exist_ok=True)
    benches = list(dict.fromkeys(s.split("/")[0] for s in d))
    parts = [f"<title>{html.escape(args.title)}</title>",
             '<meta name="claude:viewport-fit-sides" content="cover">',
             '<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>',
             '<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600;700&family=Newsreader:opsz,wght@6..72,600&family=Public+Sans:wght@400;600&display=swap">',
             f"<style>{CSS}</style>", "<main>",
             f"<header><h1>{html.escape(args.title)}</h1><p class=\"lede\">Trajectory diagnostics on KITTI 00–10 and VBR "
             f"(7 scenes): every frame, Sim3-aligned, on the same 504×280 frames for every method. Each figure: "
             f"top-down trajectories, error against frame gap, and path-length ratio along the stream.</p></header>"]
    legend = "".join(f'<span><i class="sw" style="background:{color[m]}"></i>{html.escape(m)}</span>' for m in methods)
    if args.findings:
        items = [l.strip() for l in args.findings.read_text().splitlines() if l.strip()]
        parts.append("<section><h2>What the numbers say</h2><ul class=\"findings\">"
                     + "".join(f"<li>{i}</li>" for i in items) + "</ul></section>")
    for bench in benches:
        scenes = [s for s in d if s.startswith(bench + "/")]
        groups = []
        for c in COLUMNS:
            if groups and groups[-1][0] == c[1]:
                groups[-1][1] += 1
            else:
                groups.append([c[1], 1])
        rows = []
        vals = {m: [agg([d[s][m] for s in scenes if m in d[s]], c) for c in COLUMNS] for m in methods}
        for m in methods:
            cells = []
            for k, c in enumerate(COLUMNS):
                v = vals[m][k]
                cells.append(f"<td>{c[3].format(v) if v is not None else '–'}</td>")
            rows.append(f'<tr><td><i class="sw" style="background:{color[m]}"></i> {html.escape(m)}</td>{"".join(cells)}</tr>')
        parts.append(f"<section><h2>{bench.upper()}: {len(scenes)} scenes, mean over scenes</h2><div class=\"legend\">{legend}</div>"
                     "<div class=\"tablewrap\"><table><thead><tr class=\"groups\"><th></th>"
                     + "".join(f'<th colspan="{n}" style="text-align:center">{g}</th>' for g, n in groups)
                     + "</tr><tr><th>method</th>" + "".join(f"<th>{html.escape(c[0])}</th>" for c in COLUMNS)
                     + "</tr></thead><tbody>" + "".join(rows) + "</tbody></table></div></section>")
    rows = []
    for s, res in d.items():
        ates = {m: res[m]["ate"] for m in methods if m in res}
        best = min(ates, key=ates.get)
        rows.append(f'<tr><td><a href="#{s.replace("/", "_")}">{s}</a></td>'
                    + "".join(f'<td class="{"best" if m == best else ""}">{ates[m]:.2f}</td>' if m in ates else "<td>–</td>"
                              for m in methods) + "</tr>")
    parts.append("<section><h2>ATE per scene, metres</h2><div class=\"tablewrap\"><table><thead><tr><th>scene</th>"
                 + "".join(f'<th><i class="sw" style="background:{color[m]}"></i> {html.escape(m)}</th>' for m in methods)
                 + "</tr></thead><tbody>" + "".join(rows) + "</tbody></table></div></section>")
    figs = []
    for s, res in d.items():
        name = s.replace("/", "_")
        src = args.diag / f"{name}.png"
        if src.exists():
            shutil.copy(src, args.out / "fig" / f"{name}.png")
        cap = " · ".join(f"{html.escape(m)} <b>{res[m]['ate']:.1f} m</b>" for m in methods if m in res)
        figs.append(f'<figure id="{name}"><img src="fig/{name}.png" alt="{s}: trajectories, gap curves, scale drift" loading="lazy">'
                    f"<figcaption><b>{s}</b> · {cap}</figcaption></figure>")
    toc = "".join(f'<a href="#{s.replace("/", "_")}">{s}</a>' for s in d)
    parts.append(f"<section><h2>Scenes</h2><div class=\"toc\">{toc}</div><div class=\"gallery\">{''.join(figs)}</div></section>")
    parts.append("<section><h2>How to read the columns</h2><dl>"
                 + "".join(f"<dt>{html.escape(k)}</dt><dd>{html.escape(v)}</dd>" for k, v in GLOSSARY) + "</dl></section>")
    if args.notes:
        items = [l.strip() for l in args.notes.read_text().splitlines() if l.strip()]
        parts.append("<section><h2>Protocol and status</h2><ul class=\"findings\">"
                     + "".join(f"<li>{i}</li>" for i in items) + "</ul></section>")
    parts.append("</main>")
    (args.out / "index.html").write_text("\n".join(parts))
    print(args.out / "index.html")


if __name__ == "__main__":
    main()
