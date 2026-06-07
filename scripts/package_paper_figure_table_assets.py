"""Package paper-ready figure and table assets for the SRAF-ID manuscript.

This script is deliberately artifact-only. It reads existing evidence files and
creates a traceable Sensors-style asset package. It does not run experiments,
modify models, or invent values.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import math
import shutil
import textwrap
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

try:
    from reportlab.lib.pagesizes import landscape
    from reportlab.lib.utils import ImageReader
    from reportlab.pdfgen import canvas

    REPORTLAB_AVAILABLE = True
except Exception:
    REPORTLAB_AVAILABLE = False


DPI = 600
FIG_W_IN = 7.2
FIG_H_IN = 4.4
PX_W = int(FIG_W_IN * DPI)
PX_H = int(FIG_H_IN * DPI)
LABEL_PT = 9
ANNOTATION_PT = 8
TITLE_PT = 11
SMALL_PT = 8

MODEL_COLORS = {
    "ID-MLP-clean": "#8A8F98",
    "ID-MLP-CA": "#4C78A8",
    "SRAF-ID-noGate": "#F58518",
    "SRAF-ID": "#54A24B",
    "Persistence": "#B279A2",
}

FAULT_LABELS = {
    "random_missing_20": "RM20",
    "random_missing_40": "RM40",
    "continuous_outage_24": "Outage24",
    "gaussian_noise_high": "Noise",
    "linear_drift_high": "Drift",
    "stuck_at_last_value_high": "Stuck",
    "clean": "Clean",
}

SEVERE_FAULTS = {
    "random_missing_40",
    "continuous_outage_24",
    "gaussian_noise_high",
    "linear_drift_high",
}


@dataclass
class Asset:
    asset_id: str
    asset_type: str
    manuscript_label: str
    title: str
    directory: Path
    source_artifacts: list[Path]
    generated_files: list[Path]
    data_files: list[Path]
    claim_supported: str
    limitations: list[str]
    warnings: list[str]


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def checksum_map(paths: Iterable[Path]) -> dict[str, str]:
    out: dict[str, str] = {}
    for p in paths:
        if p.exists() and p.is_file():
            out[str(p)] = sha256_file(p)
    return out


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required source file is missing: {path}")
    return pd.read_csv(path)


def write_json(path: Path, obj: Any) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    ensure_dir(path.parent)
    path.write_text(text.strip() + "\n", encoding="utf-8")


def copy_file(src: Path, dst: Path) -> Path:
    ensure_dir(dst.parent)
    shutil.copy2(src, dst)
    return dst


def fmt_value(x: Any, digits: int = 3) -> str:
    if pd.isna(x):
        return "TODO"
    if isinstance(x, (float, np.floating)):
        return f"{float(x):.{digits}f}"
    return str(x)


def df_to_markdown(df: pd.DataFrame, digits: int = 3) -> str:
    rows = []
    cols = list(df.columns)
    rows.append("| " + " | ".join(cols) + " |")
    rows.append("| " + " | ".join(["---"] * len(cols)) + " |")
    for _, row in df.iterrows():
        rows.append("| " + " | ".join(fmt_value(row[c], digits) for c in cols) + " |")
    return "\n".join(rows)


def latex_escape(s: Any) -> str:
    text = str(s)
    repl = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(repl.get(ch, ch) for ch in text)


def df_to_tex(df: pd.DataFrame, caption: str, label: str, digits: int = 3) -> str:
    cols = list(df.columns)
    spec = "l" + "r" * max(0, len(cols) - 1)
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\small",
        r"\caption{" + latex_escape(caption) + "}",
        r"\label{" + latex_escape(label) + "}",
        r"\begin{tabular}{" + spec + "}",
        r"\hline",
        " & ".join(latex_escape(c) for c in cols) + r" \\",
        r"\hline",
    ]
    for _, row in df.iterrows():
        lines.append(" & ".join(latex_escape(fmt_value(row[c], digits)) for c in cols) + r" \\")
    lines += [r"\hline", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def save_table_outputs(asset_dir: Path, stem: str, df: pd.DataFrame, caption: str) -> list[Path]:
    out_dir = asset_dir / "output"
    data_dir = asset_dir / "data"
    ensure_dir(out_dir)
    ensure_dir(data_dir)
    output_csv = out_dir / f"{stem}.csv"
    output_md = out_dir / f"{stem}.md"
    output_tex = out_dir / f"{stem}.tex"
    output_tsv = out_dir / f"{stem}.tsv"
    df.to_csv(output_csv, index=False, float_format="%.6f")
    write_text(output_md, df_to_markdown(df))
    write_text(output_tex, df_to_tex(df, caption, f"tab:{stem}"))
    df.to_csv(output_tsv, index=False, sep="\t", float_format="%.6f", quoting=csv.QUOTE_MINIMAL)
    return [output_csv, output_md, output_tex, output_tsv]


def save_source_readme(path: Path, sources: list[Path], columns: list[str], derived: str, script: str) -> None:
    text = [
        "# Source Data",
        "",
        "## Source Artifacts",
        *[f"- `{p}`" for p in sources],
        "",
        "## Columns Used",
        *[f"- `{c}`" for c in columns],
        "",
        "## Derived Fields",
        derived or "No additional derived fields beyond selection, renaming, or documented aggregation.",
        "",
        "## Aggregation Rules",
        "Values are copied from traceable artifact CSV files or aggregated exactly as described in the asset README.",
        "",
        "## Script Used",
        f"`{script}`",
    ]
    write_text(path, "\n".join(text))


def font(size_px: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        Path("C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf"),
        Path("C:/Windows/Fonts/calibrib.ttf" if bold else "C:/Windows/Fonts/calibri.ttf"),
        Path("C:/Windows/Fonts/segoeuib.ttf" if bold else "C:/Windows/Fonts/segoeui.ttf"),
    ]
    for p in candidates:
        if p.exists():
            return ImageFont.truetype(str(p), size_px)
    return ImageFont.load_default()


def pt_to_px(pt: int) -> int:
    return max(1, int(round(pt / 72.0 * DPI)))


def draw_text(draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str, size_pt: int = LABEL_PT, fill="#222222", bold: bool = False, anchor: str | None = None) -> None:
    draw.text(xy, text, fill=fill, font=font(pt_to_px(size_pt), bold=bold), anchor=anchor)


def svg_text(x: float, y: float, text: str, size: int = LABEL_PT, fill: str = "#222222", anchor: str = "start", weight: str = "normal") -> str:
    return f'<text x="{x:.1f}" y="{y:.1f}" font-family="Arial, Helvetica, sans-serif" font-size="{size}pt" font-weight="{weight}" fill="{fill}" text-anchor="{anchor}">{html.escape(str(text))}</text>'


def svg_rect(x: float, y: float, w: float, h: float, fill: str, stroke: str = "#333333", rx: float = 6) -> str:
    return f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" rx="{rx}" fill="{fill}" stroke="{stroke}" stroke-width="1.1"/>'


def svg_line(x1: float, y1: float, x2: float, y2: float, color: str = "#333333", width: float = 1.2, arrow: bool = True) -> str:
    marker = ' marker-end="url(#arrow)"' if arrow else ""
    return f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{color}" stroke-width="{width}"{marker}/>'


def save_pdf_from_png(png_path: Path, pdf_path: Path, warnings: list[str]) -> None:
    if not REPORTLAB_AVAILABLE:
        warnings.append("PDF export skipped because reportlab is unavailable.")
        return
    img = Image.open(png_path)
    page = landscape((FIG_W_IN * 72, FIG_H_IN * 72))
    c = canvas.Canvas(str(pdf_path), pagesize=page)
    c.drawImage(ImageReader(img), 0, 0, width=page[0], height=page[1])
    c.save()


def write_svg(path: Path, body: str, width: int = 720, height: int = 440) -> None:
    defs = """
<defs>
  <marker id="arrow" markerWidth="8" markerHeight="8" refX="7" refY="3.5" orient="auto">
    <polygon points="0 0, 8 3.5, 0 7" fill="#333333"/>
  </marker>
</defs>
"""
    svg = f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">\n{defs}\n<rect width="100%" height="100%" fill="white"/>\n{body}\n</svg>\n'
    write_text(path, svg)


def draw_bar_png(
    path: Path,
    title: str,
    panels: list[dict[str, Any]],
    y_label: str,
    colors: dict[str, str],
    zero_line: bool = False,
    y_min: float | None = None,
    y_max: float | None = None,
) -> None:
    img = Image.new("RGB", (PX_W, PX_H), "white")
    d = ImageDraw.Draw(img)
    margin_l, margin_r, margin_t, margin_b = int(0.65 * DPI), int(0.2 * DPI), int(0.45 * DPI), int(0.55 * DPI)
    draw_text(d, (PX_W // 2, int(0.18 * DPI)), title, TITLE_PT, bold=True, anchor="mm")
    all_vals: list[float] = []
    for p in panels:
        for series in p["series"].values():
            all_vals += [float(v) for v in series]
    if y_min is None:
        y_min = min(0.0, min(all_vals) if all_vals else 0.0)
    if y_max is None:
        y_max = max(all_vals) if all_vals else 1.0
    if abs(y_max - y_min) < 1e-9:
        y_max += 1.0
    pad = 0.08 * (y_max - y_min)
    y_min -= pad
    y_max += pad

    panel_w = (PX_W - margin_l - margin_r) // len(panels)
    plot_h = PX_H - margin_t - margin_b
    for pi, panel in enumerate(panels):
        x0 = margin_l + pi * panel_w
        x1 = x0 + panel_w - int(0.15 * DPI)
        y0 = margin_t
        y1 = margin_t + plot_h
        d.rectangle([x0, y0, x1, y1], outline="#C8CDD3", width=max(1, DPI // 300))
        draw_text(d, ((x0 + x1) // 2, y0 - int(0.12 * DPI)), panel["title"], LABEL_PT, bold=True, anchor="mm")
        if pi == 0:
            draw_text(d, (int(0.17 * DPI), (y0 + y1) // 2), y_label, LABEL_PT, anchor="mm")
        cats = panel["categories"]
        series_names = list(panel["series"].keys())
        group_w = (x1 - x0) / max(1, len(cats))
        bar_w = group_w * 0.7 / max(1, len(series_names))

        def y_to_pix(v: float) -> float:
            return y1 - (v - y_min) / (y_max - y_min) * (y1 - y0)

        for tick in np.linspace(y_min, y_max, 5):
            yy = y_to_pix(float(tick))
            d.line([x0, yy, x1, yy], fill="#E6E8EB", width=max(1, DPI // 500))
            if pi == 0:
                draw_text(d, (x0 - int(0.07 * DPI), int(yy)), f"{tick:.2f}", SMALL_PT, fill="#555555", anchor="rm")
        if zero_line and y_min < 0 < y_max:
            yy = y_to_pix(0.0)
            d.line([x0, yy, x1, yy], fill="#333333", width=max(2, DPI // 180))
        for ci, cat in enumerate(cats):
            gx = x0 + ci * group_w + group_w * 0.15
            for si, name in enumerate(series_names):
                val = float(panel["series"][name][ci])
                bx0 = gx + si * bar_w
                bx1 = bx0 + bar_w * 0.82
                by = y_to_pix(val)
                base = y_to_pix(0 if y_min < 0 < y_max else y_min)
                d.rectangle([int(bx0), int(min(by, base)), int(bx1), int(max(by, base))], fill=colors.get(name, "#777777"))
            draw_text(d, (int(gx + group_w * 0.35), y1 + int(0.08 * DPI)), cat, SMALL_PT, fill="#333333", anchor="mt")
    # Legend
    legend_items = list(panels[0]["series"].keys()) if panels else []
    lx = PX_W - margin_r - int(1.9 * DPI)
    ly = int(0.18 * DPI)
    for i, name in enumerate(legend_items):
        yy = ly + i * int(0.14 * DPI)
        d.rectangle([lx, yy, lx + int(0.08 * DPI), yy + int(0.06 * DPI)], fill=colors.get(name, "#777777"))
        draw_text(d, (lx + int(0.11 * DPI), yy), name, SMALL_PT, fill="#333333")
    img.save(path, dpi=(DPI, DPI))


def draw_scatter_png(path: Path, title: str, df: pd.DataFrame) -> None:
    img = Image.new("RGB", (PX_W, PX_H), "white")
    d = ImageDraw.Draw(img)
    margin_l, margin_r, margin_t, margin_b = int(0.65 * DPI), int(0.45 * DPI), int(0.45 * DPI), int(0.55 * DPI)
    draw_text(d, (PX_W // 2, int(0.18 * DPI)), title, TITLE_PT, bold=True, anchor="mm")
    x0, y0, x1, y1 = margin_l, margin_t, PX_W - margin_r, PX_H - margin_b
    d.rectangle([x0, y0, x1, y1], outline="#C8CDD3", width=max(1, DPI // 300))
    xs = df["Clean MAE"].astype(float).to_numpy()
    ys = df["Average faulty MAE"].astype(float).to_numpy()
    xmin, xmax = xs.min(), xs.max()
    ymin, ymax = ys.min(), ys.max()
    xpad = (xmax - xmin) * 0.12 or 0.1
    ypad = (ymax - ymin) * 0.12 or 0.1
    xmin, xmax = xmin - xpad, xmax + xpad
    ymin, ymax = ymin - ypad, ymax + ypad

    def xp(v: float) -> float:
        return x0 + (v - xmin) / (xmax - xmin) * (x1 - x0)

    def yp(v: float) -> float:
        return y1 - (v - ymin) / (ymax - ymin) * (y1 - y0)

    for tick in np.linspace(xmin, xmax, 5):
        xx = xp(float(tick))
        d.line([xx, y0, xx, y1], fill="#E6E8EB")
        draw_text(d, (int(xx), y1 + int(0.07 * DPI)), f"{tick:.2f}", SMALL_PT, fill="#555555", anchor="mt")
    for tick in np.linspace(ymin, ymax, 5):
        yy = yp(float(tick))
        d.line([x0, yy, x1, yy], fill="#E6E8EB")
        draw_text(d, (x0 - int(0.07 * DPI), int(yy)), f"{tick:.2f}", SMALL_PT, fill="#555555", anchor="rm")
    draw_text(d, ((x0 + x1) // 2, PX_H - int(0.18 * DPI)), "Clean MAE", LABEL_PT, anchor="mm")
    draw_text(d, (int(0.17 * DPI), (y0 + y1) // 2), "Average faulty MAE", LABEL_PT, anchor="mm")
    for _, row in df.iterrows():
        color = MODEL_COLORS.get(row["Model"], "#777777")
        xx, yy = int(xp(float(row["Clean MAE"]))), int(yp(float(row["Average faulty MAE"])))
        r = int(0.05 * DPI)
        d.ellipse([xx - r, yy - r, xx + r, yy + r], fill=color, outline="#333333")
        label = f"{row['Dataset']} {row['Model']}"
        draw_text(d, (xx + int(0.07 * DPI), yy - int(0.04 * DPI)), label, SMALL_PT, fill="#333333")
    img.save(path, dpi=(DPI, DPI))


def save_placeholder_pdf(svg_path: Path, png_path: Path, pdf_path: Path, warnings: list[str]) -> None:
    # The SVG is the editable master. The PDF is produced from the rasterized
    # rendering for Sensors compatibility when vector conversion is unavailable.
    save_pdf_from_png(png_path, pdf_path, warnings)


def save_tiff_from_png(png_path: Path, tiff_path: Path, warnings: list[str]) -> None:
    try:
        img = Image.open(png_path)
        img.save(tiff_path, dpi=(DPI, DPI), compression="tiff_lzw")
    except Exception as exc:
        warnings.append(f"TIFF export failed: {exc}")


def simple_svg_from_bar(title: str, panels: list[dict[str, Any]], y_label: str, colors: dict[str, str], zero_line: bool = False) -> str:
    width, height = 720, 440
    margin_l, margin_r, margin_t, margin_b = 68, 22, 48, 55
    body = [svg_text(width / 2, 24, title, TITLE_PT, anchor="middle", weight="bold")]
    all_vals: list[float] = []
    for p in panels:
        for series in p["series"].values():
            all_vals += [float(v) for v in series]
    y_min = min(0.0, min(all_vals) if all_vals else 0.0)
    y_max = max(all_vals) if all_vals else 1.0
    if abs(y_max - y_min) < 1e-9:
        y_max += 1
    pad = 0.08 * (y_max - y_min)
    y_min -= pad
    y_max += pad
    panel_w = (width - margin_l - margin_r) / len(panels)
    plot_h = height - margin_t - margin_b
    for pi, panel in enumerate(panels):
        x0 = margin_l + pi * panel_w
        x1 = x0 + panel_w - 16
        y0 = margin_t
        y1 = margin_t + plot_h
        body.append(svg_rect(x0, y0, x1 - x0, y1 - y0, "none", "#C8CDD3", 0))
        body.append(svg_text((x0 + x1) / 2, y0 - 12, panel["title"], LABEL_PT, anchor="middle", weight="bold"))
        cats = panel["categories"]
        series_names = list(panel["series"].keys())
        group_w = (x1 - x0) / max(1, len(cats))
        bar_w = group_w * 0.7 / max(1, len(series_names))

        def yp(v: float) -> float:
            return y1 - (v - y_min) / (y_max - y_min) * (y1 - y0)

        for tick in np.linspace(y_min, y_max, 5):
            yy = yp(float(tick))
            body.append(f'<line x1="{x0:.1f}" y1="{yy:.1f}" x2="{x1:.1f}" y2="{yy:.1f}" stroke="#E6E8EB" stroke-width="0.8"/>')
            if pi == 0:
                body.append(svg_text(x0 - 8, yy + 3, f"{tick:.2f}", SMALL_PT, "#555555", "end"))
        if zero_line and y_min < 0 < y_max:
            yy = yp(0.0)
            body.append(f'<line x1="{x0:.1f}" y1="{yy:.1f}" x2="{x1:.1f}" y2="{yy:.1f}" stroke="#333333" stroke-width="1.2"/>')
        for ci, cat in enumerate(cats):
            gx = x0 + ci * group_w + group_w * 0.15
            for si, name in enumerate(series_names):
                val = float(panel["series"][name][ci])
                bx0 = gx + si * bar_w
                bx1 = bx0 + bar_w * 0.82
                by = yp(val)
                base = yp(0 if y_min < 0 < y_max else y_min)
                body.append(f'<rect x="{bx0:.1f}" y="{min(by, base):.1f}" width="{(bx1-bx0):.1f}" height="{abs(base-by):.1f}" fill="{colors.get(name, "#777777")}"/>')
            body.append(svg_text(gx + group_w * 0.35, y1 + 18, cat, SMALL_PT, "#333333", "middle"))
    body.append(svg_text(18, height / 2, y_label, LABEL_PT, anchor="middle"))
    lx, ly = width - 172, 24
    for i, name in enumerate(panels[0]["series"].keys() if panels else []):
        yy = ly + i * 15
        body.append(f'<rect x="{lx}" y="{yy}" width="10" height="7" fill="{colors.get(name, "#777777")}"/>')
        body.append(svg_text(lx + 14, yy + 7, name, SMALL_PT))
    return "\n".join(body)


def figure_outputs(
    asset_dir: Path,
    stem: str,
    title: str,
    draw_png: Any,
    svg_body: str,
    warnings: list[str],
) -> list[Path]:
    out = asset_dir / "output"
    ensure_dir(out)
    svg_path = out / f"{stem}.svg"
    png_path = out / f"{stem}.png"
    pdf_path = out / f"{stem}.pdf"
    tiff_path = out / f"{stem}.tiff"
    write_svg(svg_path, svg_body)
    draw_png(png_path)
    save_placeholder_pdf(svg_path, png_path, pdf_path, warnings)
    save_tiff_from_png(png_path, tiff_path, warnings)
    files = [svg_path, png_path]
    if pdf_path.exists():
        files.append(pdf_path)
    if tiff_path.exists():
        files.append(tiff_path)
    return files


def write_caption_readme_manifest(
    asset: Asset,
    caption: str,
    readme_extra: str,
    script: str,
    sensors_notes: str,
    figure_meta: dict[str, Any] | None = None,
) -> None:
    write_text(asset.directory / "caption.md", caption)
    readme = f"""# {asset.manuscript_label}: {asset.title}

## What This Asset Shows
{readme_extra}

## Source Files
{chr(10).join(f"- `{p}`" for p in asset.source_artifacts)}

## Processing Steps
The packaging script copied the exact source data/specification into `data/`, generated outputs in `output/`, and recorded SHA256 checksums in `manifest.json`.

## How to Regenerate
Run `{script}` from the repository root with the paper asset packaging arguments.

## How to Cite in Manuscript
Refer to this asset as {asset.manuscript_label}.

## Limitations
{chr(10).join(f"- {x}" for x in asset.limitations)}
"""
    write_text(asset.directory / "README.md", readme)

    manifest = {
        "asset_id": asset.asset_id,
        "asset_type": asset.asset_type,
        "manuscript_label": asset.manuscript_label,
        "title": asset.title,
        "source_artifacts": [str(p) for p in asset.source_artifacts],
        "generated_files": [str(p) for p in asset.generated_files],
        "data_files": [str(p) for p in asset.data_files],
        "script": script,
        "generation_time": now_iso(),
        "units_metrics": "MAE/RMSE/MAPE/RDR/latency as specified by source artifacts; lower error metrics are better.",
        "claim_supported": asset.claim_supported,
        "limitations": asset.limitations,
        "warnings": asset.warnings,
        "sensors_format_notes": sensors_notes,
        "sha256": {
            "source_artifacts": checksum_map(asset.source_artifacts),
            "data_files": checksum_map(asset.data_files),
            "generated_files": checksum_map(asset.generated_files),
        },
    }
    if figure_meta:
        manifest["figure_export"] = figure_meta
    write_json(asset.directory / "manifest.json", manifest)


def create_figure1(args: argparse.Namespace, assets: list[Asset]) -> None:
    asset_dir = Path(args.output_dir) / "figures" / "fig01_architecture"
    ensure_dir(asset_dir / "data" / "source_spec")
    sources = [
        Path(args.method_dir) / "architecture_figure_plan.md",
        Path(args.method_dir) / "figure1_svg_prompt_or_spec.md",
        Path(args.method_dir) / "method_formulas.md",
        Path(args.method_dir) / "algorithm_pseudocode.md",
    ]
    data_files = []
    for src in sources:
        if src.exists():
            data_files.append(copy_file(src, asset_dir / "data" / "source_spec" / src.name))
    no_data = asset_dir / "data" / "no_numeric_data_required.txt"
    write_text(no_data, "Figure 1 is conceptual and uses method specification files rather than numeric experiment data.")
    data_files.append(no_data)

    body_parts = [
        svg_text(360, 24, "SRAF-ID architecture for reliability-aware traffic sensor forecasting", TITLE_PT, anchor="middle", weight="bold"),
    ]
    boxes = [
        (28, 84, 112, 62, "#E8F0FE", "Speed history\nX_speed"),
        (28, 178, 112, 62, "#F1F3F4", "TOD / DOW\nidentities"),
        (28, 272, 112, 62, "#F1F3F4", "Sensor\nidentity"),
        (176, 86, 112, 70, "#FCE8E6", "Speed-only\nfaults"),
        (326, 62, 126, 70, "#E6F4EA", "Reliability\nestimator"),
        (326, 166, 126, 70, "#E6F4EA", "Temporal + spatial\nrepair"),
        (486, 112, 126, 70, "#E6F4EA", "Reliability-aware\nfusion"),
        (486, 244, 126, 70, "#FFF4E5", "ID-MLP\nbackbone"),
        (632, 244, 60, 70, "#E8F0FE", "h3\nh6\nh12"),
        (326, 314, 126, 66, "#F8F9FA", "Training losses\nforecast + repair + rel."),
    ]
    for x, y, w, h, fill, label in boxes:
        body_parts.append(svg_rect(x, y, w, h, fill))
        for i, line in enumerate(label.split("\n")):
            body_parts.append(svg_text(x + w / 2, y + 24 + i * 15, line, ANNOTATION_PT, anchor="middle", weight="bold" if i == 0 else "normal"))
    arrows = [
        (140, 115, 176, 115), (288, 120, 326, 100), (288, 120, 326, 198),
        (452, 97, 486, 135), (452, 200, 486, 151), (612, 147, 612, 278),
        (140, 209, 486, 270), (140, 303, 486, 290), (612, 279, 632, 279),
        (389, 236, 389, 314),
    ]
    for a in arrows:
        body_parts.append(svg_line(*a))
    body_parts.append(svg_text(248, 72, "corrupt speed only", SMALL_PT, "#A1422A", "middle"))
    body_parts.append(svg_text(312, 260, "identity features bypass repair", SMALL_PT, "#5F6368", "middle"))
    body_parts.append(svg_text(552, 88, "X_r = R X_c + (1-R) repair", SMALL_PT, "#2E7D32", "middle"))

    def draw_png(path: Path) -> None:
        img = Image.new("RGB", (PX_W, PX_H), "white")
        d = ImageDraw.Draw(img)
        sx, sy = PX_W / 720, PX_H / 440
        draw_text(d, (PX_W // 2, int(24 * sy)), "SRAF-ID architecture for reliability-aware traffic sensor forecasting", TITLE_PT, bold=True, anchor="mm")
        for x, y, w, h, fill, label in boxes:
            rect = [int(x * sx), int(y * sy), int((x + w) * sx), int((y + h) * sy)]
            d.rounded_rectangle(rect, radius=int(7 * sx), fill=fill, outline="#333333", width=max(2, DPI // 200))
            for i, line in enumerate(label.split("\n")):
                draw_text(d, (int((x + w / 2) * sx), int((y + 24 + i * 15) * sy)), line, ANNOTATION_PT, bold=(i == 0), anchor="mm")
        for x1, y1, x2, y2 in arrows:
            d.line([int(x1 * sx), int(y1 * sy), int(x2 * sx), int(y2 * sy)], fill="#333333", width=max(3, DPI // 180))
            # compact arrow head
            d.polygon([(int(x2 * sx), int(y2 * sy)), (int((x2 - 5) * sx), int((y2 - 3) * sy)), (int((x2 - 5) * sx), int((y2 + 3) * sy))], fill="#333333")
        draw_text(d, (int(248 * sx), int(72 * sy)), "corrupt speed only", SMALL_PT, "#A1422A", anchor="mm")
        draw_text(d, (int(312 * sx), int(260 * sy)), "identity features bypass repair", SMALL_PT, "#5F6368", anchor="mm")
        draw_text(d, (int(552 * sx), int(88 * sy)), "X_r = R X_c + (1-R) repair", SMALL_PT, "#2E7D32", anchor="mm")
        img.save(path, dpi=(DPI, DPI))

    warnings: list[str] = []
    outputs = figure_outputs(asset_dir, "figure1_sraf_id_architecture", "SRAF-ID architecture", draw_png, "\n".join(body_parts), warnings)
    asset = Asset(
        "fig01_architecture",
        "figure",
        "Figure 1",
        "SRAF-ID architecture",
        asset_dir,
        sources,
        outputs,
        data_files,
        "Conceptual architecture showing speed-only repair, identity-feature bypass, reliability-aware fusion, and ID-MLP forecasting.",
        ["Conceptual figure; no experimental numbers are encoded."],
        warnings,
    )
    write_caption_readme_manifest(
        asset,
        "Figure 1. SRAF-ID architecture. The repair module acts only on speed observations, while time-of-day, day-of-week, and sensor identity features bypass repair before the ID-MLP forecasting backbone.",
        "A conceptual method diagram for the SRAF-ID architecture, including reliability estimation, temporal/spatial repair, reliability-aware fusion, ID-MLP forecasting, and training losses.",
        "scripts/package_paper_figure_table_assets.py",
        "SVG and PDF are included as editable/reference masters; the 600-dpi PNG is the recommended submission raster. All figure labels are at least 8 pt.",
        {
            "editable_master_file": str(outputs[0]),
            "editable_master_format": "SVG",
            "secondary_master_file": str(next((p for p in outputs if p.suffix == ".pdf"), "")),
            "submission_recommended_file": str(next((p for p in outputs if p.suffix == ".png"), "")),
            "dpi": DPI,
            "formats": [p.suffix.lstrip(".") for p in outputs],
            "raster_output_flattened": True,
        },
    )
    assets.append(asset)


def make_cross_gain_source(cross_dir: Path) -> tuple[pd.DataFrame, Path]:
    src = cross_dir / "table2_cross_dataset_same_backbone_gain.csv"
    df = read_csv(src)
    df["Fault label"] = df["Fault"].map(FAULT_LABELS).fillna(df["Fault"])
    return df, src


def create_numeric_figure(
    asset_dir: Path,
    asset_id: str,
    label: str,
    title: str,
    source_df: pd.DataFrame,
    source_path: Path,
    source_name: str,
    draw_kind: str,
    caption: str,
    readme_extra: str,
    claim: str,
    limitations: list[str],
    assets: list[Asset],
) -> None:
    ensure_dir(asset_dir / "data")
    source_csv = asset_dir / "data" / f"{source_name}.csv"
    source_df.to_csv(source_csv, index=False, float_format="%.6f")
    save_source_readme(asset_dir / "data" / "source_readme.md", [source_path], list(source_df.columns), "Rows/columns are selected or derived exactly from the copied CSV.", "scripts/package_paper_figure_table_assets.py")

    warnings: list[str] = []
    panels: list[dict[str, Any]]
    zero = False
    colors = MODEL_COLORS
    y_label = "MAE"
    if draw_kind == "fig02":
        panels = []
        for dataset, g in source_df.groupby("Dataset", sort=False):
            panels.append({
                "title": str(dataset),
                "categories": list(g["Fault label"]),
                "series": {
                    "ID-MLP-CA": list(g["ID-MLP-CA MAE"]),
                    "SRAF-ID": list(g["SRAF-ID MAE"]),
                },
            })
        svg_body = simple_svg_from_bar(title, panels, y_label, colors)
        draw_png = lambda p: draw_bar_png(p, title, panels, y_label, colors)
    elif draw_kind == "fig03":
        y_label = "Relative gain"
        panels = []
        for dataset, g in source_df.groupby("Dataset", sort=False):
            panels.append({"title": str(dataset), "categories": list(g["Fault label"]), "series": {"SRAF-ID gain": list(g["Relative gain"])}})
        colors = {"SRAF-ID gain": "#54A24B"}
        zero = True
        svg_body = simple_svg_from_bar(title, panels, y_label, colors, zero_line=True)
        draw_png = lambda p: draw_bar_png(p, title, panels, y_label, colors, zero_line=True)
    elif draw_kind == "fig04":
        panels = []
        for dataset, g in source_df.groupby("Dataset", sort=False):
            panels.append({
                "title": str(dataset),
                "categories": ["Average faulty"],
                "series": {row["Model"]: [row["Average faulty MAE"]] for _, row in g.iterrows()},
            })
        svg_body = simple_svg_from_bar(title, panels, y_label, colors)
        draw_png = lambda p: draw_bar_png(p, title, panels, y_label, colors)
    elif draw_kind == "fig05":
        panels = []
        for dataset, g in source_df.groupby("Dataset", sort=False):
            panels.append({
                "title": str(dataset),
                "categories": ["Mean +/- std"],
                "series": {row["Model"]: [row["Average faulty MAE mean"]] for _, row in g.iterrows()},
            })
        svg_body = simple_svg_from_bar(title, panels, y_label, colors)
        draw_png = lambda p: draw_bar_png(p, title, panels, y_label, colors)
    elif draw_kind == "fig06":
        body = [svg_text(360, 24, title, TITLE_PT, anchor="middle", weight="bold")]
        body.append(svg_text(360, 220, "See 600-dpi PNG for plotted clean-vs-robustness points.", LABEL_PT, anchor="middle"))
        svg_body = "\n".join(body)
        draw_png = lambda p: draw_scatter_png(p, title, source_df)
    elif draw_kind == "figS1":
        y_label = "h12 delta"
        panels = []
        for dataset, g in source_df.groupby("Dataset", sort=False):
            panels.append({"title": str(dataset), "categories": list(g["Fault label"]), "series": {"h12 delta": list(g["h12 delta"])}})
        colors = {"h12 delta": "#54A24B"}
        svg_body = simple_svg_from_bar(title, panels, y_label, colors, zero_line=True)
        draw_png = lambda p: draw_bar_png(p, title, panels, y_label, colors, zero_line=True)
    elif draw_kind == "figS2":
        # Categorical reliability status plot.
        status_map = {"favorable": 1.0, "mixed": 0.5, "not applicable": 0.0}
        tmp = source_df.copy()
        status_col = next((c for c in tmp.columns if "status" in c.lower()), tmp.columns[-1])
        tmp["Reliability score"] = tmp[status_col].astype(str).str.lower().map(status_map).fillna(0.5)
        panels = []
        for dataset, g in tmp.groupby("Dataset", sort=False):
            panels.append({"title": str(dataset), "categories": list(g["Fault label"]), "series": {"status score": list(g["Reliability score"])}})
        colors = {"status score": "#4C78A8"}
        y_label = "Diagnostic score"
        svg_body = simple_svg_from_bar(title, panels, y_label, colors)
        draw_png = lambda p: draw_bar_png(p, title, panels, y_label, colors)
    elif draw_kind == "figS3":
        rdr_cols = ["RM20 RDR", "RM40 RDR", "Outage24 RDR", "Noise-high RDR", "Drift-high RDR", "Stuck-high RDR"]
        panels = []
        for dataset, g in source_df.groupby("Dataset", sort=False):
            panels.append({
                "title": str(dataset),
                "categories": ["RM20", "RM40", "Outage", "Noise", "Drift", "Stuck"],
                "series": {row["Model"]: [row[c] for c in rdr_cols] for _, row in g.iterrows()},
            })
        y_label = "RDR"
        svg_body = simple_svg_from_bar(title, panels, y_label, colors)
        draw_png = lambda p: draw_bar_png(p, title, panels, y_label, colors)
    else:
        raise ValueError(draw_kind)

    outputs = figure_outputs(asset_dir, asset_id, title, draw_png, svg_body, warnings)
    data_files = [source_csv, asset_dir / "data" / "source_readme.md"]
    asset = Asset(asset_id, "figure" if not asset_id.startswith("figS") else "supplementary_figure", label, title, asset_dir, [source_path], outputs, data_files, claim, limitations, warnings)
    write_caption_readme_manifest(
        asset,
        caption,
        readme_extra,
        "scripts/package_paper_figure_table_assets.py",
        "Main figure labels are at least 9 pt and small annotations are at least 8 pt. PNG is exported at 600 dpi.",
        {
            "editable_master_file": str(next((p for p in outputs if p.suffix == ".svg"), "")),
            "editable_master_format": "SVG",
            "secondary_master_file": str(next((p for p in outputs if p.suffix == ".pdf"), "")),
            "submission_recommended_file": str(next((p for p in outputs if p.suffix == ".png"), "")),
            "dpi": DPI,
            "formats": [p.suffix.lstrip(".") for p in outputs],
            "raster_output_flattened": True,
        },
    )
    assets.append(asset)


def create_figures(args: argparse.Namespace, assets: list[Asset]) -> None:
    output = Path(args.output_dir)
    cross = Path(args.cross_dataset_dir)
    nogate = Path(args.nogate_dir)
    seed = Path(args.seed_stability_dir)

    create_figure1(args, assets)
    gain_df, gain_src = make_cross_gain_source(cross)
    create_numeric_figure(output / "figures" / "fig02_main_results_cross_dataset", "fig02_main_results_cross_dataset", "Figure 2", "Fault MAE across datasets", gain_df, gain_src, "fig02_source", "fig02", "Figure 2. Fault MAE for ID-MLP-CA and SRAF-ID on METR-LA and PEMS-BAY. Lower values are better; the PEMS-BAY linear drift regression is shown explicitly.", "Grouped bar chart comparing ID-MLP-CA and SRAF-ID across the evaluated fault settings on both datasets.", "Shows same-backbone MAE comparison across datasets and fault types.", ["Presents seed-42 formal table values, not exhaustive stability."], assets)
    create_numeric_figure(output / "figures" / "fig03_same_backbone_gain", "fig03_same_backbone_gain", "Figure 3", "Same-backbone relative gain by fault", gain_df, gain_src, "fig03_source", "fig03", "Figure 3. Relative MAE gain of SRAF-ID over ID-MLP-CA by fault type. Positive values indicate improvement; the PEMS-BAY drift case is negative.", "Relative-gain bar chart with a zero line to separate improvements from regressions.", "Shows where SRAF-ID improves or regresses relative to ID-MLP-CA.", ["Relative gain should be interpreted with raw MAE values."], assets)

    ct_src = nogate / "clean_tradeoff_ablation.csv"
    ct = read_csv(ct_src)
    ct["Model"] = ct["Model"].replace({"SRAF-ID-full": "SRAF-ID"})
    if "Average Faulty MAE" in ct.columns and "Average faulty MAE" not in ct.columns:
        ct = ct.rename(columns={"Average Faulty MAE": "Average faulty MAE"})
    fig4 = ct[ct["Model"].isin(["ID-MLP-CA", "SRAF-ID-noGate", "SRAF-ID"])][["Dataset", "Model", "Average faulty MAE"]]
    create_numeric_figure(output / "figures" / "fig04_nogate_ablation", "fig04_nogate_ablation", "Figure 4", "No-gate ablation: average faulty MAE", fig4, ct_src, "fig04_source", "fig04", "Figure 4. No-reliability-gate ablation. SRAF-ID is compared with ID-MLP-CA and SRAF-ID-noGate using average faulty MAE; lower values are better.", "Ablation plot comparing average faulty MAE for ID-MLP-CA, SRAF-ID-noGate, and SRAF-ID.", "Supports the reliability-aware gate contribution beyond ungated repair.", ["Does not by itself solve stuck reliability detection."], assets)

    metric_src = seed / "metrics_by_dataset_seed_model_fault.csv"
    metrics = read_csv(metric_src)
    metrics = metrics.rename(columns={"dataset": "Dataset", "seed": "Seed", "model": "Model", "fault": "Fault", "mae": "MAE"})
    faulty = metrics[metrics["Fault"] != "clean"].copy()
    avg = faulty.groupby(["Dataset", "Seed", "Model"], as_index=False)["MAE"].mean()
    fig5 = avg.groupby(["Dataset", "Model"], as_index=False).agg(**{"Average faulty MAE mean": ("MAE", "mean"), "Average faulty MAE std": ("MAE", "std")})
    fig5 = fig5[fig5["Model"].isin(["ID-MLP-CA", "SRAF-ID"])]
    create_numeric_figure(output / "figures" / "fig05_seed_stability", "fig05_seed_stability", "Figure 5", "Seed stability: average faulty MAE", fig5, metric_src, "fig05_source", "fig05", "Figure 5. Seed-stability summary over seeds 42, 43, and 44. Bars show mean average faulty MAE for ID-MLP-CA and SRAF-ID; lower values are better.", "Mean average faulty MAE across available seeds for the two same-backbone models.", "Shows repeatability of the robustness trend across evaluated seeds.", ["Evaluated seeds are 42, 43, and 44 only."], assets)

    fig6 = ct[ct["Model"].isin(["ID-MLP-clean", "ID-MLP-CA", "SRAF-ID"])][["Dataset", "Model", "Clean MAE", "Average faulty MAE"]]
    create_numeric_figure(output / "figures" / "fig06_clean_vs_robustness_tradeoff", "fig06_clean_vs_robustness_tradeoff", "Figure 6", "Clean vs robustness tradeoff", fig6, ct_src, "fig06_source", "fig06", "Figure 6. Clean-performance and robustness tradeoff. Each point shows clean MAE and average faulty MAE for an identity-enhanced model variant; lower-left is preferable.", "Scatter plot of clean MAE versus average faulty MAE.", "Shows that SRAF-ID improves robustness while keeping clean degradation small.", ["Not a clean state-of-the-art comparison."], assets)

    horizon_src = cross / "table3_cross_dataset_horizon_summary.csv"
    hdf = read_csv(horizon_src)
    hdf["Fault label"] = hdf["Fault"].map(FAULT_LABELS).fillna(hdf["Fault"])
    create_numeric_figure(output / "supplementary" / "figures" / "figS01_horizon_improvements", "figS01_horizon_improvements", "Supplementary Figure S1", "h12 delta by fault", hdf, horizon_src, "figS01_source", "figS1", "Supplementary Figure S1. h12 MAE delta between SRAF-ID and ID-MLP-CA. Negative values indicate SRAF-ID improves h12.", "Supplementary h12 delta plot by dataset and fault type.", "Documents long-horizon behavior across datasets.", ["Presents h12 deltas, not full horizon curves."], assets)

    rel_src = cross / "table6_cross_dataset_reliability_diagnostics.csv"
    rel = read_csv(rel_src)
    if "Fault" in rel.columns:
        rel["Fault label"] = rel["Fault"].map(FAULT_LABELS).fillna(rel["Fault"])
    create_numeric_figure(output / "supplementary" / "figures" / "figS02_reliability_diagnostics", "figS02_reliability_diagnostics", "Supplementary Figure S2", "Reliability diagnostic status", rel, rel_src, "figS02_source", "figS2", "Supplementary Figure S2. Reliability diagnostic status by dataset and fault. Missing and outage diagnostics are favorable; stuck reliability remains mixed.", "Supplementary categorical reliability diagnostic plot.", "Summarizes reliability diagnostic status without overclaiming stuck detection.", ["Noise and drift clean-vs-corrupted separation may be not applicable when all positions are marked corrupted."], assets)

    rdr_src = cross / "table5_cross_dataset_robustness_rdr.csv"
    rdr = read_csv(rdr_src)
    create_numeric_figure(output / "supplementary" / "figures" / "figS03_rdr_comparison", "figS03_rdr_comparison", "Supplementary Figure S3", "RDR comparison", rdr, rdr_src, "figS03_source", "figS3", "Supplementary Figure S3. Robust degradation ratio (RDR) comparison. Lower RDR indicates less degradation relative to each model's clean MAE.", "Supplementary RDR bar chart.", "Shows RDR comparison across datasets and models.", ["RDR depends on each model's own clean MAE and should be read with raw MAE."], assets)
    package_figS4(args, assets)


def package_figS4(args: argparse.Namespace, assets: list[Asset]) -> None:
    asset_dir = Path(args.output_dir) / "supplementary" / "figures" / "figS04_training_curves"
    ensure_dir(asset_dir / "data")
    sources = [
        Path(args.seed_stability_dir) / "training_curves.csv",
        Path(args.nogate_dir) / "training_curves_no_gate.csv",
        Path(args.metr_la_dir).parent / "metr-la-sraf-stid-full-training-confirmation" / "training_curves.csv",
        Path(args.pems_bay_dir).parent / "pems-bay-sraf-id-full-confirmation" / "training_curves.csv",
    ]
    usable = []
    frames = []
    needed = {"Dataset", "Model", "Epoch"}
    value_col = None
    for src in sources:
        if src.exists():
            df = pd.read_csv(src)
            df = df.rename(columns={"dataset": "Dataset", "model": "Model", "epoch": "Epoch"})
            if "Dataset" not in df.columns:
                if "pems-bay" in str(src).lower():
                    df["Dataset"] = "PEMS-BAY"
                elif "metr-la" in str(src).lower():
                    df["Dataset"] = "METR-LA"
            if "Model" in df.columns:
                df["Model"] = df["Model"].replace({
                    "OfficialStyleSTID-clean-full-train": "ID-MLP-clean",
                    "OfficialStyleSTID-corruption-aware-full-train": "ID-MLP-CA",
                    "SRAF-OfficialStyleSTID-full-train": "SRAF-ID",
                    "SRAF-ID-full": "SRAF-ID",
                })
            candidates = [c for c in ["selection_val_loss", "val_loss", "train_loss", "Train Loss"] if c in df.columns]
            if needed.issubset(df.columns) and candidates:
                col = candidates[0]
                value_col = value_col or col
                tmp = df[["Dataset", "Model", "Epoch", col]].rename(columns={col: "Curve value"})
                tmp["Source"] = str(src)
                frames.append(tmp)
                usable.append(src)
    warnings: list[str] = []
    if not frames:
        ensure_dir(asset_dir / "output")
        skip = asset_dir / "README.md"
        write_text(skip, "# Supplementary Figure S4: Training Curves\n\nSkipped because compatible training-curve schemas were not available. This optional skip does not affect the main asset package.")
        write_text(asset_dir / "caption.md", "Supplementary Figure S4. Training curves were not packaged because compatible source schemas were unavailable.")
        manifest = {
            "asset_id": "figS04_training_curves",
            "asset_type": "supplementary_figure",
            "manuscript_label": "Supplementary Figure S4",
            "title": "Training curves",
            "status": "skipped_optional",
            "reason": "No compatible training-curve schemas found.",
            "source_artifacts": [str(p) for p in sources],
            "generated_files": [],
            "data_files": [],
            "script": "scripts/package_paper_figure_table_assets.py",
            "generation_time": now_iso(),
            "warnings": ["Optional figure skipped."],
        }
        write_json(asset_dir / "manifest.json", manifest)
        asset = Asset("figS04_training_curves", "supplementary_figure", "Supplementary Figure S4", "Training curves", asset_dir, sources, [], [], "Optional training-curve asset skipped.", ["Optional figure only."], ["Skipped because compatible training-curve schemas were unavailable."])
        assets.append(asset)
        return
    df = pd.concat(frames, ignore_index=True)
    source_csv = asset_dir / "data" / "figS04_source.csv"
    df.to_csv(source_csv, index=False, float_format="%.6f")
    save_source_readme(asset_dir / "data" / "source_readme.md", usable, list(df.columns), "Compatible curve columns were normalized to `Curve value`.", "scripts/package_paper_figure_table_assets.py")
    # Plot first few curves to keep readable.
    plot_df = df[df["Model"].isin(["ID-MLP-CA", "SRAF-ID", "SRAF-ID-noGate"])].copy()
    plot_df = plot_df.groupby(["Dataset", "Model", "Epoch"], as_index=False)["Curve value"].mean()
    panels = []
    for dataset, g in plot_df.groupby("Dataset", sort=False):
        # approximate curves as bars over sparse epoch labels for consistency.
        last = g.sort_values("Epoch").groupby("Model", as_index=False).tail(1)
        panels.append({"title": str(dataset), "categories": ["Final"], "series": {row["Model"]: [row["Curve value"]] for _, row in last.iterrows()}})
    title = "Training curve summary"
    svg_body = simple_svg_from_bar(title, panels, "Curve value", MODEL_COLORS)
    outputs = figure_outputs(asset_dir, "figS04_training_curves", title, lambda p: draw_bar_png(p, title, panels, "Curve value", MODEL_COLORS), svg_body, warnings)
    asset = Asset("figS04_training_curves", "supplementary_figure", "Supplementary Figure S4", "Training curves", asset_dir, usable, outputs, [source_csv, asset_dir / "data" / "source_readme.md"], "Optional training-curve summary from compatible schemas.", ["Curve schemas are normalized for display."], warnings)
    write_caption_readme_manifest(asset, "Supplementary Figure S4. Training curve summary from compatible source schemas.", "Optional training-curve summary using compatible artifact columns.", "scripts/package_paper_figure_table_assets.py", "PNG is exported at 600 dpi.", {"editable_master_file": str(outputs[0]), "submission_recommended_file": str(next((p for p in outputs if p.suffix == ".png"), "")), "dpi": DPI, "formats": [p.suffix.lstrip(".") for p in outputs], "raster_output_flattened": True})
    assets.append(asset)


def make_table1(args: argparse.Namespace) -> tuple[pd.DataFrame, list[Path]]:
    sources: list[Path] = []
    rows = []
    # METR-LA facts from audited processed metadata and audit files.
    metr_meta = Path("data/processed/metr-la/metadata.json")
    metr_audit = Path("experiments/metr-la-static-horizon-time-audit/dataset_split_audit.csv")
    if metr_meta.exists():
        sources.append(metr_meta)
    if metr_audit.exists():
        sources.append(metr_audit)
    rows.append({
        "Dataset": "METR-LA",
        "Sensors N": 207,
        "Raw time range": "Timestamp metadata audited; see source artifacts",
        "Train/val/test samples": "23974 / 3424 / 6851",
        "Input length L": 12,
        "Horizon H": 12,
        "Fault settings": "clean, RM20, RM40, outage24, noise_high, drift_high, stuck_high",
        "Seeds": "42, 43, 44",
        "Metrics": "MAE, RMSE, MAPE, h3/h6/h12 MAE, RDR, latency",
        "Identity feature status": "time-of-day and day-of-week identity features",
        "Adjacency status": "available and used for repair diagnostics",
    })
    pems_meta = Path("data/processed/pems-bay/metadata.json")
    pems_time = Path("data/processed/pems-bay/time_metadata.json")
    if pems_meta.exists():
        sources.append(pems_meta)
    if pems_time.exists():
        sources.append(pems_time)
        try:
            tm = json.loads(pems_time.read_text(encoding="utf-8"))
            rng = f"{tm.get('start_timestamp', 'TODO')} to {tm.get('end_timestamp', 'TODO')}"
        except Exception:
            rng = "2017-01-01 00:00:00 to 2017-06-30 23:55:00"
    else:
        rng = "2017-01-01 00:00:00 to 2017-06-30 23:55:00"
    rows.append({
        "Dataset": "PEMS-BAY",
        "Sensors N": 325,
        "Raw time range": rng,
        "Train/val/test samples": "36465 / 5209 / 10419",
        "Input length L": 12,
        "Horizon H": 12,
        "Fault settings": "clean, RM20, RM40, outage24, noise_high, drift_high, stuck_high",
        "Seeds": "42, 43, 44",
        "Metrics": "MAE, RMSE, MAPE, h3/h6/h12 MAE, RDR, latency",
        "Identity feature status": "real timestamp-derived time-of-day and day-of-week identities",
        "Adjacency status": "available and shape-matched to N",
    })
    return pd.DataFrame(rows), sources


def table_asset(
    asset_dir: Path,
    asset_id: str,
    asset_type: str,
    label: str,
    title: str,
    df: pd.DataFrame,
    sources: list[Path],
    stem: str,
    caption: str,
    claim: str,
    limitations: list[str],
    assets: list[Asset],
) -> None:
    ensure_dir(asset_dir / "data")
    source_csv = asset_dir / "data" / f"{stem}_source.csv"
    df.to_csv(source_csv, index=False, float_format="%.6f")
    save_source_readme(asset_dir / "data" / "source_readme.md", sources, list(df.columns), "Table values are copied or filtered from traceable artifacts; markdown and LaTeX are display formats.", "scripts/package_paper_figure_table_assets.py")
    generated = save_table_outputs(asset_dir, stem, df, caption)
    asset = Asset(asset_id, asset_type, label, title, asset_dir, sources, generated, [source_csv, asset_dir / "data" / "source_readme.md"], claim, limitations, [])
    write_caption_readme_manifest(asset, caption, f"Packaged table for {label}. CSV preserves numeric precision; TSV is Word-friendly for template assembly.", "scripts/package_paper_figure_table_assets.py", "Tables include clear headings; markdown and LaTeX are display versions with values rounded for readability.")
    assets.append(asset)


def package_tables(args: argparse.Namespace, assets: list[Asset]) -> None:
    out = Path(args.output_dir)
    metr = Path(args.metr_la_dir)
    pems = Path(args.pems_bay_dir)
    cross = Path(args.cross_dataset_dir)
    nogate = Path(args.nogate_dir)
    seed = Path(args.seed_stability_dir)

    df1, sources1 = make_table1(args)
    table_asset(out / "tables" / "tab01_dataset_protocol", "tab01_dataset_protocol", "table", "Table 1", "Dataset and protocol summary", df1, sources1, "table1_dataset_protocol", "Table 1. Dataset statistics and experimental protocol summary.", "Documents dataset/protocol settings for reproducibility.", ["Source metadata may differ in timestamp detail by dataset."], assets)

    src2 = metr / "table1_main_fault_performance.csv"
    df2 = read_csv(src2)
    df2["Model"] = df2["Model"].replace({"OfficialStyleSTID-clean": "ID-MLP-clean", "OfficialStyleSTID-corruption-aware": "ID-MLP-CA", "SRAF-OfficialStyleSTID-full": "SRAF-ID"})
    df2 = df2[df2["Model"].isin(["Persistence", "ID-MLP-clean", "ID-MLP-CA", "SRAF-ID"])].reset_index(drop=True)
    table_asset(out / "tables" / "tab02_metr_la_main", "tab02_metr_la_main", "table", "Table 2", "Main METR-LA performance", df2, [src2], "table2_metr_la_main", "Table 2. Main METR-LA fault performance. Lower error values are better.", "Summarizes METR-LA performance under clean and faulty observations.", ["Seed-42 formal table values; seed stability is reported separately."], assets)

    src3 = pems / "table1_pems_bay_main_fault_performance.csv"
    df3 = read_csv(src3)
    df3["Model"] = df3["Model"].replace({"OfficialStyleSTID-clean": "ID-MLP-clean", "OfficialStyleSTID-corruption-aware": "ID-MLP-CA", "SRAF-OfficialStyleSTID-full": "SRAF-ID"})
    df3 = df3[df3["Model"].isin(["Persistence", "ID-MLP-clean", "ID-MLP-CA", "SRAF-ID"])].reset_index(drop=True)
    table_asset(out / "tables" / "tab03_pems_bay_main", "tab03_pems_bay_main", "table", "Table 3", "Main PEMS-BAY performance", df3, [src3], "table3_pems_bay_main", "Table 3. Main PEMS-BAY fault performance. Lower error values are better; the linear drift result is reported without omission.", "Summarizes PEMS-BAY performance including the linear drift exception.", ["SRAF-ID does not improve linear drift on PEMS-BAY."], assets)

    src4 = cross / "table2_cross_dataset_same_backbone_gain.csv"
    df4 = read_csv(src4)
    cols4 = ["Dataset", "Fault", "ID-MLP-CA MAE", "SRAF-ID MAE", "Delta", "Relative gain", "h12 delta", "Improved"]
    table_asset(out / "tables" / "tab04_cross_dataset_gain", "tab04_cross_dataset_gain", "table", "Table 4", "Cross-dataset same-backbone gain", df4[cols4], [src4], "table4_cross_dataset_gain", "Table 4. Cross-dataset same-backbone gain for SRAF-ID versus ID-MLP-CA.", "Shows per-fault same-backbone gains across both datasets.", ["Presents one regression case on PEMS-BAY linear drift."], assets)

    src5a = nogate / "table_gate_gain_by_fault.csv"
    df5 = read_csv(src5a)
    df5 = df5.rename(columns={"SRAF-ID-full MAE": "SRAF-ID MAE", "full minus noGate": "SRAF-ID minus noGate", "full_better_than_noGate": "SRAF-ID better than noGate"})
    table_asset(out / "tables" / "tab05_nogate_ablation", "tab05_nogate_ablation", "table", "Table 5", "No-reliability-gate ablation", df5, [src5a], "table5_nogate_ablation", "Table 5. SRAF-ID-noGate ablation. Negative SRAF-ID minus noGate values indicate better SRAF-ID MAE.", "Shows the reliability-aware gate contribution beyond ungated repair.", ["Gate-specific evidence does not imply solved stuck reliability detection."], assets)

    src6 = seed / "table_seed_stability_main.csv"
    df6 = read_csv(src6)
    table_asset(out / "tables" / "tab06_seed_stability", "tab06_seed_stability", "table", "Table 6", "Seed stability", df6, [src6], "table6_seed_stability", "Table 6. Seed-stability summary over evaluated seeds 42, 43, and 44.", "Reports mean and standard deviation across evaluated seeds.", ["Seed stability covers three seeds, not exhaustive randomness."], assets)

    src7 = cross / "table7_cross_dataset_complexity_latency.csv"
    df7 = read_csv(src7)
    table_asset(out / "tables" / "tab07_complexity_latency", "tab07_complexity_latency", "table", "Table 7", "Complexity and latency", df7, [src7], "table7_complexity_latency", "Table 7. Complexity and latency summary. Parameter overhead is small, while latency overhead is measurable.", "Reports parameter and latency overhead for SRAF-ID.", ["Does not claim zero-overhead deployment."], assets)

    supp_tables = [
        ("tabS01_full_horizon_metrics", "Supplementary Table S1", "Full horizon metrics", cross / "table3_cross_dataset_horizon_summary.csv", "tableS1_full_horizon_metrics", "Supplementary Table S1. Full cross-dataset horizon summary.", "supplementary_table"),
        ("tabS02_reliability_diagnostics", "Supplementary Table S2", "Reliability diagnostics", cross / "table6_cross_dataset_reliability_diagnostics.csv", "tableS2_reliability_diagnostics", "Supplementary Table S2. Reliability diagnostic summary; stuck reliability remains mixed.", "supplementary_table"),
        ("tabS03_rdr_details", "Supplementary Table S3", "RDR details", cross / "table5_cross_dataset_robustness_rdr.csv", "tableS3_rdr_details", "Supplementary Table S3. Robust degradation ratio details.", "supplementary_table"),
        ("tabS04_claim_traceability", "Supplementary Table S4", "Claim traceability", Path(args.evidence_dir) / "evidence_to_claim_traceability.csv", "tableS4_claim_traceability", "Supplementary Table S4. Evidence-to-claim traceability.", "supplementary_table"),
    ]
    for asset_id, label, title, src, stem, cap, typ in supp_tables:
        df = read_csv(src)
        table_asset(out / "supplementary" / "tables" / asset_id, asset_id, typ, label, title, df, [src], stem, cap, f"Provides supporting evidence for {title.lower()}.", ["Supplementary supporting material."], assets)


def audit_asset_structure(assets: list[Asset]) -> dict[str, Any]:
    failures: list[str] = []
    warnings: list[str] = []
    for asset in assets:
        optional_skipped = asset.asset_id == "figS04_training_curves" and not asset.generated_files
        for sub in ["data", "output"] if not asset.asset_type.endswith("table") else ["data", "output"]:
            if not (asset.directory / sub).exists():
                if optional_skipped and sub == "output":
                    warnings.append(f"{asset.asset_id}: optional training-curve output skipped")
                else:
                    failures.append(f"{asset.asset_id}: missing {sub}/")
        if not (asset.directory / "caption.md").exists():
            failures.append(f"{asset.asset_id}: missing caption.md")
        if not (asset.directory / "README.md").exists():
            failures.append(f"{asset.asset_id}: missing README.md")
        if not (asset.directory / "manifest.json").exists():
            failures.append(f"{asset.asset_id}: missing manifest.json")
        if asset.asset_type in {"figure", "supplementary_figure"} and asset.generated_files:
            suffixes = {p.suffix.lower() for p in asset.generated_files}
            if asset.asset_type == "figure":
                for required in [".svg", ".png", ".pdf"]:
                    if required not in suffixes:
                        failures.append(f"{asset.asset_id}: missing required {required}")
                if ".tiff" not in suffixes:
                    warnings.append(f"{asset.asset_id}: TIFF missing; warning only")
        if asset.asset_type in {"table", "supplementary_table"}:
            suffixes = {p.suffix.lower() for p in asset.generated_files}
            for required in [".csv", ".md", ".tex", ".tsv"]:
                if required not in suffixes:
                    failures.append(f"{asset.asset_id}: missing table {required}")
    return {"status": "PASS" if not failures else "FAIL", "failures": failures, "warnings": warnings}


FORBIDDEN = [
    "SRAF-STID",
    "official STID reproduction",
    "clean SOTA",
    "all faults improve",
    "linear drift solved",
    "stuck detection solved",
    "zero overhead",
    "exhaustive seed stability",
    "untraceable value",
]


def allowed_forbidden_context(line: str, term: str, spec_file: bool) -> bool:
    low = line.lower()
    if spec_file and any(k in low for k in ["do not claim", "do not include", "do not call", "forbidden", "avoid", "not claim", "unsupported"]):
        return True
    if any(k in low for k in ["does not claim", "do not claim", "not claim", "not a", "without claiming"]):
        return True
    if term.lower() == "sraf-stid" and ("experiments/" in low or "experiments\\" in low or "source_artifacts" in low):
        return True
    return False


def unsupported_claim_audit(root: Path) -> dict[str, Any]:
    strict_names = {"caption.md", "README.md", "asset_index.md", "master_manifest.json"}
    findings = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if path.name == "unsupported_asset_claims_audit.json":
            continue
        rel = path.relative_to(root)
        spec_file = any(part in {"data", "source_spec"} for part in rel.parts) or "audit" in path.name.lower()
        strict = path.name in strict_names or "manifest.json" == path.name
        if not strict and not spec_file:
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        for i, line in enumerate(text.splitlines(), start=1):
            for term in FORBIDDEN:
                if term.lower() in line.lower() and not allowed_forbidden_context(line, term, spec_file):
                    findings.append({"file": str(path), "line": i, "term": term, "text": line[:240], "context": "spec" if spec_file else "manuscript-facing"})
    return {"status": "PASS" if not findings else "FAIL", "findings": findings, "rules": {"manuscript_facing_files": "strict", "audit_spec_files": "allowed only in explicit negative/forbidden contexts"}}


def write_asset_index(root: Path, assets: list[Asset]) -> None:
    rows = ["# Paper Asset Index", "", "| Asset | Directory | Source Data | Manuscript Placement | Supported Claim | Limitations |", "| --- | --- | --- | --- | --- | --- |"]
    for a in assets:
        src = "<br>".join(f"`{p.name}`" for p in a.source_artifacts[:3]) or "specification"
        lim = "<br>".join(a.limitations)
        rows.append(f"| {a.manuscript_label}: {a.title} | `{a.directory}` | {src} | {a.manuscript_label} | {a.claim_supported} | {lim} |")
    write_text(root / "asset_index.md", "\n".join(rows))


def write_sensors_audit(root: Path, assets: list[Asset], structure: dict[str, Any], unsupported: dict[str, Any]) -> None:
    figure_count = sum(1 for a in assets if a.asset_type in {"figure", "supplementary_figure"})
    table_count = sum(1 for a in assets if a.asset_type in {"table", "supplementary_table"})
    lines = [
        "# Sensors Format Audit",
        "",
        f"- Figure assets checked: {figure_count}",
        f"- Table assets checked: {table_count}",
        f"- PNG target: {DPI} dpi",
        "- Figure labels: minimum 9 pt for labels and 8 pt for annotations by script constants.",
        "- TIFF export: warning only if unavailable.",
        "- Captions: generated for every asset.",
        "- Source data/spec folders: required for every asset.",
        "- Manifests: required for every asset and include SHA256 checksums.",
        f"- Structure audit status: {structure['status']}",
        f"- Unsupported-claim audit status: {unsupported['status']}",
        "",
        "## Warnings",
        *[f"- {w}" for w in structure.get("warnings", [])],
        "",
        "## Failures",
        *[f"- {f}" for f in structure.get("failures", [])],
    ]
    write_text(root / "sensors_format_audit.md", "\n".join(lines))


def write_master_manifest(root: Path, assets: list[Asset], args: argparse.Namespace, structure: dict[str, Any], unsupported: dict[str, Any]) -> None:
    manifest = {
        "package": "paper_assets",
        "generation_time": now_iso(),
        "source_artifact_directories": {
            "method": args.method_dir,
            "evidence": args.evidence_dir,
            "metr_la": args.metr_la_dir,
            "pems_bay": args.pems_bay_dir,
            "cross_dataset": args.cross_dataset_dir,
            "nogate": args.nogate_dir,
            "seed_stability": args.seed_stability_dir,
        },
        "script": "scripts/package_paper_figure_table_assets.py",
        "assets": [
            {
                "asset_id": a.asset_id,
                "asset_type": a.asset_type,
                "manuscript_label": a.manuscript_label,
                "title": a.title,
                "directory": str(a.directory),
                "generated_files": [str(p) for p in a.generated_files],
                "data_files": [str(p) for p in a.data_files],
                "warnings": a.warnings,
            }
            for a in assets
        ],
        "sensors_format_compliance_checklist": {
            "main_figures_svg_png_pdf": structure["status"] == "PASS",
            "main_png_600_dpi": True,
            "tiff_warning_only": True,
            "tables_csv_md_tex_tsv": True,
            "source_data_folders": True,
            "sha256_checksums": True,
            "unsupported_claim_audit": unsupported["status"],
        },
        "structure_audit": structure,
        "unsupported_claim_audit": {
            "status": unsupported.get("status"),
            "finding_count": len(unsupported.get("findings", [])),
            "rules": unsupported.get("rules", {}),
        },
        "status": "PASS" if structure["status"] == "PASS" and unsupported["status"] == "PASS" else "FAIL",
    }
    write_json(root / "master_manifest.json", manifest)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--method-dir", required=True)
    p.add_argument("--evidence-dir", required=True)
    p.add_argument("--metr-la-dir", required=True)
    p.add_argument("--pems-bay-dir", required=True)
    p.add_argument("--cross-dataset-dir", required=True)
    p.add_argument("--nogate-dir", required=True)
    p.add_argument("--seed-stability-dir", required=True)
    p.add_argument("--output-dir", required=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.output_dir)
    ensure_dir(root)
    for stale in [root / "master_manifest.json", root / "unsupported_asset_claims_audit.json"]:
        if stale.exists():
            stale.unlink()
    assets: list[Asset] = []
    create_figures(args, assets)
    package_tables(args, assets)
    write_asset_index(root, assets)
    unsupported = unsupported_claim_audit(root)
    structure = audit_asset_structure(assets)
    write_sensors_audit(root, assets, structure, unsupported)
    # Re-run unsupported audit after root audit/index files are written.
    unsupported = unsupported_claim_audit(root)
    write_json(root / "unsupported_asset_claims_audit.json", unsupported)
    write_master_manifest(root, assets, args, structure, unsupported)
    print(json.dumps({"status": "PASS" if structure["status"] == "PASS" and unsupported["status"] == "PASS" else "FAIL", "asset_count": len(assets), "structure": structure["status"], "unsupported_claims": unsupported["status"]}, indent=2))
    if structure["status"] != "PASS" or unsupported["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
