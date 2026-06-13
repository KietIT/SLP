from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def read_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def pct(value: str | float) -> str:
    return f"{float(value) * 100:.1f}%"


def rounded_pct(value: str | float) -> float:
    return round(float(value) * 100, 1)


def norm(text: str) -> str:
    return " ".join((text or "").lower().split())


def disagreement_score(row: dict) -> int:
    ref = norm(row.get("text", ""))
    hyp = norm(row.get("prediction", ""))
    return len(set(ref.split()).symmetric_difference(set(hyp.split()))) + abs(len(ref) - len(hyp))


def js_string(value: str) -> str:
    return json.dumps(value, ensure_ascii=False)


def write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def find_first_existing(paths: list[Path]) -> Path:
    for path in paths:
        if path.exists():
            return path
    raise FileNotFoundError(paths[0])


def count_prediction_changes(before_path: Path, after_path: Path) -> tuple[int, int]:
    if not before_path.exists() or not after_path.exists():
        return (0, 0)
    before = read_csv(before_path)
    after = read_csv(after_path)
    total = min(len(before), len(after))
    changed = sum(
        1 for i in range(total)
        if before[i].get("prediction", "") != after[i].get("prediction", "")
    )
    return changed, total


COMMON_JS = r'''
const C = {
  ink: "#132033",
  muted: "#667085",
  faint: "#F2F5F7",
  paper: "#FBFAF7",
  rule: "#D8DEE6",
  blue: "#2F6FED",
  teal: "#16A3A3",
  green: "#168A5B",
  amber: "#B7791F",
  red: "#C2413A",
  plum: "#6D4AFF",
  navy: "#0E2A47",
  white: "#FFFFFF",
};

export function bg(slide, ctx, dark = false) {
  ctx.addShape(slide, { x: 0, y: 0, width: ctx.W, height: ctx.H, fill: dark ? C.navy : C.paper, line: ctx.line("#00000000", 0) });
}

export function footer(slide, ctx, n, dark = false) {
  ctx.addText(slide, {
    x: 54, y: 686, width: 900, height: 20,
    text: "VIVOS + MUSAN | Whisper-base zero-shot + forced VI decoding | seed=42 | subset results",
    fontSize: 11, color: dark ? "#C9D4E5" : C.muted,
  });
  ctx.addText(slide, {
    x: 1188, y: 684, width: 40, height: 22,
    text: String(n).padStart(2, "0"),
    fontSize: 11, color: dark ? "#C9D4E5" : C.muted, align: "right",
  });
}

export function kicker(slide, ctx, label, dark = false) {
  ctx.addShape(slide, { x: 54, y: 42, width: 8, height: 8, fill: dark ? C.teal : C.blue, line: ctx.line("#00000000", 0) });
  ctx.addText(slide, { x: 74, y: 33, width: 420, height: 28, text: label.toUpperCase(), fontSize: 12, bold: true, color: dark ? "#B7C7DD" : C.muted });
}

export function title(slide, ctx, text, dark = false) {
  ctx.addText(slide, {
    x: 54, y: 70, width: 930, height: 96,
    text, fontSize: 38, bold: true, color: dark ? C.white : C.ink,
    typeface: ctx.fonts.title, insets: { left: 0, right: 0, top: 0, bottom: 0 },
  });
}

export function body(slide, ctx, text, x, y, w, h, dark = false, size = 20) {
  ctx.addText(slide, {
    x, y, width: w, height: h, text, fontSize: size,
    color: dark ? "#D8E2F0" : C.ink, insets: { left: 0, right: 0, top: 0, bottom: 0 },
  });
}

export function metric(slide, ctx, x, y, w, label, value, note, color = C.blue, dark = false) {
  ctx.addShape(slide, { x, y, width: w, height: 118, fill: dark ? "#163956" : C.white, line: ctx.line(dark ? "#35546D" : C.rule, 1) });
  ctx.addText(slide, { x: x + 18, y: y + 16, width: w - 36, height: 24, text: label, fontSize: 12, bold: true, color: dark ? "#AFC0D6" : C.muted });
  ctx.addText(slide, { x: x + 18, y: y + 42, width: w - 36, height: 42, text: value, fontSize: 30, bold: true, color });
  ctx.addText(slide, { x: x + 18, y: y + 84, width: w - 36, height: 24, text: note, fontSize: 12, color: dark ? "#C9D4E5" : C.muted });
}

export function bullet(slide, ctx, x, y, text, dark = false, color = C.blue) {
  ctx.addShape(slide, { x, y: y + 8, width: 7, height: 7, fill: color, line: ctx.line("#00000000", 0) });
  ctx.addText(slide, { x: x + 18, y, width: 520, height: 38, text, fontSize: 18, color: dark ? "#E6EEF8" : C.ink });
}

export function flowNode(slide, ctx, x, y, w, label, note, color = C.blue) {
  ctx.addShape(slide, { x, y, width: w, height: 86, fill: C.white, line: ctx.line(color, 2) });
  ctx.addText(slide, { x: x + 14, y: y + 13, width: w - 28, height: 26, text: label, fontSize: 18, bold: true, color: C.ink });
  ctx.addText(slide, { x: x + 14, y: y + 43, width: w - 28, height: 34, text: note, fontSize: 13, color: C.muted });
}

export function arrow(slide, ctx, x1, y, x2, color = C.rule) {
  ctx.addShape(slide, { x: x1, y, width: x2 - x1, height: 2, fill: color, line: ctx.line(color, 0) });
  ctx.addShape(slide, { x: x2 - 8, y: y - 5, width: 10, height: 10, fill: color, line: ctx.line(color, 0) });
}

export function barChart(slide, ctx, x, y, w, h, data, maxValue, color = C.blue, label = "%") {
  const rowH = h / data.length;
  for (let i = 0; i < data.length; i++) {
    const d = data[i];
    const yy = y + i * rowH;
    ctx.addText(slide, { x, y: yy + 4, width: 92, height: 28, text: d.name, fontSize: 16, bold: true, color: C.ink, align: "right" });
    ctx.addShape(slide, { x: x + 112, y: yy + 7, width: w - 190, height: 20, fill: "#E8EDF3", line: ctx.line("#00000000", 0) });
    ctx.addShape(slide, { x: x + 112, y: yy + 7, width: Math.max(3, (w - 190) * d.value / maxValue), height: 20, fill: d.color || color, line: ctx.line("#00000000", 0) });
    ctx.addText(slide, { x: x + w - 66, y: yy + 2, width: 64, height: 30, text: `${d.value.toFixed(1)}${label}`, fontSize: 15, bold: true, color: C.ink, align: "right" });
  }
}

export function smallTable(slide, ctx, x, y, cols, rows) {
  const colW = [120, 92, 92, 92, 108, 108];
  const rowH = 34;
  let xx = x;
  for (let c = 0; c < cols.length; c++) {
    ctx.addShape(slide, { x: xx, y, width: colW[c], height: rowH, fill: "#E8EEF5", line: ctx.line(C.rule, 1) });
    ctx.addText(slide, { x: xx + 8, y: y + 8, width: colW[c] - 16, height: 18, text: cols[c], fontSize: 11, bold: true, color: C.ink, align: c === 0 ? "left" : "center" });
    xx += colW[c];
  }
  for (let r = 0; r < rows.length; r++) {
    xx = x;
    for (let c = 0; c < cols.length; c++) {
      ctx.addShape(slide, { x: xx, y: y + rowH * (r + 1), width: colW[c], height: rowH, fill: C.white, line: ctx.line(C.rule, 1) });
      ctx.addText(slide, { x: xx + 8, y: y + rowH * (r + 1) + 8, width: colW[c] - 16, height: 18, text: String(rows[r][c]), fontSize: 11, color: C.ink, align: c === 0 ? "left" : "center" });
      xx += colW[c];
    }
  }
}

export { C };
'''


def slide_module(n: int, body: str) -> str:
    return f'import {{ bg, footer, kicker, title, body, metric, bullet, flowNode, arrow, barChart, smallTable, C }} from "./common.mjs";\n\nexport async function slide{n:02d}(presentation, ctx) {{\n  const slide = presentation.slides.add();\n{body}\n  return slide;\n}}\n'


def main() -> None:
    p = argparse.ArgumentParser(description="Prepare artifact-tool slide modules for the midterm deck.")
    p.add_argument("--outputs_dir", default="outputs")
    p.add_argument("--workspace", default="outputs/manual-midterm/presentations/tone-asr-midterm")
    args = p.parse_args()

    outputs = Path(args.outputs_dir)
    workspace = Path(args.workspace)
    slides = workspace / "slides"
    (workspace / "preview").mkdir(parents=True, exist_ok=True)
    (workspace / "layout").mkdir(parents=True, exist_ok=True)
    (workspace / "qa").mkdir(parents=True, exist_ok=True)
    slides.mkdir(parents=True, exist_ok=True)

    dataset = read_csv(outputs / "dataset_stats.csv")
    base_clean = read_csv(outputs / "metrics_whisper_clean.csv")
    base_noisy = read_csv(outputs / "metrics_whisper_noisy_by_snr.csv")
    clean = read_csv(find_first_existing([outputs / "metrics_whisper_clean_forced.csv", outputs / "metrics_whisper_clean.csv"]))
    noisy = read_csv(find_first_existing([outputs / "metrics_whisper_noisy_forced_by_snr.csv", outputs / "metrics_whisper_noisy_by_snr.csv"]))
    preds = read_csv(find_first_existing([outputs / "whisper_noisy_forced.csv", outputs / "whisper_noisy.csv"]))
    before_preds = read_csv(outputs / "whisper_noisy.csv") if (outputs / "whisper_noisy.csv").exists() else []
    clean_changed, clean_total = count_prediction_changes(outputs / "whisper_clean.csv", outputs / "whisper_clean_forced.csv")
    noisy_changed, noisy_total = count_prediction_changes(outputs / "whisper_noisy.csv", outputs / "whisper_noisy_forced.csv")
    clean_all = next(r for r in clean if r["group"] == "all")
    noisy_all = next(r for r in noisy if r["group"] == "all")
    base_clean_all = next(r for r in base_clean if r["group"] == "all")
    base_noisy_all = next(r for r in base_noisy if r["group"] == "all")
    by_snr = [r for r in noisy if r["group"] != "all"]
    by_snr.sort(key=lambda r: float(r["group"]), reverse=True)
    examples = [r for r in preds if norm(r["text"]) != norm(r["prediction"])]
    examples.sort(key=disagreement_score, reverse=True)
    examples = examples[:2]
    before_by_utt = {r.get("utt_id", ""): r for r in before_preds}
    decoder_case = next(
        (
            {
                "utt_id": row.get("utt_id", ""),
                "snr": row.get("snr", ""),
                "text": row.get("text", ""),
                "before": before_by_utt[row.get("utt_id", "")].get("prediction", ""),
                "after": row.get("prediction", ""),
            }
            for row in preds
            if row.get("utt_id", "") in before_by_utt
            and row.get("prediction", "") != before_by_utt[row.get("utt_id", "")].get("prediction", "")
        ),
        {
            "utt_id": examples[0].get("utt_id", ""),
            "snr": examples[0].get("snr", ""),
            "text": examples[0].get("text", ""),
            "before": "",
            "after": examples[0].get("prediction", ""),
        },
    )

    data = {
        "clean_wer_before": rounded_pct(base_clean_all["wer"]),
        "clean_wer": rounded_pct(clean_all["wer"]),
        "clean_cer_before": rounded_pct(base_clean_all["cer"]),
        "clean_cer": rounded_pct(clean_all["cer"]),
        "clean_ter": rounded_pct(clean_all["ter_simple"]),
        "clean_der": rounded_pct(clean_all["der_simple"]),
        "noisy_wer_before": rounded_pct(base_noisy_all["wer"]),
        "noisy_wer": rounded_pct(noisy_all["wer"]),
        "noisy_cer_before": rounded_pct(base_noisy_all["cer"]),
        "noisy_cer": rounded_pct(noisy_all["cer"]),
        "noisy_ter": rounded_pct(noisy_all["ter_simple"]),
        "noisy_der": rounded_pct(noisy_all["der_simple"]),
        "prediction_changes": {
            "clean": {"changed": clean_changed, "total": clean_total},
            "noisy": {"changed": noisy_changed, "total": noisy_total},
        },
        "snr": [
            {
                "name": r["group"].replace(".0", " dB"),
                "wer": rounded_pct(r["wer"]),
                "cer": rounded_pct(r["cer"]),
                "ter": rounded_pct(r["ter_simple"]),
                "der": rounded_pct(r["der_simple"]),
            }
            for r in by_snr
        ],
        "examples": examples,
        "decoder_case": decoder_case,
        "dataset": dataset,
    }
    write(workspace / "data.json", json.dumps(data, ensure_ascii=False, indent=2))
    write(slides / "common.mjs", COMMON_JS)

    slide_bodies = []
    slide_bodies.append(f'''
  bg(slide, ctx, true);
  kicker(slide, ctx, "Midterm Progress", true);
  ctx.addText(slide, {{ x: 54, y: 100, width: 940, height: 130, text: "Tone-aware Vietnamese ASR pipeline is now runnable end-to-end.", fontSize: 48, bold: true, color: C.white, typeface: ctx.fonts.title }});
  body(slide, ctx, "Mentor: Vu Ha Anh | Team: Nguyen Xuan Trung, Trinh Vy Kiet, Nguyen Thanh Phat, Pham Hoang Phuc", 58, 232, 900, 32, true, 17);
  body(slide, ctx, "VIVOS clean speech + MUSAN noise -> controlled SNR benchmark -> Whisper-base baseline with forced VI decoding -> WER/CER/TER/DER tables.", 58, 274, 840, 70, true, 22);
  metric(slide, ctx, 58, 374, 260, "Clean WER", "{pct(clean_all['wer'])}", "30 VIVOS utterances", C.teal, true);
  metric(slide, ctx, 342, 374, 260, "Noisy WER", "{pct(noisy_all['wer'])}", "120 noisy utterances", C.amber, true);
  metric(slide, ctx, 626, 374, 260, "Worst SNR WER", "{pct(next(r for r in by_snr if r['group'] == '0.0')['wer'])}", "0 dB condition", C.red, true);
  ctx.addText(slide, {{ x: 940, y: 382, width: 250, height: 90, text: "Safe claim: pipeline + baseline + small decoding fix, not final MTL improvement.", fontSize: 20, bold: true, color: "#D8E2F0" }});
  footer(slide, ctx, 1, true);
''')

    slide_bodies.append('''
  bg(slide, ctx);
  kicker(slide, ctx, "Research Gap");
  title(slide, ctx, "Vietnamese ASR needs noise robustness and tone-aware evaluation.");
  bullet(slide, ctx, 76, 208, "Vietnamese tones and diacritics change lexical meaning; WER/CER alone hide important failure modes.", false, C.blue);
  bullet(slide, ctx, 76, 274, "Real speech settings include traffic, cafe, classroom, fan, rain, and overlapping noise.", false, C.teal);
  bullet(slide, ctx, 76, 340, "The proposal's core question remains: does explicit tone supervision improve robustness beyond ordinary noisy LoRA?", false, C.plum);
  metric(slide, ctx, 760, 206, 300, "Midterm boundary", "Baseline first", "No overclaim before MTL training", C.blue);
  metric(slide, ctx, 760, 350, 300, "Next proof", "E3 vs E4", "Noisy LoRA vs tone-aware MTL", C.plum);
  footer(slide, ctx, 2);
''')

    slide_bodies.append(f'''
  bg(slide, ctx);
  kicker(slide, ctx, "Data Benchmark");
  title(slide, ctx, "A fixed VIVOS + MUSAN subset gives reproducible noisy-ASR evidence.");
  flowNode(slide, ctx, 70, 210, 220, "VIVOS test", "760 clean utterances\\n0.746 hours", C.blue);
  arrow(slide, ctx, 300, 252, 378);
  flowNode(slide, ctx, 390, 210, 250, "MUSAN noise", "930 noise files\\nnoise_type metadata", C.teal);
  arrow(slide, ctx, 650, 252, 728);
  flowNode(slide, ctx, 740, 210, 280, "Controlled SNR", "50 utterances x 4 SNR\\n20 / 10 / 5 / 0 dB", C.amber);
  ctx.addText(slide, {{ x: 70, y: 372, width: 950, height: 78, text: "The generated noisy manifest stores source_audio, noise_path, noise_type, snr, and seed=42, so the subset can be rerun and audited.", fontSize: 22, color: C.ink }});
  metric(slide, ctx, 70, 500, 250, "Clean manifest", "760", "VIVOS test rows", C.blue);
  metric(slide, ctx, 350, 500, 250, "Noisy rows", "200", "50 per SNR", C.amber);
  metric(slide, ctx, 630, 500, 250, "Avg duration", "3.31s", "noisy subset", C.teal);
  footer(slide, ctx, 3);
''')

    slide_bodies.append('''
  bg(slide, ctx);
  kicker(slide, ctx, "Implementation");
  title(slide, ctx, "The repo now runs a complete data-to-metrics path.");
  flowNode(slide, ctx, 70, 215, 190, "Manifest", "VIVOS/MUSAN JSONL", C.blue);
  arrow(slide, ctx, 270, 257, 325);
  flowNode(slide, ctx, 337, 215, 190, "Noise mix", "fixed seed + SNR", C.amber);
  arrow(slide, ctx, 537, 257, 592);
  flowNode(slide, ctx, 604, 215, 190, "Inference", "Whisper-base\\nforced VI", C.teal);
  arrow(slide, ctx, 804, 257, 859);
  flowNode(slide, ctx, 871, 215, 220, "Metrics", "WER/CER/TER/DER", C.plum);
  bullet(slide, ctx, 84, 382, "Validated artifacts: dataset_stats.csv, forced prediction CSVs, forced metrics CSVs, midterm_summary.md.", false, C.green);
  bullet(slide, ctx, 84, 438, "Code fix: infer.py now passes language=vi and task=transcribe into model.generate().", false, C.blue);
  bullet(slide, ctx, 84, 494, "Hardware note: current run used CPU inference; LoRA/MTL training should move to Colab GPU.", false, C.red);
  footer(slide, ctx, 4);
''')

    slide_bodies.append(f'''
  bg(slide, ctx);
  kicker(slide, ctx, "Baseline Result");
  title(slide, ctx, "Forced Vietnamese decoding helps slightly, but adaptation is still needed.");
  barChart(slide, ctx, 86, 238, 520, 130, [
    {{ name: "Clean old", value: {rounded_pct(base_clean_all["wer"])}, color: C.blue }},
    {{ name: "Clean forced", value: {rounded_pct(clean_all["wer"])}, color: C.teal }},
    {{ name: "Noisy old", value: {rounded_pct(base_noisy_all["wer"])}, color: C.amber }},
    {{ name: "Noisy forced", value: {rounded_pct(noisy_all["wer"])}, color: C.red }},
  ], 60, C.blue, "%");
  metric(slide, ctx, 720, 230, 260, "Clean WER", "{pct(clean_all['wer'])}", "{clean_changed}/{clean_total} predictions changed", C.teal);
  metric(slide, ctx, 720, 376, 260, "Noisy WER", "{pct(noisy_all['wer'])}", "{noisy_changed}/{noisy_total} predictions changed", C.amber);
  ctx.addText(slide, {{ x: 86, y: 480, width: 610, height: 76, text: "Takeaway: decoder forcing removed a few non-Vietnamese outputs; noisy WER changed only {pct(base_noisy_all['wer'])} -> {pct(noisy_all['wer'])}.", fontSize: 22, color: C.ink }});
  footer(slide, ctx, 5);
''')

    wer_items = ",\n    ".join([f'{{ name: "{r["group"].replace(".0", " dB")}", value: {rounded_pct(r["wer"])}, color: {json.dumps("#2F6FED" if r["group"] == "20.0" else "#16A3A3" if r["group"] == "10.0" else "#B7791F" if r["group"] == "5.0" else "#C2413A")} }}' for r in by_snr])
    slide_bodies.append(f'''
  bg(slide, ctx);
  kicker(slide, ctx, "SNR Sensitivity");
  title(slide, ctx, "The 0 dB condition is the clearest stress case.");
  barChart(slide, ctx, 86, 214, 620, 220, [
    {wer_items}
  ], 60, C.blue, "%");
  ctx.addText(slide, {{ x: 770, y: 230, width: 330, height: 95, text: "WER by SNR", fontSize: 28, bold: true, color: C.ink }});
  ctx.addText(slide, {{ x: 770, y: 304, width: 330, height: 110, text: "20 dB remains close to clean, while 0 dB pushes WER above 53%. This gives a concrete target for noisy LoRA and tone-aware MTL.", fontSize: 21, color: C.ink }});
  footer(slide, ctx, 6);
''')

    rows = [[r["group"].replace(".0", " dB"), r["n"], pct(r["wer"]), pct(r["cer"]), pct(r["ter_simple"]), pct(r["der_simple"])] for r in by_snr]
    slide_bodies.append(f'''
  bg(slide, ctx);
  kicker(slide, ctx, "Vietnamese Metrics");
  title(slide, ctx, "TER/DER prototypes expose Vietnamese-specific failure modes.");
  smallTable(slide, ctx, 68, 220, ["SNR", "N", "WER", "CER", "TER", "DER"], {json.dumps(rows, ensure_ascii=False)});
  ctx.addText(slide, {{ x: 760, y: 224, width: 330, height: 160, text: "TER and DER are still prototype metrics, but they already let us discuss tone and diacritic behavior separately from generic WER/CER.", fontSize: 22, color: C.ink }});
  ctx.addText(slide, {{ x: 760, y: 408, width: 330, height: 105, text: "Post-midterm work: replace simple position matching with edit-aligned syllable metrics.", fontSize: 20, color: C.muted }});
  footer(slide, ctx, 7);
''')

    remaining_error = examples[0]
    slide_bodies.append(f'''
  bg(slide, ctx);
  kicker(slide, ctx, "Error Analysis");
  title(slide, ctx, "Decoder control fixes language drift, but acoustic errors remain.");
  ctx.addShape(slide, {{ x: 72, y: 204, width: 500, height: 300, fill: C.white, line: ctx.line(C.rule, 1) }});
  ctx.addText(slide, {{ x: 96, y: 224, width: 452, height: 24, text: "Decoder case | SNR {decoder_case.get('snr')} dB", fontSize: 15, bold: true, color: C.blue }});
  ctx.addText(slide, {{ x: 96, y: 258, width: 452, height: 56, text: {js_string("Ref: " + decoder_case.get('text', ''))}, fontSize: 16, color: C.ink }});
  ctx.addText(slide, {{ x: 96, y: 326, width: 452, height: 52, text: {js_string("Before: " + decoder_case.get('before', '').strip())}, fontSize: 16, color: C.red }});
  ctx.addText(slide, {{ x: 96, y: 398, width: 452, height: 52, text: {js_string("Forced VI: " + decoder_case.get('after', '').strip())}, fontSize: 16, color: C.green }});
  ctx.addShape(slide, {{ x: 628, y: 204, width: 500, height: 300, fill: C.white, line: ctx.line(C.rule, 1) }});
  ctx.addText(slide, {{ x: 652, y: 224, width: 452, height: 24, text: "Remaining ASR error | SNR {remaining_error.get('snr')} dB", fontSize: 15, bold: true, color: C.blue }});
  ctx.addText(slide, {{ x: 652, y: 262, width: 452, height: 88, text: {js_string("Ref: " + remaining_error.get('text', ''))}, fontSize: 16, color: C.ink }});
  ctx.addText(slide, {{ x: 652, y: 368, width: 452, height: 92, text: {js_string("Pred: " + remaining_error.get('prediction', '').strip())}, fontSize: 16, color: C.red }});
  footer(slide, ctx, 8);
''')

    slide_bodies.append('''
  bg(slide, ctx);
  kicker(slide, ctx, "Status");
  title(slide, ctx, "Midterm status is green for pipeline, yellow for model proof.");
  ctx.addText(slide, { x: 84, y: 210, width: 260, height: 34, text: "Done", fontSize: 28, bold: true, color: C.green });
  bullet(slide, ctx, 92, 266, "VIVOS + MUSAN manifests", false, C.green);
  bullet(slide, ctx, 92, 320, "Noisy benchmark with SNR metadata", false, C.green);
  bullet(slide, ctx, 92, 374, "Whisper-base baseline with forced VI decoding", false, C.green);
  bullet(slide, ctx, 92, 428, "Before/after WER-CER comparison", false, C.green);
  ctx.addText(slide, { x: 650, y: 210, width: 300, height: 34, text: "Next", fontSize: 28, bold: true, color: C.amber });
  bullet(slide, ctx, 658, 266, "PhoWhisper zero-shot on same subset", false, C.amber);
  bullet(slide, ctx, 658, 320, "Clean LoRA and noisy LoRA on Colab", false, C.amber);
  bullet(slide, ctx, 658, 374, "Tone-aware MTL vs noisy LoRA", false, C.amber);
  bullet(slide, ctx, 658, 428, "Edit-aligned TER/DER", false, C.amber);
  footer(slide, ctx, 9);
''')

    slide_bodies.append('''
  bg(slide, ctx, true);
  kicker(slide, ctx, "Conclusion", true);
  ctx.addText(slide, { x: 58, y: 96, width: 920, height: 112, text: "We should keep the research direction, but present midterm as reproducible progress.", fontSize: 44, bold: true, color: C.white, typeface: ctx.fonts.title });
  metric(slide, ctx, 70, 278, 270, "Deliverable", "Pipeline", "data -> noise -> ASR -> metrics", C.teal, true);
  metric(slide, ctx, 380, 278, 270, "Evidence", "Baseline", "forced-VI VIVOS/MUSAN outputs", C.amber, true);
  metric(slide, ctx, 690, 278, 270, "After midterm", "MTL proof", "compare E3 vs E4 on Colab", C.plum, true);
  ctx.addText(slide, { x: 74, y: 478, width: 900, height: 60, text: "Recommended wording: current contribution is a reproducible noisy-ASR benchmark and baseline. The tone-aware MTL improvement claim remains the next experiment.", fontSize: 24, color: "#D8E2F0" });
  footer(slide, ctx, 10, true);
''')

    for i, body in enumerate(slide_bodies, 1):
        write(slides / f"slide-{i:02d}.mjs", slide_module(i, body))

    write(workspace / "profile-plan.txt", "task mode: create\nprimary deck-profile: engineering-platform\nrequired proof objects: pipeline workflow, benchmark metrics, error examples, risk/next-step slide\n")
    write(workspace / "claim-spine.txt", "Thesis: The project has credible midterm progress because the noisy-ASR pipeline now runs on real VIVOS/MUSAN data.\nAudience: SLP instructor and project reviewers.\nArc: problem -> benchmark -> implementation -> baseline results -> next MTL proof.\n")
    write(workspace / "design-system.txt", "Slide size 1280x720. Warm paper background, deep navy cover/closing, Calibri/Aptos family, blue/teal/amber/red accents. Charts are editable shapes with direct labels.\n")
    write(workspace / "contact-sheet-plan.txt", "10 slides with varied macro layouts: cover metric rail, problem bullets, workflow diagram, process map, bar proof, SNR bar chart, metric table, error examples, status split, closing metric rail.\n")
    write(workspace / "source-notes.txt", "Sources: generated CSV outputs from local run: dataset_stats.csv, whisper_clean/noisy.csv, whisper_clean/noisy_forced.csv, metrics_whisper_clean/noisy_by_snr.csv, and forced metric CSVs. No external brand assets used.\n")
    print(workspace)


if __name__ == "__main__":
    main()
