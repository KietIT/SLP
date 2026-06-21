import fs from "node:fs/promises";
import path from "node:path";
import { spawnSync } from "node:child_process";

import {
  createSlideContext,
  ensureArtifactToolWorkspace,
  importArtifactTool,
  saveBlobToFile,
} from "file:///C:/Users/phath/.codex/plugins/cache/openai-primary-runtime/presentations/26.601.10930/skills/presentations/scripts/artifact_tool_utils.mjs";

if (!process.env.HOME && process.env.USERPROFILE) {
  process.env.HOME = process.env.USERPROFILE;
}

const ROOT_DIR = process.cwd();
const WORK_DIR = path.resolve("outputs", "manual-slp-midterm", "presentations", "slp-midterm-final");
const OUTPUT_DIR = path.join(WORK_DIR, "output");
const PREVIEW_DIR = path.join(OUTPUT_DIR, "previews");
const WORKSPACE_DIR = path.join(WORK_DIR, "artifact-workspace");
const LOGO_PATH = path.resolve(
  "outputs",
  "manual-slp-midterm",
  "extract",
  "source_unzip",
  "ppt",
  "media",
  "image4.png",
);
const FINAL_PPTX = path.join(ROOT_DIR, "SLP - Midterm SU2026 - completed.pptx");
const OUTPUT_PPTX = path.join(OUTPUT_DIR, "SLP - Midterm SU2026 - completed.pptx");
const CONTACT_SHEET = path.join(OUTPUT_DIR, "contact-sheet.png");
const MANIFEST_PATH = path.join(OUTPUT_DIR, "manifest.json");
const SLIDE_SIZE = { width: 1280, height: 720 };

const colors = {
  ink: "#0F172A",
  muted: "#475569",
  light: "#F8FAFC",
  panel: "#FFFFFF",
  border: "#CBD5E1",
  softBorder: "#E2E8F0",
  red: "#C00000",
  orange: "#F97316",
  amber: "#F59E0B",
  blue: "#0B63CE",
  green: "#45A942",
  cyan: "#0891B2",
  purple: "#7C3AED",
  slate: "#334155",
};

function line(ctx, fill = colors.border, width = 1) {
  return ctx.line(fill, width);
}

function addText(ctx, slide, options) {
  return ctx.addText(slide, {
    color: colors.ink,
    typeface: "Aptos",
    insets: { left: 8, right: 8, top: 6, bottom: 6 },
    ...options,
  });
}

function addRect(ctx, slide, options) {
  return ctx.addShape(slide, {
    geometry: "rect",
    fill: colors.panel,
    line: line(ctx, colors.softBorder, 1),
    ...options,
  });
}

function addTitle(ctx, slide, title, subtitle) {
  addText(ctx, slide, {
    x: 58,
    y: 110,
    width: 900,
    height: 60,
    text: title,
    fontSize: 36,
    bold: true,
    color: colors.red,
    typeface: "Aptos Display",
  });
  if (subtitle) {
    addText(ctx, slide, {
      x: 60,
      y: 166,
      width: 880,
      height: 36,
      text: subtitle,
      fontSize: 17,
      color: colors.muted,
    });
  }
}

async function addLogo(ctx, slide) {
  await ctx.addImage(slide, {
    path: LOGO_PATH,
    x: 54,
    y: 38,
    width: 220,
    height: 62,
    fit: "contain",
    alt: "FPT University logo",
  });
}

async function baseSlide(presentation, ctx, title, subtitle) {
  const slide = presentation.slides.add();
  addRect(ctx, slide, {
    x: 0,
    y: 0,
    width: 1280,
    height: 720,
    fill: "#FFFFFF",
    line: ctx.line("#FFFFFF", 0),
  });
  addRect(ctx, slide, {
    x: 0,
    y: 704,
    width: 1280,
    height: 16,
    fill: colors.light,
    line: ctx.line(colors.light, 0),
  });
  addRect(ctx, slide, {
    x: 0,
    y: 0,
    width: 1280,
    height: 9,
    fill: colors.red,
    line: ctx.line(colors.red, 0),
  });
  addRect(ctx, slide, {
    x: 0,
    y: 9,
    width: 430,
    height: 5,
    fill: colors.blue,
    line: ctx.line(colors.blue, 0),
  });
  addRect(ctx, slide, {
    x: 430,
    y: 9,
    width: 430,
    height: 5,
    fill: colors.orange,
    line: ctx.line(colors.orange, 0),
  });
  addRect(ctx, slide, {
    x: 860,
    y: 9,
    width: 420,
    height: 5,
    fill: colors.green,
    line: ctx.line(colors.green, 0),
  });
  await addLogo(ctx, slide);
  addTitle(ctx, slide, title, subtitle);
  return slide;
}

function addFooter(ctx, slide, index) {
  addText(ctx, slide, {
    x: 58,
    y: 675,
    width: 800,
    height: 24,
    text: "SLP301 Midterm SU2026 | Tone-aware LoRA Adaptation of PhoWhisper",
    fontSize: 12,
    color: colors.muted,
  });
  addText(ctx, slide, {
    x: 1165,
    y: 675,
    width: 60,
    height: 24,
    text: String(index).padStart(2, "0"),
    fontSize: 12,
    color: colors.muted,
    align: "right",
  });
}

function pill(ctx, slide, x, y, width, text, fill, color = colors.ink) {
  addRect(ctx, slide, {
    x,
    y,
    width,
    height: 32,
    fill,
    line: line(ctx, fill, 1),
  });
  addText(ctx, slide, {
    x: x + 10,
    y: y + 5,
    width: width - 20,
    height: 22,
    text,
    fontSize: 13,
    bold: true,
    color,
    align: "center",
  });
}

function card(ctx, slide, x, y, width, height, title, body, accent = colors.blue) {
  addRect(ctx, slide, {
    x,
    y,
    width,
    height,
    fill: colors.panel,
    line: line(ctx, colors.border, 1),
  });
  addRect(ctx, slide, {
    x,
    y,
    width: 8,
    height,
    fill: accent,
    line: ctx.line(accent, 0),
  });
  addText(ctx, slide, {
    x: x + 22,
    y: y + 16,
    width: width - 34,
    height: 30,
    text: title,
    fontSize: 19,
    bold: true,
    color: accent,
  });
  addText(ctx, slide, {
    x: x + 22,
    y: y + 52,
    width: width - 34,
    height: height - 62,
    text: body,
    fontSize: 15,
    color: colors.ink,
  });
}

function statCard(ctx, slide, x, y, width, label, value, note, accent) {
  addRect(ctx, slide, {
    x,
    y,
    width,
    height: 118,
    fill: "#FFFFFF",
    line: line(ctx, colors.border, 1),
  });
  addText(ctx, slide, {
    x: x + 18,
    y: y + 14,
    width: width - 36,
    height: 24,
    text: label,
    fontSize: 13,
    bold: true,
    color: colors.muted,
  });
  addText(ctx, slide, {
    x: x + 18,
    y: y + 38,
    width: width - 36,
    height: 46,
    text: value,
    fontSize: 34,
    bold: true,
    color: accent,
  });
  addText(ctx, slide, {
    x: x + 18,
    y: y + 86,
    width: width - 36,
    height: 24,
    text: note,
    fontSize: 12,
    color: colors.muted,
  });
}

function simpleBox(ctx, slide, x, y, width, height, label, body, accent = colors.blue, fill = "#FFFFFF") {
  addRect(ctx, slide, {
    x,
    y,
    width,
    height,
    fill,
    line: line(ctx, accent, 1.2),
  });
  addText(ctx, slide, {
    x: x + 10,
    y: y + 8,
    width: width - 20,
    height: 22,
    text: label,
    fontSize: 14,
    bold: true,
    color: accent,
    align: "center",
  });
  addText(ctx, slide, {
    x: x + 10,
    y: y + 34,
    width: width - 20,
    height: height - 42,
    text: body,
    fontSize: 12,
    color: colors.ink,
    align: "center",
  });
}

function connector(ctx, slide, x1, y1, x2, y2, color = colors.amber) {
  if (Math.abs(x2 - x1) >= Math.abs(y2 - y1)) {
    const x = Math.min(x1, x2);
    const y = y1 - 1;
    addRect(ctx, slide, {
      x,
      y,
      width: Math.max(2, Math.abs(x2 - x1)),
      height: 3,
      fill: color,
      line: ctx.line(color, 0),
    });
  } else {
    const x = x1 - 1;
    const y = Math.min(y1, y2);
    addRect(ctx, slide, {
      x,
      y,
      width: 3,
      height: Math.max(2, Math.abs(y2 - y1)),
      fill: color,
      line: ctx.line(color, 0),
    });
  }
}

function addTable(ctx, slide, x, y, colWidths, rowHeight, rows, options = {}) {
  const headerFill = options.headerFill ?? colors.ink;
  const headerColor = options.headerColor ?? "#FFFFFF";
  const fontSize = options.fontSize ?? 13;
  let currentY = y;
  for (const [rowIndex, row] of rows.entries()) {
    let currentX = x;
    const isHeader = rowIndex === 0;
    for (let colIndex = 0; colIndex < colWidths.length; colIndex += 1) {
      const fill = isHeader ? headerFill : rowIndex % 2 === 0 ? "#F8FAFC" : "#FFFFFF";
      addRect(ctx, slide, {
        x: currentX,
        y: currentY,
        width: colWidths[colIndex],
        height: rowHeight,
        fill,
        line: line(ctx, colors.border, 1),
      });
      addText(ctx, slide, {
        x: currentX + 4,
        y: currentY + 4,
        width: colWidths[colIndex] - 8,
        height: rowHeight - 8,
        text: String(row[colIndex] ?? ""),
        fontSize,
        bold: isHeader || options.boldFirstColumn && colIndex === 0,
        color: isHeader ? headerColor : colors.ink,
        align: colIndex === 0 ? "left" : "center",
      });
      currentX += colWidths[colIndex];
    }
    currentY += rowHeight;
  }
}

async function slide01(presentation, ctx) {
  const slide = await baseSlide(presentation, ctx, "", "");
  addText(ctx, slide, {
    x: 86,
    y: 170,
    width: 1040,
    height: 155,
    text: "TONE-AWARE LORA ADAPTATION OF PHOWHISPER",
    fontSize: 43,
    bold: true,
    color: colors.red,
    typeface: "Aptos Display",
  });
  addText(ctx, slide, {
    x: 88,
    y: 310,
    width: 1060,
    height: 74,
    text: "For noise-robust Vietnamese ASR under controlled acoustic noise",
    fontSize: 27,
    color: colors.ink,
    typeface: "Aptos Display",
  });
  addRect(ctx, slide, {
    x: 88,
    y: 400,
    width: 880,
    height: 2,
    fill: colors.amber,
    line: ctx.line(colors.amber, 0),
  });
  addText(ctx, slide, {
    x: 88,
    y: 428,
    width: 780,
    height: 58,
    text: "Nguyen Xuan Trung - SE193716\nTrinh Vy Kiet - SE192636\nNguyen Thanh Phat - SE192617\nPham Hoang Phuc - SE192874",
    fontSize: 17,
    color: colors.muted,
  });
  pill(ctx, slide, 88, 535, 160, "Pipeline done", "#DBEAFE", colors.blue);
  pill(ctx, slide, 266, 535, 160, "Baseline done", "#DCFCE7", colors.green);
  pill(ctx, slide, 444, 535, 206, "LoRA planned next", "#FEF3C7", "#A16207");
  addFooter(ctx, slide, 1);
  return slide;
}

async function slide02(presentation, ctx) {
  const slide = await baseSlide(presentation, ctx, "TABLE OF CONTENTS", "What the project has done and what remains");
  const items = [
    ["01", "Motivation and Problem", "Why Vietnamese ASR needs tone-aware robustness."],
    ["02", "Research Gap and Question", "What the project tests beyond ordinary noisy ASR."],
    ["03", "Completed Pipeline", "Dataset, noise mixing, inference, and metrics are reproducible."],
    ["04", "Planned Method", "PhoWhisper + LoRA + decoder-side tone head."],
    ["05", "Midterm Results and Next Steps", "Baseline evidence, LoRA experiments, and future work."],
  ];
  items.forEach(([num, title, body], index) => {
    const y = 225 + index * 74;
    addRect(ctx, slide, {
      x: 96,
      y,
      width: 82,
      height: 52,
      fill: index % 2 === 0 ? colors.blue : colors.orange,
      line: ctx.line(index % 2 === 0 ? colors.blue : colors.orange, 0),
    });
    addText(ctx, slide, {
      x: 106,
      y: y + 6,
      width: 62,
      height: 40,
      text: num,
      fontSize: 24,
      bold: true,
      color: "#FFFFFF",
      align: "center",
      valign: "middle",
    });
    addText(ctx, slide, {
      x: 205,
      y: y - 2,
      width: 800,
      height: 30,
      text: title,
      fontSize: 22,
      bold: true,
      color: colors.ink,
    });
    addText(ctx, slide, {
      x: 205,
      y: y + 28,
      width: 840,
      height: 26,
      text: body,
      fontSize: 15,
      color: colors.muted,
    });
  });
  addFooter(ctx, slide, 2);
  return slide;
}

async function slide03(presentation, ctx) {
  const slide = await baseSlide(presentation, ctx, "MOTIVATION AND PROBLEM STATEMENT", "Noise hurts Vietnamese ASR beyond word-level recognition");
  card(
    ctx,
    slide,
    74,
    230,
    340,
    210,
    "Vietnamese is tone-sensitive",
    "A wrong tone mark can change a syllable into a different word or meaning. ASR quality should therefore be checked at word, character, tone, and diacritic level.",
    colors.blue,
  );
  card(
    ctx,
    slide,
    470,
    230,
    340,
    210,
    "Real speech is noisy",
    "Cafe, traffic, fan, and rain-like noise reduce acoustic clarity. Tone cues are often weaker under low SNR conditions.",
    colors.orange,
  );
  card(
    ctx,
    slide,
    866,
    230,
    340,
    210,
    "WER/CER are not enough",
    "WER and CER show general ASR errors, but they do not isolate Vietnamese-specific tone and diacritic mistakes.",
    colors.green,
  );
  addRect(ctx, slide, {
    x: 106,
    y: 492,
    width: 1068,
    height: 92,
    fill: "#FFF7ED",
    line: line(ctx, "#FDBA74", 1),
  });
  addText(ctx, slide, {
    x: 130,
    y: 508,
    width: 1018,
    height: 28,
    text: "Problem",
    fontSize: 20,
    bold: true,
    color: colors.orange,
  });
  addText(ctx, slide, {
    x: 130,
    y: 538,
    width: 1018,
    height: 34,
    text: "How can PhoWhisper be adapted to noisy Vietnamese speech while preserving tonal correctness?",
    fontSize: 22,
    bold: true,
    color: colors.ink,
  });
  addFooter(ctx, slide, 3);
  return slide;
}

async function slide04(presentation, ctx) {
  const slide = await baseSlide(presentation, ctx, "RESEARCH GAP AND QUESTION", "The project is positioned around tone-aware adaptation, not speech enhancement");
  card(
    ctx,
    slide,
    78,
    220,
    320,
    174,
    "Model gap",
    "Whisper/PhoWhisper are strong ASR models, but the repo has not yet trained an explicit tone objective.",
    colors.blue,
  );
  card(
    ctx,
    slide,
    480,
    220,
    320,
    174,
    "Evaluation gap",
    "WER/CER are implemented together with simple TER and DER to expose Vietnamese-specific errors.",
    colors.orange,
  );
  card(
    ctx,
    slide,
    882,
    220,
    320,
    174,
    "Benchmark gap",
    "The current evidence is a controlled noisy-ASR baseline, not a full LoRA comparison yet.",
    colors.green,
  );
  addRect(ctx, slide, {
    x: 100,
    y: 445,
    width: 1080,
    height: 96,
    fill: "#F8FAFC",
    line: line(ctx, colors.border, 1),
  });
  addText(ctx, slide, {
    x: 126,
    y: 463,
    width: 1030,
    height: 26,
    text: "Research question",
    fontSize: 18,
    bold: true,
    color: colors.red,
  });
  addText(ctx, slide, {
    x: 126,
    y: 495,
    width: 1030,
    height: 36,
    text: "Does explicit tone supervision improve Vietnamese ASR robustness under controlled acoustic noise beyond ordinary noisy LoRA fine-tuning?",
    fontSize: 21,
    bold: true,
    color: colors.ink,
  });
  pill(ctx, slide, 174, 580, 220, "Current proof: baseline", "#DBEAFE", colors.blue);
  pill(ctx, slide, 430, 580, 225, "Next proof: LoRA runs", "#FEF3C7", "#A16207");
  pill(ctx, slide, 690, 580, 280, "Final proof: tone-aware MTL", "#DCFCE7", colors.green);
  addFooter(ctx, slide, 4);
  return slide;
}

async function slide05(presentation, ctx) {
  const slide = await baseSlide(presentation, ctx, "COMPLETED PIPELINE", "What the repo can already reproduce end to end");
  const y = 258;
  const boxes = [
    ["1. Data manifest", "VIVOS clean speech\ntranscript + wav path", colors.blue],
    ["2. Noise mixing", "MUSAN/demo noise\nSNR: 20/10/5/0 dB", colors.orange],
    ["3. Inference", "Whisper-base\nforced Vietnamese", colors.green],
    ["4. Evaluation", "WER, CER,\nTER-simple, DER-simple", colors.red],
    ["5. Report assets", "Markdown / LaTeX /\nslide builders", colors.purple],
  ];
  boxes.forEach(([label, body, accent], index) => {
    const x = 72 + index * 235;
    simpleBox(ctx, slide, x, y, 178, 124, label, body, accent);
    if (index < boxes.length - 1) {
      connector(ctx, slide, x + 178, y + 62, x + 230, y + 62, colors.amber);
    }
  });
  addRect(ctx, slide, {
    x: 86,
    y: 455,
    width: 1090,
    height: 112,
    fill: "#F0FDF4",
    line: line(ctx, "#86EFAC", 1),
  });
  addText(ctx, slide, {
    x: 110,
    y: 474,
    width: 1035,
    height: 24,
    text: "Important status",
    fontSize: 19,
    bold: true,
    color: colors.green,
  });
  addText(ctx, slide, {
    x: 110,
    y: 505,
    width: 1035,
    height: 48,
    text: "The pipeline and baseline results exist. Fine-tuned LoRA checkpoints and tone-aware MTL results are not available yet, so the deck must present those as planned experiments.",
    fontSize: 20,
    bold: true,
    color: colors.ink,
  });
  addFooter(ctx, slide, 5);
  return slide;
}

async function slide06(presentation, ctx) {
  const slide = await baseSlide(presentation, ctx, "PLANNED METHOD", "Tone-aware LoRA adaptation of PhoWhisper");
  addText(ctx, slide, {
    x: 900,
    y: 120,
    width: 270,
    height: 30,
    text: "PLANNED TRAINING, NOT MIDTERM RESULT",
    fontSize: 13,
    bold: true,
    color: colors.red,
    align: "center",
    fill: "#FEE2E2",
    line: line(ctx, "#FCA5A5", 1),
  });

  simpleBox(ctx, slide, 78, 238, 150, 84, "Input speech", "Vietnamese utterance\nclean or noisy", colors.blue, "#EFF6FF");
  simpleBox(ctx, slide, 270, 238, 150, 84, "Noise injection", "Cafe / traffic / fan\n20, 10, 5, 0 dB", colors.orange, "#FFF7ED");
  simpleBox(ctx, slide, 462, 238, 150, 84, "Log-Mel", "Spectrogram\nfeature input", colors.cyan, "#ECFEFF");
  simpleBox(ctx, slide, 668, 198, 238, 164, "PhoWhisper backbone", "Encoder + decoder\nLoRA on attention layers\nq_proj / v_proj", colors.green, "#F0FDF4");
  simpleBox(ctx, slide, 970, 218, 176, 64, "ASR output", "Predicted transcript", colors.red, "#FEF2F2");
  simpleBox(ctx, slide, 970, 314, 176, 64, "Tone head", "6 tone classes", colors.purple, "#F5F3FF");

  connector(ctx, slide, 228, 280, 270, 280);
  connector(ctx, slide, 420, 280, 462, 280);
  connector(ctx, slide, 612, 280, 668, 280);
  connector(ctx, slide, 906, 250, 970, 250);
  connector(ctx, slide, 906, 346, 970, 346, colors.purple);

  simpleBox(ctx, slide, 108, 466, 190, 72, "Reference transcript", "Ground-truth text", colors.slate, "#F8FAFC");
  simpleBox(ctx, slide, 356, 458, 190, 88, "Tone label extraction", "Rule-based labels\nfrom Vietnamese text", colors.orange, "#FFF7ED");
  simpleBox(ctx, slide, 640, 458, 172, 88, "ASR loss", "L_ASR", colors.red, "#FEF2F2");
  simpleBox(ctx, slide, 838, 458, 172, 88, "Tone loss", "L_tone", colors.purple, "#F5F3FF");
  simpleBox(ctx, slide, 572, 580, 390, 58, "Joint objective", "L_total = L_ASR + lambda * L_tone", colors.amber, "#FFFBEB");
  simpleBox(ctx, slide, 1014, 580, 150, 58, "Metrics", "WER/CER\nTER/DER by SNR", colors.green, "#F0FDF4");

  connector(ctx, slide, 298, 502, 356, 502, colors.slate);
  connector(ctx, slide, 546, 502, 640, 502, colors.orange);
  connector(ctx, slide, 1010, 502, 1010, 609, colors.purple);
  connector(ctx, slide, 1010, 609, 1014, 609, colors.purple);
  connector(ctx, slide, 812, 502, 838, 502, colors.purple);
  connector(ctx, slide, 726, 546, 726, 580, colors.red);
  connector(ctx, slide, 924, 546, 924, 580, colors.purple);

  addFooter(ctx, slide, 6);
  return slide;
}

async function slide07(presentation, ctx) {
  const slide = await baseSlide(presentation, ctx, "DATASET AND EVALUATION", "Data sources, experimental subset, and metrics");
  statCard(ctx, slide, 76, 222, 250, "Clean baseline", "30", "utterances evaluated", colors.blue);
  statCard(ctx, slide, 360, 222, 250, "Noisy baseline", "120", "30 utterances x 4 SNR", colors.orange);
  statCard(ctx, slide, 644, 222, 250, "Generated noisy set", "200", "50 utterances x 4 SNR", colors.green);
  statCard(ctx, slide, 928, 222, 250, "Average duration", "3.31s", "noisy evaluation subset", colors.purple);

  card(
    ctx,
    slide,
    76,
    390,
    335,
    148,
    "Data sources",
    "VIVOS: Vietnamese clean ASR speech\nMUSAN/demo noise: controlled acoustic mixing\nFLEURS: optional external evaluation",
    colors.blue,
  );
  card(
    ctx,
    slide,
    472,
    390,
    335,
    148,
    "Model sources",
    "Midterm baseline: Whisper-base with Vietnamese decoding\nNext model: PhoWhisper-base with LoRA adapters",
    colors.green,
  );
  card(
    ctx,
    slide,
    868,
    390,
    335,
    148,
    "Metrics",
    "WER: word errors\nCER: character errors\nTER-simple: tone error proxy\nDER-simple: diacritic error proxy",
    colors.orange,
  );
  addText(ctx, slide, {
    x: 95,
    y: 580,
    width: 1030,
    height: 34,
    text: "Lower is better for all metrics. TER/DER are currently simple repo-level proxies and should be strengthened with alignment-aware scoring later.",
    fontSize: 17,
    color: colors.muted,
  });
  addFooter(ctx, slide, 7);
  return slide;
}

async function slide08(presentation, ctx) {
  const slide = await baseSlide(presentation, ctx, "MIDTERM RESULTS", "Baseline: Whisper-base + forced Vietnamese decoding");
  addTable(
    ctx,
    slide,
    74,
    220,
    [138, 58, 88, 88, 98, 98],
    42,
    [
      ["Condition", "N", "WER", "CER", "TER", "DER"],
      ["Clean", "30", "43.06%", "19.61%", "23.13%", "12.22%"],
      ["Noisy all", "120", "47.95%", "23.55%", "25.98%", "10.82%"],
    ],
    { fontSize: 15, boldFirstColumn: true, headerFill: colors.red },
  );
  addTable(
    ctx,
    slide,
    74,
    390,
    [86, 58, 88, 88, 98, 98],
    38,
    [
      ["SNR", "N", "WER", "CER", "TER", "DER"],
      ["20 dB", "30", "43.42%", "20.02%", "21.00%", "10.17%"],
      ["10 dB", "30", "49.82%", "24.02%", "25.62%", "9.62%"],
      ["5 dB", "30", "45.55%", "22.39%", "25.62%", "11.24%"],
      ["0 dB", "30", "53.02%", "27.78%", "31.67%", "12.41%"],
    ],
    { fontSize: 13, boldFirstColumn: true, headerFill: colors.ink },
  );

  addText(ctx, slide, {
    x: 708,
    y: 219,
    width: 430,
    height: 28,
    text: "WER by SNR",
    fontSize: 22,
    bold: true,
    color: colors.ink,
  });
  const bars = [
    ["20 dB", 43.42, colors.green],
    ["10 dB", 49.82, colors.orange],
    ["5 dB", 45.55, colors.blue],
    ["0 dB", 53.02, colors.red],
  ];
  bars.forEach(([label, value, color], index) => {
    const y = 274 + index * 62;
    addText(ctx, slide, {
      x: 710,
      y: y - 4,
      width: 70,
      height: 26,
      text: label,
      fontSize: 15,
      bold: true,
      color: colors.ink,
    });
    addRect(ctx, slide, {
      x: 792,
      y,
      width: 330,
      height: 20,
      fill: "#E2E8F0",
      line: ctx.line("#E2E8F0", 0),
    });
    addRect(ctx, slide, {
      x: 792,
      y,
      width: 330 * (value / 60),
      height: 20,
      fill: color,
      line: ctx.line(color, 0),
    });
    addText(ctx, slide, {
      x: 1130,
      y: y - 4,
      width: 78,
      height: 26,
      text: `${value.toFixed(2)}%`,
      fontSize: 14,
      bold: true,
      color,
      align: "right",
    });
  });
  addRect(ctx, slide, {
    x: 704,
    y: 548,
    width: 470,
    height: 70,
    fill: "#FEF2F2",
    line: line(ctx, "#FCA5A5", 1),
  });
  addText(ctx, slide, {
    x: 724,
    y: 562,
    width: 430,
    height: 44,
    text: "Noise increases WER/CER. The hardest setting is 0 dB, with WER = 53.02% and CER = 27.78%. Forced Vietnamese decoding gives only small gains.",
    fontSize: 15,
    bold: true,
    color: colors.ink,
  });
  addFooter(ctx, slide, 8);
  return slide;
}

async function slide09(presentation, ctx) {
  const slide = await baseSlide(presentation, ctx, "NEXT STEPS", "Turn the completed baseline pipeline into LoRA evidence");
  const steps = [
    ["E1", "PhoWhisper zero-shot", "Run PhoWhisper-base on the same clean/noisy subset as the Whisper baseline.", colors.blue],
    ["E2", "Clean LoRA", "Fine-tune PhoWhisper with LoRA on clean VIVOS transcripts and compare to zero-shot.", colors.green],
    ["E3", "Noisy LoRA", "Train with MUSAN augmentation at 20, 10, 5, and 0 dB to improve noise robustness.", colors.orange],
    ["E4", "Tone-aware MTL", "Add decoder-side tone classification head and train with L_ASR + lambda * L_tone.", colors.purple],
    ["E5", "Final evaluation", "Report WER/CER/TER/DER by SNR, then run ablation: E3 versus E4.", colors.red],
  ];
  steps.forEach(([code, title, body, accent], index) => {
    const y = 205 + index * 78;
    addRect(ctx, slide, {
      x: 92,
      y,
      width: 72,
      height: 54,
      fill: accent,
      line: ctx.line(accent, 0),
    });
    addText(ctx, slide, {
      x: 102,
      y: y + 8,
      width: 52,
      height: 38,
      text: code,
      fontSize: 21,
      bold: true,
      color: "#FFFFFF",
      align: "center",
      valign: "middle",
    });
    addRect(ctx, slide, {
      x: 190,
      y,
      width: 912,
      height: 54,
      fill: index % 2 === 0 ? "#F8FAFC" : "#FFFFFF",
      line: line(ctx, colors.border, 1),
    });
    addText(ctx, slide, {
      x: 208,
      y: y + 5,
      width: 250,
      height: 24,
      text: title,
      fontSize: 18,
      bold: true,
      color: accent,
    });
    addText(ctx, slide, {
      x: 470,
      y: y + 7,
      width: 610,
      height: 38,
      text: body,
      fontSize: 15,
      color: colors.ink,
    });
    if (index < steps.length - 1) {
      connector(ctx, slide, 128, y + 54, 128, y + 78, colors.amber);
    }
  });
  addRect(ctx, slide, {
    x: 150,
    y: 612,
    width: 945,
    height: 36,
    fill: "#FFFBEB",
    line: line(ctx, "#FCD34D", 1),
  });
  addText(ctx, slide, {
    x: 168,
    y: 619,
    width: 910,
    height: 22,
    text: "Decision point: if E4 improves TER/DER without hurting WER/CER, the tone-aware objective is justified.",
    fontSize: 15,
    bold: true,
    color: colors.ink,
    align: "center",
  });
  addFooter(ctx, slide, 9);
  return slide;
}

async function slide10(presentation, ctx) {
  const slide = await baseSlide(presentation, ctx, "CONCLUSION AND REFERENCES", "Clear boundary between completed work and planned research");
  card(
    ctx,
    slide,
    76,
    208,
    520,
    162,
    "What can be claimed now",
    "A reproducible noisy Vietnamese ASR baseline pipeline is complete. It generates controlled SNR data, runs inference, computes metrics, and reports baseline degradation under noise.",
    colors.green,
  );
  card(
    ctx,
    slide,
    684,
    208,
    520,
    162,
    "What cannot be claimed yet",
    "No LoRA fine-tuned checkpoint or tone-aware MTL result has been produced yet. Claims about improvement must wait for E1-E5 experiments.",
    colors.red,
  );
  addText(ctx, slide, {
    x: 76,
    y: 415,
    width: 1060,
    height: 28,
    text: "References",
    fontSize: 22,
    bold: true,
    color: colors.red,
  });
  addText(ctx, slide, {
    x: 82,
    y: 456,
    width: 1080,
    height: 140,
    text:
      "[1] Radford et al., Robust Speech Recognition via Large-Scale Weak Supervision, ICML 2023.\n" +
      "[2] Le et al., PhoWhisper: Automatic Speech Recognition for Vietnamese, 2024.\n" +
      "[3] Hu et al., LoRA: Low-Rank Adaptation of Large Language Models, 2021.\n" +
      "[4] Snyder et al., MUSAN: A Music, Speech, and Noise Corpus, 2015.\n" +
      "[5] Conneau et al., FLEURS: Few-shot Learning Evaluation of Universal Representations of Speech, 2022.\n" +
      "[6] AILAB, VNUHCM-University of Science, VIVOS: Vietnamese Speech Corpus for ASR.",
    fontSize: 13,
    color: colors.ink,
  });
  addText(ctx, slide, {
    x: 846,
    y: 606,
    width: 310,
    height: 42,
    text: "THANK YOU",
    fontSize: 34,
    bold: true,
    color: colors.red,
    align: "right",
    typeface: "Aptos Display",
  });
  addFooter(ctx, slide, 10);
  return slide;
}

async function buildDeck() {
  await ensureArtifactToolWorkspace(WORKSPACE_DIR);
  const artifact = await importArtifactTool(WORKSPACE_DIR);
  const { Presentation, PresentationFile } = artifact;
  const presentation = Presentation.create({ slideSize: SLIDE_SIZE });
  const ctx = createSlideContext(artifact, {
    slideSize: SLIDE_SIZE,
    outputDir: OUTPUT_DIR,
    assetDir: path.join(WORKSPACE_DIR, "assets"),
    workspaceDir: WORKSPACE_DIR,
    titleFont: "Aptos Display",
    bodyFont: "Aptos",
  });

  const slideFns = [
    slide01,
    slide02,
    slide03,
    slide04,
    slide05,
    slide06,
    slide07,
    slide08,
    slide09,
    slide10,
  ];

  const slides = [];
  for (const fn of slideFns) {
    slides.push(await fn(presentation, ctx));
  }

  await fs.mkdir(PREVIEW_DIR, { recursive: true });
  const previewPaths = [];
  for (const [index, slide] of slides.entries()) {
    const previewPath = path.join(PREVIEW_DIR, `slide-${String(index + 1).padStart(2, "0")}.png`);
    const preview = await presentation.export({ slide, format: "png", scale: 1 });
    await saveBlobToFile(preview, previewPath);
    previewPaths.push(previewPath);
  }

  await fs.mkdir(OUTPUT_DIR, { recursive: true });
  const pptx = await PresentationFile.exportPptx(presentation);
  await pptx.save(OUTPUT_PPTX);
  await fs.copyFile(OUTPUT_PPTX, FINAL_PPTX);

  const contactScript =
    "C:\\Users\\phath\\.codex\\plugins\\cache\\openai-primary-runtime\\presentations\\26.601.10930\\skills\\presentations\\scripts\\make_contact_sheet.py";
  const contactResult = spawnSync("python", [contactScript, "--output", CONTACT_SHEET, ...previewPaths], {
    encoding: "utf8",
  });
  const contactSheet = contactResult.status === 0 ? CONTACT_SHEET : undefined;

  const manifest = {
    outputPptx: OUTPUT_PPTX,
    finalPptx: FINAL_PPTX,
    slideCount: slides.length,
    previewPaths,
    contactSheet,
    contactError: contactResult.status === 0 ? undefined : contactResult.stderr || contactResult.stdout,
  };
  await fs.writeFile(MANIFEST_PATH, `${JSON.stringify(manifest, null, 2)}\n`, "utf8");
  console.log(JSON.stringify(manifest, null, 2));
}

buildDeck().catch((error) => {
  console.error(error.stack || error.message || String(error));
  process.exit(1);
});
