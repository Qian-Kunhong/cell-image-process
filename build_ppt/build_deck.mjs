import fs from "node:fs/promises";
import path from "node:path";
import { pathToFileURL } from "node:url";
import { Presentation, PresentationFile } from "file:///C:/Users/dodos/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/@oai/artifact-tool/dist/artifact_tool.mjs";

const workspaceDir = path.resolve(".");
const buildDir = path.join(workspaceDir, "build_ppt");
const assetDir = path.join(buildDir, "assets");
const previewDir = path.join(buildDir, "previews");
const outputDir = path.join(workspaceDir, "output", "nuclear_morphology_presentation");
const FINAL_PPTX = path.join(outputDir, "Nuclear_morphology_study_plan.pptx");
const SKILL_DIR = "C:\\Users\\dodos\\.codex\\plugins\\cache\\openai-primary-runtime\\presentations\\26.1013.11914\\skills\\presentations";
const RUNTIME_PYTHON = "C:\\Users\\dodos\\.cache\\codex-runtimes\\codex-primary-runtime\\dependencies\\python\\python.exe";
process.env.RUNTIME_NODE_MODULES = "C:\\Users\\dodos\\.cache\\codex-runtimes\\codex-primary-runtime\\dependencies\\node\\node_modules";

await fs.mkdir(previewDir, { recursive: true });
await fs.mkdir(outputDir, { recursive: true });

const deck = Presentation.create({ slideSize: { width: 1280, height: 720 } });
const C = {
  navy: "#071521",
  navy2: "#0D2233",
  blue: "#3EA6FF",
  cyan: "#61D6D0",
  pale: "#DCEAF5",
  white: "#F8FBFD",
  muted: "#9CB3C4",
  line: "#2B4659",
  warm: "#F4B860",
  red: "#FF7A7A",
};
const FONT = "Aptos";

function rect(slide, x, y, w, h, fill, radius = "rounded-md", line = "none") {
  return slide.shapes.add({
    geometry: radius === "none" ? "rect" : "roundRect",
    position: { left: x, top: y, width: w, height: h },
    fill,
    line: line === "none" ? { style: "solid", fill: "none", width: 0 } : { style: "solid", fill: line, width: 1 },
    ...(radius === "none" ? {} : { borderRadius: radius }),
  });
}

function textBox(slide, text, x, y, w, h, opts = {}) {
  const shape = slide.shapes.add({
    geometry: "textbox",
    position: { left: x, top: y, width: w, height: h },
    fill: "none",
    line: { style: "solid", fill: "none", width: 0 },
  });
  shape.text = text;
  shape.text.style = {
    fontFamily: FONT,
    fontSize: opts.size ?? 22,
    bold: opts.bold ?? false,
    color: opts.color ?? C.white,
    alignment: opts.align ?? "left",
    verticalAlignment: opts.valign ?? "top",
  };
  return shape;
}

function title(slide, heading, sub = "") {
  textBox(slide, heading, 70, 48, 1120, 58, { size: 32, bold: true });
  rect(slide, 70, 116, 68, 5, C.cyan, "none");
  if (sub) textBox(slide, sub, 158, 105, 1032, 32, { size: 16, color: C.muted });
}

function footer(slide, n) {
  textBox(slide, String(n).padStart(2, "0"), 1170, 670, 44, 24, { size: 12, color: C.muted, align: "right" });
}

async function imageBytes(filename) {
  const b = await fs.readFile(path.join(assetDir, filename));
  return b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength);
}

async function addImage(slide, filename, x, y, w, h, fit = "cover") {
  return slide.images.add({
    blob: await imageBytes(filename),
    contentType: "image/png",
    alt: filename,
    fit,
    position: { left: x, top: y, width: w, height: h },
  });
}

function note(slide, sourceText) {
  slide.speakerNotes.text = sourceText;
}

// 1. Cover
{
  const s = deck.slides.add();
  s.background.fill = C.navy;
  await addImage(s, "D21_HA10_DAPI.png", 740, 0, 540, 720, "cover");
  rect(s, 706, 0, 80, 720, C.navy2, "none");
  textBox(s, "DAPI-BASED STUDY PLAN", 74, 82, 420, 28, { size: 15, bold: true, color: C.cyan });
  textBox(s, "Nuclear morphology as a shared experimental readout", 74, 150, 590, 220, { size: 46, bold: true });
  textBox(s, "Study framework and first experiment", 76, 392, 500, 44, { size: 23, color: C.pale });
  textBox(s, "Focus: stained nuclear morphology", 76, 548, 420, 30, { size: 16, color: C.muted });
  textBox(s, "10× DAPI images", 76, 580, 420, 30, { size: 16, color: C.muted });
  note(s, "Image source: F:\\Naruchit data\\1_Data of published manuscripts\\3_Hepatocyte differentiation\\03_Raw data\\Immunostaining\\IN Cell\\D0_3_17_21_E-cad N-cad\\10x\\D21\\HA10-DAPI.tif. Display contrast normalized consistently across the 16-image series.");
}

// 2. Study question
{
  const s = deck.slides.add(); s.background.fill = C.navy; title(s, "Study question"); footer(s, 2);
  textBox(s, "Can DAPI nuclear morphology provide a common representation of experimental state?", 110, 164, 1060, 84, { size: 34, bold: true, align: "center", valign: "middle" });
  const items = [
    ["Shared input", "DAPI is available across stages and staining panels"],
    ["Common space", "Each nucleus becomes a comparable feature vector"],
    ["Study output", "Samples are compared by state shift and heterogeneity"],
  ];
  items.forEach((it, i) => {
    const x = 105 + i * 370;
    rect(s, x, 326, 330, 184, C.navy2, "rounded-lg", C.line);
    textBox(s, it[0], x + 24, 354, 282, 34, { size: 20, bold: true, color: i === 1 ? C.cyan : C.blue, align: "center" });
    textBox(s, it[1], x + 30, 408, 270, 72, { size: 18, color: C.pale, align: "center", valign: "middle" });
  });
  textBox(s, "The first task is to establish that the signal exists.", 150, 570, 980, 44, { size: 24, bold: true, color: C.warm, align: "center" });
}

// 3. Roadmap
{
  const s = deck.slides.add(); s.background.fill = C.navy; title(s, "Research roadmap", "Each experiment answers one question"); footer(s, 3);
  const steps = [
    ["1", "Detect", "Do treatments produce measurable nuclear morphology differences?"],
    ["2", "Map", "Do samples form a reproducible morphology state space?"],
    ["3", "Relate", "Does morphology agree with paired measurements?"],
    ["4", "Predict", "Can early morphology estimate later outcomes and stability?"],
  ];
  steps.forEach((st, i) => {
    const y = 158 + i * 124;
    rect(s, 104, y, 62, 62, i === 0 ? C.cyan : C.line, "rounded-full");
    textBox(s, st[0], 104, y + 9, 62, 42, { size: 23, bold: true, color: i === 0 ? C.navy : C.white, align: "center", valign: "middle" });
    textBox(s, st[1], 202, y - 2, 200, 34, { size: 25, bold: true, color: i === 0 ? C.cyan : C.white });
    textBox(s, st[2], 420, y - 1, 720, 58, { size: 20, color: C.pale, valign: "middle" });
    if (i < 3) rect(s, 133, y + 68, 4, 50, C.line, "none");
  });
  textBox(s, "Current scope", 930, 595, 150, 28, { size: 15, bold: true, color: C.cyan, align: "right" });
  textBox(s, "Experiment 1", 1090, 591, 130, 34, { size: 20, bold: true, align: "right" });
}

// 4. First experiment data
{
  const s = deck.slides.add(); s.background.fill = C.navy; title(s, "Experiment 1 data", "D0 only, one imaging series, four treatment levels"); footer(s, 4);
  const files = ["D0_HA00_DAPI.png", "D0_HA01_DAPI.png", "D0_HA05_DAPI.png", "D0_HA10_DAPI.png"];
  const labels = ["HA00", "HA01", "HA05", "HA10"];
  for (let i = 0; i < 4; i++) {
    const x = 72 + i * 300;
    await addImage(s, files[i], x, 174, 260, 260, "cover");
    textBox(s, labels[i], x, 446, 260, 34, { size: 22, bold: true, color: i === 3 ? C.cyan : C.white, align: "center" });
  }
  textBox(s, "Primary comparison", 90, 535, 210, 32, { size: 18, bold: true, color: C.blue });
  textBox(s, "Nuclear morphology distributions across four treatment levels", 305, 531, 850, 40, { size: 23, bold: true });
  textBox(s, "Why D0 first: one time point removes stage as a competing source of variation.", 90, 590, 1070, 36, { size: 19, color: C.pale });
  note(s, "Image sources: F:\\Naruchit data\\1_Data of published manuscripts\\3_Hepatocyte differentiation\\03_Raw data\\Immunostaining\\IN Cell\\D0_3_17_21_E-cad N-cad\\10x\\D0\\HA00_DAPI.tif; HA01_DAPI.tif; HA05_DAPI.tif; HA10_DAPI.tif. Display contrast normalized consistently across the 16-image series.");
}

// 5. Pipeline
{
  const s = deck.slides.add(); s.background.fill = C.navy; title(s, "Experiment 1 workflow"); footer(s, 5);
  const stages = [
    ["DAPI image", "Same magnification and acquisition series"],
    ["Segmentation", "Nuclear masks with border and size QC"],
    ["Features", "Shape, intensity and texture per nucleus"],
    ["Spatial blocks", "Distribution summaries within each region"],
    ["Comparison", "State shift, classification and heterogeneity"],
  ];
  stages.forEach((st, i) => {
    const x = 58 + i * 244;
    rect(s, x, 220, 198, 214, i === 0 ? "#12344A" : C.navy2, "rounded-lg", i === 0 ? C.blue : C.line);
    textBox(s, String(i + 1).padStart(2, "0"), x + 20, 244, 48, 28, { size: 15, bold: true, color: C.cyan });
    textBox(s, st[0], x + 20, 290, 158, 36, { size: 23, bold: true });
    textBox(s, st[1], x + 20, 346, 158, 64, { size: 17, color: C.pale });
    if (i < 4) textBox(s, ">", x + 200, 300, 42, 50, { size: 30, bold: true, color: C.line, align: "center" });
  });
  textBox(s, "Unit of inference", 84, 524, 190, 30, { size: 18, bold: true, color: C.warm });
  textBox(s, "Spatial block or independent image, not individual nuclei", 285, 518, 880, 42, { size: 24, bold: true });
  textBox(s, "Single nuclei describe the distribution. They do not create independent experimental replicates.", 84, 578, 1070, 36, { size: 18, color: C.muted });
}

// 6. Features
{
  const s = deck.slides.add(); s.background.fill = C.navy; title(s, "Nuclear morphology features", "Two analyses separate morphology from acquisition intensity"); footer(s, 6);
  await addImage(s, "D3_HA05_DAPI.png", 70, 168, 365, 365, "cover");
  textBox(s, "Shape only", 500, 168, 250, 34, { size: 25, bold: true, color: C.cyan });
  textBox(s, "Area\nCircularity\nAspect ratio\nEccentricity\nSolidity\nBoundary irregularity", 500, 218, 290, 260, { size: 21, color: C.pale });
  textBox(s, "Full feature set", 850, 168, 300, 34, { size: 25, bold: true, color: C.blue });
  textBox(s, "DAPI intensity distribution\nNuclear texture\nSpatial density\nFeature dispersion\nMorphology subtype proportions", 850, 218, 330, 244, { size: 21, color: C.pale });
  rect(s, 500, 520, 650, 2, C.line, "none");
  textBox(s, "A result that survives the shape only analysis is less sensitive to staining brightness.", 500, 546, 650, 66, { size: 21, bold: true, color: C.warm });
  note(s, "Image source: F:\\Naruchit data\\1_Data of published manuscripts\\3_Hepatocyte differentiation\\03_Raw data\\Immunostaining\\IN Cell\\D0_3_17_21_E-cad N-cad\\10x\\D3\\HA05_DAPI.tif. Display contrast normalized consistently across the 16-image series.");
}

// 7. Tests
{
  const s = deck.slides.add(); s.background.fill = C.navy; title(s, "Analyses and decision criteria"); footer(s, 7);
  const rows = [
    ["Feature distributions", "Do group centers or distribution widths differ?", "Effect size and uncertainty"],
    ["PCA and UMAP", "Do groups occupy different regions?", "Consistent structure across blocks"],
    ["Classification", "Can morphology identify treatment level?", "Blocked validation above chance"],
    ["Heterogeneity", "Does treatment change within-sample dispersion?", "Stable CV, IQR or state-space spread"],
  ];
  textBox(s, "Analysis", 86, 166, 240, 30, { size: 17, bold: true, color: C.muted });
  textBox(s, "Question", 356, 166, 470, 30, { size: 17, bold: true, color: C.muted });
  textBox(s, "Evidence to proceed", 856, 166, 330, 30, { size: 17, bold: true, color: C.muted });
  rows.forEach((r, i) => {
    const y = 210 + i * 90;
    rect(s, 78, y - 12, 1110, 74, i % 2 === 0 ? C.navy2 : "#0A1B29", "rounded-md");
    textBox(s, r[0], 96, y, 235, 42, { size: 20, bold: true, color: i === 2 ? C.cyan : C.white, valign: "middle" });
    textBox(s, r[1], 356, y, 462, 46, { size: 18, color: C.pale, valign: "middle" });
    textBox(s, r[2], 856, y, 306, 46, { size: 18, color: C.pale, valign: "middle" });
  });
  textBox(s, "A significant cell-level p-value alone does not support the study claim.", 120, 602, 1040, 42, { size: 22, bold: true, color: C.red, align: "center" });
}

// 8. Outcomes
{
  const s = deck.slides.add(); s.background.fill = C.navy; title(s, "Expected outcomes and next step"); footer(s, 8);
  const outcomes = [
    ["Clear state shift", "Treatment groups separate across blocks and models.", "Extend the same pipeline to D3, D17 and D21."],
    ["Heterogeneity only", "Group means overlap, but distribution widths differ.", "Make heterogeneity the primary response."],
    ["No robust signal", "Separation disappears after blocked validation.", "Review image QC, feature stability and replication."],
  ];
  outcomes.forEach((o, i) => {
    const y = 162 + i * 142;
    rect(s, 78, y, 8, 104, [C.cyan, C.warm, C.red][i], "none");
    textBox(s, o[0], 112, y + 2, 280, 38, { size: 25, bold: true, color: [C.cyan, C.warm, C.red][i] });
    textBox(s, o[1], 400, y + 2, 720, 38, { size: 20, color: C.white });
    textBox(s, o[2], 400, y + 50, 720, 40, { size: 18, color: C.muted });
  });
  rect(s, 78, 606, 1110, 2, C.line, "none");
  textBox(s, "Immediate deliverable", 84, 626, 220, 30, { size: 16, bold: true, color: C.blue });
  textBox(s, "D0 nuclear morphology report for HA00, HA01, HA05 and HA10", 310, 619, 840, 38, { size: 23, bold: true });
}

for (let i = 0; i < deck.slides.items.length; i++) {
  const png = await deck.export({ slide: deck.slides.items[i], format: "png", scale: 1 });
  const bytes = new Uint8Array(await png.arrayBuffer());
  await fs.writeFile(path.join(previewDir, `slide-${String(i + 1).padStart(2, "0")}.png`), bytes);
}
const montage = await deck.export({ format: "webp", montage: true });
await fs.writeFile(path.join(buildDir, "deck-montage.webp"), new Uint8Array(await montage.arrayBuffer()));

const requirements = {
  explicitTotalSlideCount: 8,
  requiredNativeTableOwnerSlides: [],
  requiredNativeChartOwnerSlides: [],
};
const fontPolicy = { basis: "design", families: [FONT] };
const expectedSlideSizeEmu = "12192000,6858000";
const { finalizePresentation } = await import(pathToFileURL(path.join(SKILL_DIR, "container_tools", "artifact_tool_utils.mjs")).href);
const stagingDir = path.join(buildDir, ".codex-finalizer");
await fs.mkdir(stagingDir, { recursive: true });
const candidatePath = path.join(stagingDir, "candidate.pptx");
await (await PresentationFile.exportPptx(deck)).save(candidatePath);
await finalizePresentation({
  ...requirements,
  workspaceDir,
  candidatePath,
  finalPath: FINAL_PPTX,
  pythonExecutable: RUNTIME_PYTHON,
  integrityValidatorPath: path.join(SKILL_DIR, "container_tools", "inspect_presentation_package_integrity.py"),
  layoutValidatorPath: path.join(SKILL_DIR, "container_tools", "inspect_presentation_layout_geometry.py"),
  layoutArgs: ["--expected-slide-size-emu", expectedSlideSizeEmu, "--validate-heading-fit"],
  requiredNativeTableOwnerSlides: [],
  fontPolicy,
  verifyArtifactToolImport: true,
  receiptPath: path.join(stagingDir, "Nuclear_morphology_study_plan.validation.json"),
});

console.log(FINAL_PPTX);
