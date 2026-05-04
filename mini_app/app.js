const models = [
  { name: "YOLOv5m", folds: [0.769, 0.769, 0.762, 0.759], mean: 0.765, family: "YOLOv5" },
  { name: "YOLOv5s", folds: [0.769, 0.776, 0.75, 0.747], mean: 0.761, family: "YOLOv5" },
  { name: "YOLOv5x", folds: [0.764, 0.762, 0.767, 0.751], mean: 0.761, family: "YOLOv5" },
  { name: "YOLO11l", folds: [0.762, 0.763, 0.752, 0.757], mean: 0.759, family: "YOLO11" },
  { name: "YOLOv5n", folds: [0.77, 0.767, 0.747, 0.747], mean: 0.758, family: "YOLOv5" },
  { name: "YOLOv5l", folds: [0.775, 0.767, 0.742, 0.744], mean: 0.757, family: "YOLOv5" },
  { name: "YOLOv8n", folds: [0.766, 0.762, 0.753, 0.742], mean: 0.756, family: "YOLOv8" },
  { name: "YOLOv8n-BS", folds: [0.775, 0.764, 0.751, 0.735], mean: 0.756, family: "Custom" },
  { name: "YOLO11n", folds: [0.755, 0.767, 0.749, 0.751], mean: 0.756, family: "YOLO11" },
  { name: "YOLO11m", folds: [0.753, 0.763, 0.748, 0.744], mean: 0.752, family: "YOLO11" },
  { name: "YOLOv8s", folds: [0.76, 0.755, 0.743, 0.746], mean: 0.751, family: "YOLOv8" },
  { name: "YOLOv8m", folds: [0.764, 0.752, 0.741, 0.741], mean: 0.749, family: "YOLOv8" },
  { name: "YOLO11s", folds: [0.763, 0.755, 0.745, 0.732], mean: 0.749, family: "YOLO11" },
  { name: "YOLO11x", folds: [0.754, 0.76, 0.754, 0.729], mean: 0.749, family: "YOLO11" },
  { name: "YOLOv8x", folds: [0.737, 0.756, 0.74, 0.749], mean: 0.746, family: "YOLOv8" },
  { name: "YOLOv8-Unet", folds: [0.773, 0.748, 0.744, 0.718], mean: 0.746, family: "Custom" },
  { name: "YOLOv10m", folds: [0.744, 0.747, 0.732, 0.747], mean: 0.743, family: "YOLOv10" },
  { name: "YOLOv8l", folds: [0.748, 0.757, 0.727, 0.736], mean: 0.742, family: "YOLOv8" },
  { name: "YOLOv10l", folds: [0.748, 0.747, 0.731, 0.736], mean: 0.741, family: "YOLOv10" },
  { name: "YOLOv10x", folds: [0.742, 0.733, 0.731, 0.732], mean: 0.735, family: "YOLOv10" },
  { name: "YOLOv10n", folds: [0.746, 0.744, 0.715, 0.725], mean: 0.733, family: "YOLOv10" },
  { name: "YOLOv10s", folds: [0.739, 0.745, 0.719, 0.723], mean: 0.732, family: "YOLOv10" },
  { name: "YOLOv8n-DD", folds: [0.313, 0.326, 0.375, 0.451], mean: 0.366, family: "Custom" },
];

const splits = [
  { label: "80 / 20", key: "80/20", ap50: 0.78, confidenceShift: 0.04, strictness: "optimistic" },
  { label: "75 / 25", key: "75/25", ap50: 0.766, confidenceShift: 0, strictness: "baseline" },
  { label: "70 / 30", key: "70/30", ap50: 0.736, confidenceShift: -0.06, strictness: "stricter" },
];

const inspectionCases = [
  {
    fold: 1,
    eyebrow: "fold 1 case: scratch and patch",
    title: "Long scratch with patch defect",
    background: ["#cdd5dc", "#eef3f5", "#b9c4cc"],
    texture: "horizontal",
    defects: [
      { kind: "scratch", x: 126, y: 116, w: 270, h: 34, color: "#873f3f", shape: "slash", rotation: -0.08, baseConfidence: 0.86 },
      { kind: "patch", x: 616, y: 88, w: 156, h: 104, color: "#735b42", shape: "blob", baseConfidence: 0.81 },
      { kind: "inclusion", x: 438, y: 314, w: 124, h: 82, color: "#354c60", shape: "blob", baseConfidence: 0.76 },
    ],
  },
  {
    fold: 2,
    eyebrow: "fold 2 case: pitted surface",
    title: "Clustered pitting with small inclusions",
    background: ["#d3d6d0", "#f3f4ef", "#b8beb7"],
    texture: "speckled",
    defects: [
      { kind: "pitted_surface", x: 176, y: 112, w: 118, h: 92, color: "#5f5347", shape: "pits", baseConfidence: 0.8 },
      { kind: "pitted_surface", x: 492, y: 282, w: 150, h: 96, color: "#62584c", shape: "pits", baseConfidence: 0.78 },
      { kind: "inclusion", x: 708, y: 126, w: 84, h: 66, color: "#405669", shape: "blob", baseConfidence: 0.74 },
    ],
  },
  {
    fold: 3,
    eyebrow: "fold 3 case: rolled-in scale",
    title: "Rolled-in scale and subtle crazing",
    background: ["#c9d0d4", "#f2f4f6", "#aeb8bf"],
    texture: "diagonal",
    defects: [
      { kind: "rolled-in_scale", x: 150, y: 270, w: 292, h: 62, color: "#4f5960", shape: "band", rotation: 0.05, baseConfidence: 0.77 },
      { kind: "crazing", x: 584, y: 118, w: 188, h: 112, color: "#6f4b4b", shape: "cracks", baseConfidence: 0.72 },
      { kind: "scratch", x: 520, y: 372, w: 220, h: 30, color: "#854747", shape: "slash", rotation: 0.1, baseConfidence: 0.73 },
    ],
  },
  {
    fold: 4,
    eyebrow: "fold 4 case: mixed small defects",
    title: "Small defects with lower fold AP50",
    background: ["#cfd3d4", "#f5f6f4", "#b6bab9"],
    texture: "vertical",
    defects: [
      { kind: "inclusion", x: 226, y: 142, w: 82, h: 70, color: "#3d5161", shape: "blob", baseConfidence: 0.72 },
      { kind: "patch", x: 658, y: 252, w: 132, h: 84, color: "#705a43", shape: "blob", baseConfidence: 0.7 },
      { kind: "crazing", x: 390, y: 300, w: 168, h: 104, color: "#704948", shape: "cracks", baseConfidence: 0.67 },
    ],
  },
];

const state = {
  model: "YOLOv5m",
  fold: 1,
  split: "75/25",
};

const modelSelect = document.querySelector("#modelSelect");
const foldSegments = document.querySelector("#foldSegments");
const splitSelect = document.querySelector("#splitSelect");
const foldAp50 = document.querySelector("#foldAp50");
const meanAp50 = document.querySelector("#meanAp50");
const modelRank = document.querySelector("#modelRank");
const modelBars = document.querySelector("#modelBars");
const splitBars = document.querySelector("#splitBars");
const splitAp50 = document.querySelector("#splitAp50");
const statusPill = document.querySelector("#statusPill");
const inspectionEyebrow = document.querySelector("#inspectionEyebrow");
const inspectionTitle = document.querySelector("#inspectionTitle");
const inspectionAction = document.querySelector("#inspectionAction");
const inspectionReason = document.querySelector("#inspectionReason");
const stabilityScore = document.querySelector("#stabilityScore");
const stabilityReason = document.querySelector("#stabilityReason");
const riskLevel = document.querySelector("#riskLevel");
const fnImpact = document.querySelector("#fnImpact");
const fpImpact = document.querySelector("#fpImpact");
const deploymentStance = document.querySelector("#deploymentStance");
const engineeringNote = document.querySelector("#engineeringNote");
const canvas = document.querySelector("#defectCanvas");
const ctx = canvas.getContext("2d");

function format(value) {
  return Number(value).toFixed(3);
}

function selectedModel() {
  return models.find((model) => model.name === state.model);
}

function modelRankNumber(name) {
  return models.findIndex((model) => model.name === name) + 1;
}

function selectedSplit() {
  return splits.find((item) => item.key === state.split);
}

function selectedCase() {
  return inspectionCases.find((item) => item.fold === state.fold);
}

function modelTier(model) {
  if (model.name === "YOLOv8n-DD") return "weak";
  if (model.mean < 0.74) return "low";
  if (model.mean < 0.75) return "moderate";
  return "strong";
}

function foldSpread(model) {
  return Math.max(...model.folds) - Math.min(...model.folds);
}

function benchmarkDecision(model, split) {
  const tier = modelTier(model);
  const spread = foldSpread(model);

  if (tier === "weak") {
    return {
      action: "Do not deploy",
      reason: "Mean AP50 is far below the competitive models and missed-defect risk is high.",
      risk: "High escape risk",
      fn: "Likely missed defects",
      fp: "Unstable review load",
      stance: "Research baseline only",
      note: "This model is useful as a contrast case, not as a manufacturing inspection candidate.",
    };
  }

  if (tier === "low" || split.strictness === "stricter") {
    return {
      action: "Engineering review",
      reason: "Model performance is usable for comparison, but split sensitivity or lower AP50 warrants review.",
      risk: "Moderate process risk",
      fn: "Small defects need review",
      fp: "May increase rework",
      stance: "Pilot with controls",
      note: "Use fold-level results and human inspection before any production decision.",
    };
  }

  if (spread <= 0.018 && model.mean >= 0.756) {
    return {
      action: "Benchmark candidate",
      reason: "Mean AP50 and fold stability are competitive under the cross-validation protocol.",
      risk: "Controlled benchmark risk",
      fn: "Monitor rare defects",
      fp: "Manageable review load",
      stance: "Decision support",
      note: "This model is a strong candidate for prototype evaluation, with human quality review retained.",
    };
  }

  return {
    action: "Compare before pilot",
    reason: "Mean AP50 is competitive, but fold-to-fold spread should be considered in model selection.",
    risk: "Partition-sensitive risk",
    fn: "Validate on new lots",
    fp: "Track review burden",
    stance: "Decision support",
    note: "Prefer statistically supported comparisons over a single AP50 ranking.",
  };
}

function describeStability(model) {
  const spread = foldSpread(model);
  if (spread <= 0.018) return "Stable across folds";
  if (spread <= 0.035) return "Moderate fold variation";
  return "High fold variation";
}

function setupControls() {
  models.forEach((model) => {
    const option = document.createElement("option");
    option.value = model.name;
    option.textContent = `${model.name} (${model.family})`;
    modelSelect.appendChild(option);
  });
  modelSelect.value = state.model;

  [1, 2, 3, 4].forEach((fold) => {
    const button = document.createElement("button");
    button.type = "button";
    button.textContent = String(fold);
    button.setAttribute("role", "radio");
    button.setAttribute("aria-checked", String(fold === state.fold));
    button.addEventListener("click", () => {
      state.fold = fold;
      update();
    });
    foldSegments.appendChild(button);
  });

  modelSelect.addEventListener("change", (event) => {
    state.model = event.target.value;
    update();
  });

  splitSelect.addEventListener("change", (event) => {
    state.split = event.target.value;
    update();
  });
}

function drawSteelSurface(model, split, inspectionCase) {
  const { width, height } = canvas;
  ctx.clearRect(0, 0, width, height);

  const gradient = ctx.createLinearGradient(0, 0, width, height);
  gradient.addColorStop(0, inspectionCase.background[0]);
  gradient.addColorStop(0.5, inspectionCase.background[1]);
  gradient.addColorStop(1, inspectionCase.background[2]);
  ctx.fillStyle = gradient;
  ctx.fillRect(0, 0, width, height);

  drawSurfaceTexture(inspectionCase.texture, width, height);

  inspectionCase.defects.forEach((defect) => drawDefect(defect));

  const boxes = detectionsFor(model, split, inspectionCase);
  boxes.forEach((box) => drawDetectionBox(box));

  ctx.fillStyle = "rgba(23, 33, 43, 0.84)";
  ctx.fillRect(18, height - 62, 474, 42);
  ctx.fillStyle = "#ffffff";
  ctx.font = "700 18px system-ui, sans-serif";
  ctx.fillText(
    `${model.name} | fold ${state.fold} | ${split.label} split | AP50 ${format(model.folds[state.fold - 1])}`,
    34,
    height - 35,
  );
}

function drawSurfaceTexture(texture, width, height) {
  if (texture === "speckled") {
    for (let i = 0; i < 260; i += 1) {
      const x = (i * 47) % width;
      const y = (i * 83) % height;
      ctx.fillStyle = `rgba(45, 55, 62, ${0.04 + (i % 4) * 0.018})`;
      ctx.beginPath();
      ctx.arc(x, y, 1 + (i % 4), 0, Math.PI * 2);
      ctx.fill();
    }
    return;
  }

  const diagonal = texture === "diagonal";
  const vertical = texture === "vertical";
  for (let i = -height; i < width + height; i += 30) {
    ctx.strokeStyle = `rgba(60, 72, 82, ${i % 60 === 0 ? 0.15 : 0.08})`;
    ctx.lineWidth = 1;
    ctx.beginPath();
    if (vertical) {
      ctx.moveTo(i + height, 0);
      ctx.bezierCurveTo(i + height - 10, height * 0.3, i + height + 12, height * 0.72, i + height - 4, height);
    } else if (diagonal) {
      ctx.moveTo(i, 0);
      ctx.lineTo(i + height, height);
    } else {
      const y = i + height;
      ctx.moveTo(0, y + Math.sin(y) * 4);
      ctx.bezierCurveTo(width * 0.25, y - 8, width * 0.65, y + 12, width, y - 3);
    }
    ctx.stroke();
  }
}

function drawDefect(defect) {
  ctx.save();
  ctx.globalAlpha = 0.7;
  ctx.fillStyle = defect.color;
  ctx.strokeStyle = defect.color;
  ctx.lineWidth = 5;

  if (defect.shape === "slash") {
    ctx.translate(defect.x, defect.y);
    ctx.rotate(defect.rotation || 0);
    ctx.fillRect(0, 0, defect.w, defect.h);
  } else if (defect.shape === "band") {
    ctx.translate(defect.x, defect.y);
    ctx.rotate(defect.rotation || 0);
    roundedRect(0, 0, defect.w, defect.h, 18);
    ctx.fill();
    for (let x = 18; x < defect.w; x += 42) {
      ctx.strokeStyle = "rgba(255, 255, 255, 0.22)";
      ctx.beginPath();
      ctx.moveTo(x, 8);
      ctx.lineTo(x + 22, defect.h - 10);
      ctx.stroke();
    }
  } else if (defect.shape === "pits") {
    roundedRect(defect.x, defect.y, defect.w, defect.h, 20);
    ctx.fill();
    ctx.fillStyle = "rgba(20, 26, 30, 0.4)";
    for (let i = 0; i < 16; i += 1) {
      const x = defect.x + 12 + ((i * 23) % Math.max(24, defect.w - 24));
      const y = defect.y + 12 + ((i * 31) % Math.max(24, defect.h - 24));
      ctx.beginPath();
      ctx.arc(x, y, 3 + (i % 4), 0, Math.PI * 2);
      ctx.fill();
    }
  } else if (defect.shape === "cracks") {
    ctx.strokeStyle = defect.color;
    for (let i = 0; i < 7; i += 1) {
      const startX = defect.x + 12 + i * 22;
      const startY = defect.y + 14 + (i % 3) * 20;
      ctx.beginPath();
      ctx.moveTo(startX, startY);
      ctx.lineTo(startX + 22, startY + 14);
      ctx.lineTo(startX + 44, startY + 2);
      ctx.lineTo(startX + 66, startY + 24);
      ctx.stroke();
    }
  } else {
    roundedRect(defect.x, defect.y, defect.w, defect.h, 18);
    ctx.fill();
  }

  ctx.restore();
}

function detectionsFor(model, split, inspectionCase) {
  const tier = modelTier(model);
  const foldScore = model.folds[state.fold - 1];
  const foldShift = (foldScore - model.mean) * 1.2;
  const modelShift = model.mean >= 0.76 ? 0.04 : model.mean >= 0.75 ? 0.02 : model.mean >= 0.74 ? -0.02 : -0.05;
  const shouldMissLast = tier === "weak" || (tier === "low" && split.strictness === "stricter");

  const detections = inspectionCase.defects
    .filter((_, index) => !(shouldMissLast && index === inspectionCase.defects.length - 1))
    .map((defect, index) => {
      const weakOffset = tier === "weak" ? 22 : tier === "low" ? 10 : 0;
      const splitOffset = split.strictness === "stricter" ? 7 : split.strictness === "optimistic" ? -4 : 0;
      const confidence = clamp(defect.baseConfidence + split.confidenceShift + modelShift + foldShift - index * 0.015, 0.32, 0.94);
      return {
        label: `${defect.kind} ${format(confidence)}`,
        x: defect.x - 14 + weakOffset + splitOffset,
        y: defect.y - 18 + weakOffset * 0.5,
        w: defect.w + 30 - weakOffset * 0.65,
        h: defect.h + 38 - weakOffset * 0.45,
        quality: tier === "weak" ? "weak" : confidence < 0.68 ? "uncertain" : "good",
      };
    });

  if (tier === "weak" || (split.strictness === "stricter" && tier !== "strong")) {
    detections.push({
      label: tier === "weak" ? "false alarm 0.39" : "possible pit 0.58",
      x: 752,
      y: 344,
      w: 94,
      h: 72,
      quality: "false",
    });
  }

  return detections;
}

function clamp(value, min, max) {
  return Math.max(min, Math.min(max, value));
}

function roundedRect(x, y, width, height, radius) {
  ctx.beginPath();
  ctx.moveTo(x + radius, y);
  ctx.lineTo(x + width - radius, y);
  ctx.quadraticCurveTo(x + width, y, x + width, y + radius);
  ctx.lineTo(x + width, y + height - radius);
  ctx.quadraticCurveTo(x + width, y + height, x + width - radius, y + height);
  ctx.lineTo(x + radius, y + height);
  ctx.quadraticCurveTo(x, y + height, x, y + height - radius);
  ctx.lineTo(x, y + radius);
  ctx.quadraticCurveTo(x, y, x + radius, y);
}

function drawDetectionBox(box) {
  const color = box.quality === "weak" || box.quality === "false" ? "#b54646" : box.quality === "uncertain" ? "#a46a25" : "#2f8c62";
  ctx.strokeStyle = color;
  ctx.lineWidth = 4;
  ctx.strokeRect(box.x, box.y, box.w, box.h);
  ctx.fillStyle = color;
  ctx.fillRect(box.x, box.y - 30, Math.max(132, box.label.length * 10), 30);
  ctx.fillStyle = "#ffffff";
  ctx.font = "700 16px system-ui, sans-serif";
  ctx.fillText(box.label, box.x + 10, box.y - 10);
}

function renderModelBars() {
  modelBars.innerHTML = "";
  models.forEach((model) => {
    const row = document.createElement("div");
    row.className = "bar-row";

    const name = document.createElement("span");
    name.className = "bar-name";
    name.textContent = model.name;

    const track = document.createElement("div");
    track.className = "bar-track";

    const fill = document.createElement("div");
    fill.className = "bar-fill";
    if (model.name === state.model) fill.classList.add("selected");
    if (model.name === "YOLOv8n-DD") fill.classList.add("weak");
    fill.style.width = `${Math.max(2, (model.mean / 0.78) * 100)}%`;
    track.appendChild(fill);

    const value = document.createElement("span");
    value.className = "bar-value";
    value.textContent = format(model.mean);

    row.append(name, track, value);
    modelBars.appendChild(row);
  });
}

function renderSplitBars() {
  splitBars.innerHTML = "";
  splits.forEach((split) => {
    const item = document.createElement("div");
    item.className = "split-item";
    if (split.key === state.split) item.classList.add("selected");

    const column = document.createElement("div");
    column.className = "split-column";
    column.style.height = `${Math.max(18, (split.ap50 / 0.79) * 130)}px`;
    column.title = `${split.label}: ${format(split.ap50)}`;

    const label = document.createElement("div");
    label.className = "split-label";
    label.textContent = `${split.label} | ${format(split.ap50)}`;

    item.append(column, label);
    splitBars.appendChild(item);
  });
}

function update() {
  const model = selectedModel();
  const split = selectedSplit();
  const inspectionCase = selectedCase();
  const tier = modelTier(model);
  const weak = tier === "weak";
  const decision = benchmarkDecision(model, split);
  const spread = foldSpread(model);

  foldAp50.textContent = format(model.folds[state.fold - 1]);
  meanAp50.textContent = format(model.mean);
  modelRank.textContent = `#${modelRankNumber(model.name)}`;
  splitAp50.textContent = `AP50 ${format(split.ap50)}`;
  inspectionEyebrow.textContent = `${inspectionCase.eyebrow} | ${split.strictness} split`;
  statusPill.textContent = weak ? "underperforming" : tier === "low" ? "lower tier" : "competitive";
  statusPill.classList.toggle("weak", weak);
  inspectionTitle.textContent = weak
    ? `${inspectionCase.title}: lower-confidence boxes and missed defect`
    : `${inspectionCase.title}: predicted defect boxes`;
  inspectionAction.textContent = decision.action;
  inspectionReason.textContent = decision.reason;
  stabilityScore.textContent = `${format(spread)} spread`;
  stabilityReason.textContent = describeStability(model);
  riskLevel.textContent = decision.risk;
  fnImpact.textContent = decision.fn;
  fpImpact.textContent = decision.fp;
  deploymentStance.textContent = decision.stance;
  engineeringNote.textContent = decision.note;

  [...foldSegments.children].forEach((button, index) => {
    button.setAttribute("aria-checked", String(index + 1 === state.fold));
  });

  renderModelBars();
  renderSplitBars();
  drawSteelSurface(model, split, inspectionCase);
}

setupControls();
update();
