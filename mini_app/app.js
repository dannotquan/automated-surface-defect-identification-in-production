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
  { label: "80 / 20", key: "80/20", ap50: 0.78 },
  { label: "75 / 25", key: "75/25", ap50: 0.766 },
  { label: "70 / 30", key: "70/30", ap50: 0.736 },
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
const inspectionTitle = document.querySelector("#inspectionTitle");
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

function drawSteelSurface(model) {
  const { width, height } = canvas;
  ctx.clearRect(0, 0, width, height);

  const gradient = ctx.createLinearGradient(0, 0, width, height);
  gradient.addColorStop(0, "#cfd6dc");
  gradient.addColorStop(0.5, "#eef2f4");
  gradient.addColorStop(1, "#b9c2ca");
  ctx.fillStyle = gradient;
  ctx.fillRect(0, 0, width, height);

  for (let y = 24; y < height; y += 28) {
    ctx.strokeStyle = `rgba(60, 72, 82, ${y % 56 === 0 ? 0.15 : 0.08})`;
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(0, y + Math.sin(y) * 4);
    ctx.bezierCurveTo(width * 0.25, y - 8, width * 0.65, y + 12, width, y - 3);
    ctx.stroke();
  }

  const defects = [
    { label: "scratch", x: 126, y: 116, w: 260, h: 36, color: "#873f3f" },
    { label: "patch", x: 616, y: 88, w: 156, h: 104, color: "#735b42" },
    { label: "inclusion", x: 438, y: 314, w: 124, h: 82, color: "#354c60" },
  ];

  defects.forEach((defect, index) => {
    ctx.save();
    ctx.globalAlpha = 0.68;
    ctx.fillStyle = defect.color;
    if (index === 0) {
      ctx.translate(defect.x, defect.y);
      ctx.rotate(-0.08);
      ctx.fillRect(0, 0, defect.w, defect.h);
      ctx.restore();
    } else {
      roundedRect(defect.x, defect.y, defect.w, defect.h, 18);
      ctx.fill();
      ctx.restore();
    }
  });

  const weak = model.name === "YOLOv8n-DD";
  const boxes = weak
    ? [
        { label: "scratch 0.54", x: 112, y: 94, w: 220, h: 72 },
        { label: "patch 0.41", x: 584, y: 68, w: 126, h: 82 },
      ]
    : [
        { label: "scratch 0.87", x: 112, y: 92, w: 292, h: 78 },
        { label: "patch 0.82", x: 596, y: 70, w: 196, h: 142 },
        { label: "inclusion 0.76", x: 420, y: 292, w: 162, h: 126 },
      ];

  boxes.forEach((box) => drawDetectionBox(box, weak));

  ctx.fillStyle = "rgba(23, 33, 43, 0.84)";
  ctx.fillRect(18, height - 58, 360, 38);
  ctx.fillStyle = "#ffffff";
  ctx.font = "700 18px system-ui, sans-serif";
  ctx.fillText(`${model.name} | fold ${state.fold} | AP50 ${format(model.folds[state.fold - 1])}`, 34, height - 33);
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

function drawDetectionBox(box, weak) {
  const color = weak ? "#b54646" : "#2f8c62";
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
  const weak = model.name === "YOLOv8n-DD";
  const split = splits.find((item) => item.key === state.split);

  foldAp50.textContent = format(model.folds[state.fold - 1]);
  meanAp50.textContent = format(model.mean);
  modelRank.textContent = `#${modelRankNumber(model.name)}`;
  splitAp50.textContent = `AP50 ${format(split.ap50)}`;
  statusPill.textContent = weak ? "underperforming" : "competitive";
  statusPill.classList.toggle("weak", weak);
  inspectionTitle.textContent = weak
    ? "Lower-confidence defect boxes and missed inclusion"
    : "Steel surface with predicted defect boxes";

  [...foldSegments.children].forEach((button, index) => {
    button.setAttribute("aria-checked", String(index + 1 === state.fold));
  });

  renderModelBars();
  renderSplitBars();
  drawSteelSurface(model);
}

setupControls();
update();

