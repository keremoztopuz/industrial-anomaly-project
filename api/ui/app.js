"use strict";

const form = document.querySelector("#prediction-form");
const controls = document.querySelector("#controls");
const category = document.querySelector("#category");
const upload = document.querySelector("#upload");
const run = document.querySelector("#run");
const retry = document.querySelector("#retry");
const status = document.querySelector("#status");
const error = document.querySelector("#error");
const preview = document.querySelector("#preview-image");
const emptyPreview = document.querySelector("#preview-empty");
const selection = document.querySelector("#selection");
const datasetLabel = document.querySelector("#dataset-label");
const sampleHelp = document.querySelector("#sample-help");
const sampleButtons = [...document.querySelectorAll("[data-sample]")];
const result = document.querySelector("#result");
const verdict = document.querySelector("#verdict");

let samples = [];
let selectedFile = null;
let previewURL = null;
let ready = false;
let busy = false;

function sampleFor(kind) {
  return samples.find(sample => sample.category === category.value && sample.kind === kind);
}

function updateControls() {
  controls.disabled = !ready || busy;
  run.disabled = !selectedFile || !ready || busy;
  for (const button of sampleButtons) {
    button.disabled = !sampleFor(button.dataset.sample) || busy;
  }
  if (samples.length && ready) {
    sampleHelp.textContent = sampleFor("normal") || sampleFor("defective")
      ? "Original normal and defective images from the MVTec AD test set."
      : "No samples for this category. Upload a PNG or JPEG instead.";
  }
  form.setAttribute("aria-busy", String(busy));
}

function showError(message) {
  error.textContent = message;
  error.hidden = !message;
}

function selectFile(file, sample = null) {
  if (previewURL) URL.revokeObjectURL(previewURL);
  previewURL = null;
  selectedFile = file;
  result.hidden = true;
  showError("");
  preview.hidden = !file;
  emptyPreview.hidden = Boolean(file);
  emptyPreview.textContent = "Upload an image or choose a sample.";
  selection.textContent = file ? file.name : "No image selected";
  datasetLabel.hidden = !sample;
  datasetLabel.textContent = sample
    ? `Dataset label: ${sample.kind === "normal" ? "Normal (good)" : `Defective (${sample.label})`}. This is the reference label, not the model's prediction.`
    : "";
  if (file) {
    previewURL = URL.createObjectURL(file);
    preview.src = previewURL;
    preview.alt = `Selected test image: ${file.name}`;
  } else {
    preview.removeAttribute("src");
  }
  updateControls();
}

preview.addEventListener("error", () => {
  if (!selectedFile) return;
  preview.hidden = true;
  emptyPreview.hidden = false;
  emptyPreview.textContent = "Preview unavailable. The server will validate this file.";
});

async function request(path, options = {}) {
  let response;
  try {
    response = await fetch(new URL(path, document.baseURI), {
      ...options,
      signal: AbortSignal.timeout(120000),
    });
  } catch (cause) {
    throw new Error(cause.name === "TimeoutError"
      ? "The request timed out. The service may still be starting; try again."
      : "Could not reach the service. Check your connection and try again.");
  }
  if (!response.ok) {
    const messages = {
      400: "This file could not be read. Choose a PNG or JPEG image.",
      404: "This item is no longer available. Refresh the page and try again.",
      413: "This image exceeds the server's file size or pixel limit.",
      422: "Choose a category and a PNG or JPEG image.",
    };
    throw new Error(messages[response.status] || "The server could not complete the request. Try again.");
  }
  return response;
}

async function loadCategories() {
  ready = false;
  retry.hidden = true;
  status.textContent = "Loading categories…";
  showError("");
  updateControls();
  try {
    const data = await (await request("../categories")).json();
    if (!data.categories.length) throw new Error("No product categories are available.");
    category.replaceChildren(...data.categories.map(name => new Option(name, name)));
    ready = true;
  } catch (cause) {
    category.replaceChildren(new Option("Categories unavailable", ""));
    showError(`Could not load categories. ${cause.message}`);
    retry.hidden = false;
  } finally {
    status.textContent = "";
    updateControls();
  }
}

category.addEventListener("change", () => {
  upload.value = "";
  selectFile(null);
  status.textContent = "";
});

upload.addEventListener("change", () => {
  selectFile(upload.files[0] || null);
  status.textContent = "";
});

for (const button of sampleButtons) {
  button.addEventListener("click", async () => {
    const sample = sampleFor(button.dataset.sample);
    if (!sample || busy) return;
    upload.value = "";
    selectFile(null);
    busy = true;
    status.textContent = "Loading sample…";
    updateControls();
    try {
      const response = await request(sample.file);
      const blob = await response.blob();
      selectFile(new File([blob], sample.file.split("/").pop(), { type: "image/png" }), sample);
    } catch (cause) {
      showError(`Could not load this sample. You can upload an image instead. ${cause.message}`);
    } finally {
      busy = false;
      status.textContent = "";
      updateControls();
    }
  });
}

form.addEventListener("submit", async event => {
  event.preventDefault();
  if (!ready || busy || !selectedFile) return;
  busy = true;
  result.hidden = true;
  showError("");
  status.textContent = "Running prediction… Please wait.";
  updateControls();
  const body = new FormData();
  body.append("upload_file", selectedFile, selectedFile.name);
  try {
    const response = await request(`../predict/${encodeURIComponent(category.value)}`, { method: "POST", body });
    const data = await response.json();
    const decision = data.is_anomaly === null ? "unavailable" : data.is_anomaly ? "anomaly" : "normal";
    result.dataset.decision = decision;
    verdict.textContent = {
      anomaly: "Anomaly detected",
      normal: "No anomaly detected",
      unavailable: "Decision unavailable",
    }[decision];
    document.querySelector("#score").textContent = data.anomaly_score.toFixed(4);
    document.querySelector("#threshold").textContent = data.threshold === null ? "Not calibrated" : data.threshold.toFixed(4);
    document.querySelector("#decision-note").textContent = decision === "unavailable"
      ? "No threshold is configured for this category, so only the score is available."
      : "An image is flagged when its anomaly score is greater than the category threshold.";
    result.hidden = false;
    verdict.focus();
  } catch (cause) {
    showError(cause.message);
  } finally {
    busy = false;
    status.textContent = "";
    updateControls();
  }
});

retry.addEventListener("click", loadCategories);
loadCategories();

request("samples.json").then(response => response.json()).then(data => {
  samples = data.samples;
  sampleHelp.textContent = "Original normal and defective images from the MVTec AD test set.";
  updateControls();
}).catch(() => {
  sampleHelp.textContent = "Samples are unavailable. You can upload a PNG or JPEG instead.";
});

request("../model").then(response => response.json()).then(data => {
  document.querySelector("#model").textContent = `${data.name} · version ${data.version}`;
}).catch(() => {
  document.querySelector("#model").textContent = "Model information unavailable.";
});
