const fileInput = document.getElementById("fileInput");
const dropZone = document.getElementById("dropZone");
const analyzeButton = document.getElementById("analyzeButton");
const audioPlayer = document.getElementById("audioPlayer");
const toast = document.getElementById("toast");

let selectedFile = null;
let activePayload = null;
let activeObjectUrl = null;
let toastTimer = null;

const escapeText = value => String(value).replace(/[&<>"']/g, character => ({
  "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"
})[character]);

function announce(message, error = false) {
  toast.textContent = message;
  toast.classList.toggle("error", error);
  toast.classList.add("show");
  window.clearTimeout(toastTimer);
  toastTimer = window.setTimeout(() => toast.classList.remove("show"), 3500);
}


function clearResults() {
  activePayload = null;
  document.getElementById("exportReport").disabled = true;
  document.getElementById("scoreValue").textContent = "—";
  document.getElementById("scoreArc").setAttribute("stroke-dashoffset", "439.82");
  document.getElementById("scoreArc").setAttribute("stroke", "#6b4eff");
  const badge = document.getElementById("riskBadge");
  badge.className = "risk-badge awaiting";
  badge.textContent = "AWAITING ANALYSIS";
  document.getElementById("decisionTitle").textContent = "Ready for review";
  document.getElementById("decisionText").textContent = "Analyze the selected sample to view its signal indicators.";
  document.getElementById("durationChip").textContent = "—:—";
  document.getElementById("waveformStatus").textContent = "NO SIGNAL ANALYZED";
  document.getElementById("sampleRateLabel").textContent = "SAMPLE RATE —";
  document.getElementById("channelLabel").textContent = "CHANNELS —";
  document.getElementById("waveform").hidden = true;
  document.getElementById("waveform").innerHTML = "";
  document.getElementById("waveformEmpty").hidden = false;
  document.getElementById("featureList").innerHTML = [
    ["Signal energy", "Average squared amplitude", "fill-mint"],
    ["Zero-crossing rate", "Sign changes per sample", "fill-purple"],
    ["Spectral flatness", "Noise-like vs tonal spectrum", "fill-coral"],
    ["Spectral centroid", "Frequency balance proxy", "fill-blue"]
  ].map(item => '<div class="feature-row"><div><strong>' + item[0] + '</strong><span>' + item[1] + '</span></div><b>—</b><div class="feature-track"><i class="feature-fill ' + item[2] + ' w-0"></i></div></div>').join("");
}

function formatDuration(seconds) {
  const total = Math.max(0, Math.round(Number(seconds) || 0));
  return Math.floor(total / 60) + ":" + String(total % 60).padStart(2, "0");
}

function formatBytes(bytes) {
  return bytes > 1024 * 1024
    ? (bytes / 1024 / 1024).toFixed(2) + " MB"
    : Math.max(1, Math.round(bytes / 1024)) + " KB";
}

function setBusy(busy, label) {
  analyzeButton.disabled = busy || !selectedFile;
  analyzeButton.innerHTML = busy
    ? '<span class="button-play">◌</span> ' + escapeText(label || "Analyzing…")
    : '<span class="button-play">▶</span> Analyze audio';
  document.getElementById("demoReference").disabled = busy;
  document.getElementById("demoProcessed").disabled = busy;
}

function chooseFile(file) {
  if (!file) return;
  if (!file.name.toLowerCase().endsWith(".wav") && file.type !== "audio/wav" && file.type !== "audio/x-wav") {
    announce("Please choose a WAV file.", true);
    return;
  }
  if (file.size <= 0) {
    announce("The selected file is empty.", true);
    return;
  }
  if (file.size > 12 * 1024 * 1024) {
    announce("Files must be 12 MB or smaller.", true);
    return;
  }
  selectedFile = file;
  clearResults();
  document.getElementById("fileName").textContent = file.name;
  document.getElementById("fileMeta").textContent = formatBytes(file.size) + " · ready for local review";
  document.getElementById("fileState").textContent = "READY";
  document.getElementById("fileState").className = "file-state ready";
  dropZone.classList.add("has-file");
  analyzeButton.disabled = false;
  if (activeObjectUrl) URL.revokeObjectURL(activeObjectUrl);
  activeObjectUrl = URL.createObjectURL(file);
  audioPlayer.src = activeObjectUrl;
  audioPlayer.hidden = false;
  announce("Audio selected. Analyze it to view the signal review.");
}

async function analyzeSelected() {
  if (!selectedFile) {
    announce("Choose a WAV file first.", true);
    return;
  }
  setBusy(true, "Analyzing…");
  try {
    const response = await fetch("/api/analyze", {
      method: "POST",
      headers: { "Content-Type": "application/octet-stream", "Accept": "application/json" },
      body: selectedFile,
      cache: "no-store"
    });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.error || "Audio analysis failed.");
    payload.source = "uploaded";
    payload.file_name = selectedFile.name;
    renderPayload(payload);
    announce("Signal review complete for " + selectedFile.name + ".");
  } catch (error) {
    announce(error.message || "Could not analyze this audio.", true);
  } finally {
    setBusy(false);
  }
}

async function loadDemo(kind) {
  setBusy(true, "Loading demo…");
  try {
    const response = await fetch("/api/demo?kind=" + encodeURIComponent(kind), {
      headers: { "Accept": "application/json" },
      cache: "no-store"
    });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.error || "Could not load demo signal.");
    selectedFile = null;
    fileInput.value = "";
    analyzeButton.disabled = true;
    if (activeObjectUrl) URL.revokeObjectURL(activeObjectUrl);
    activeObjectUrl = null;
    document.getElementById("fileName").textContent = kind === "reference" ? "Reference test tone" : "Processed test tone";
    document.getElementById("fileMeta").textContent = "Generated locally · not human speech";
    document.getElementById("fileState").textContent = "DEMO";
    document.getElementById("fileState").className = "file-state";
    dropZone.classList.remove("has-file");
    if (activeObjectUrl) URL.revokeObjectURL(activeObjectUrl);
    activeObjectUrl = null;
    audioPlayer.src = "/api/demo.wav?kind=" + encodeURIComponent(kind);
    audioPlayer.hidden = false;
    payload.source = "demo";
    payload.file_name = kind === "reference" ? "reference-test-tone.wav" : "processed-test-tone.wav";
    renderPayload(payload);
    announce("Demo test tone loaded. It is not speech.");
  } catch (error) {
    announce(error.message || "Could not load demo signal.", true);
  } finally {
    setBusy(false);
  }
}

function renderWaveform(samples) {
  const empty = document.getElementById("waveformEmpty");
  const target = document.getElementById("waveform");
  if (!samples || samples.length < 2) {
    target.hidden = true;
    empty.hidden = false;
    return;
  }
  const width = 780, height = 148, mid = 74, amp = 49;
  const step = width / (samples.length - 1);
  const path = samples.map((sample, index) => (index ? "L" : "M") + (index * step).toFixed(2) + " " + (mid - Math.max(-1, Math.min(1, sample)) * amp).toFixed(2)).join(" ");
  const area = path + " L " + width + " " + mid + " L 0 " + mid + " Z";
  const bars = samples.map((sample, index) => {
    if (index % 4 !== 0) return "";
    const y = mid - Math.abs(sample) * amp;
    return '<line x1="' + (index * step).toFixed(2) + '" y1="' + y.toFixed(2) + '" x2="' + (index * step).toFixed(2) + '" y2="' + (mid + Math.abs(sample) * amp).toFixed(2) + '" stroke="#9e8af7" stroke-width="1.5" stroke-linecap="round" opacity=".62"/>';
  }).join("");
  target.innerHTML = '<svg viewBox="0 0 ' + width + ' ' + height + '" role="img" aria-label="Waveform preview"><defs><linearGradient id="waveFill" x1="0" y1="0" x2="0" y2="1"><stop offset="0%" stop-color="#8c74f5" stop-opacity=".18"/><stop offset="100%" stop-color="#8c74f5" stop-opacity=".02"/></linearGradient></defs><line x1="0" y1="' + mid + '" x2="' + width + '" y2="' + mid + '" stroke="#eee8f6" stroke-width="1"/><path d="' + area + '" fill="url(#waveFill)"/>' + bars + '<path d="' + path + '" fill="none" stroke="#7156e9" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"/></svg>';
  target.hidden = false;
  empty.hidden = true;
}

function renderFeatures(data) {
  const features = data.features || {};
  const values = [
    { key: "energy", title: "Signal energy", value: Number(features.energy || 0), format: value => value.toFixed(5), width: value => Math.min(100, value * 450), color: "fill-mint", note: "Average squared amplitude" },
    { key: "zcr", title: "Zero-crossing rate", value: Number(features.zcr || 0), format: value => value.toFixed(5), width: value => Math.min(100, value * 300), color: "fill-purple", note: "Sign changes per sample" },
    { key: "flatness", title: "Spectral flatness", value: Number(features.flatness || 0), format: value => value.toFixed(5), width: value => Math.min(100, value * 100), color: "fill-coral", note: "Noise-like vs tonal spectrum" },
    { key: "spectral_centroid_proxy", title: "Spectral centroid", value: Number(features.spectral_centroid_proxy || 0), format: value => Math.round(value).toLocaleString(), width: value => Math.min(100, value / 8000 * 100), color: "fill-blue", note: "Frequency balance proxy" }
  ];
  document.getElementById("featureList").innerHTML = values.map(item =>
    '<div class="feature-row"><div><strong>' + item.title + '</strong><span>' + item.note + '</span></div><b>' + item.format(item.value) + '</b><div class="feature-track"><i class="feature-fill ' + item.color + ' w-' + Math.round(item.width(item.value) / 5) * 5 + '"></i></div></div>'
  ).join("");
}

function renderPayload(data) {
  activePayload = data;
  document.getElementById("scoreValue").textContent = String(data.score_percent);
  document.getElementById("scoreArc").setAttribute("stroke-dashoffset", String(439.82 * (1 - Math.max(0, Math.min(100, data.score_percent)) / 100)));
  const risk = data.risk_level || "low";
  document.getElementById("scoreArc").setAttribute("stroke", risk === "high" ? "#e87983" : risk === "review" ? "#d8a04c" : "#6b4eff");
  const badge = document.getElementById("riskBadge");
  badge.className = "risk-badge " + risk;
  badge.textContent = risk === "high" ? "STRONG HEURISTIC FLAGS" : risk === "review" ? "REVIEW INDICATED" : "NO STRONG FLAGS";
  document.getElementById("decisionTitle").textContent = risk === "high" ? "Review this sample" : risk === "review" ? "Some signals stand out" : "No strong cues detected";
  document.getElementById("decisionText").textContent = risk === "high"
    ? "Several thresholds were triggered by this signal. This does not establish that the voice is cloned."
    : risk === "review"
      ? "One or more simple signal thresholds were triggered. Compare with trusted reference audio before drawing conclusions."
      : "This sample did not cross the current heuristic thresholds. That is not proof that the voice is genuine.";
  document.getElementById("durationChip").textContent = formatDuration(data.duration_seconds);
  document.getElementById("waveformStatus").textContent = "SIGNAL READY";
  document.getElementById("sampleRateLabel").textContent = (Number(data.sample_rate) / 1000).toFixed(1) + " KHZ";
  document.getElementById("channelLabel").textContent = data.channels === 1 ? "MONO" : "STEREO · DOWNMIXED";
  renderWaveform(data.waveform);
  renderFeatures(data);
  document.getElementById("exportReport").disabled = false;
}

function exportReport() {
  if (!activePayload) return announce("Analyze an audio sample before exporting.", true);
  const report = {
    product: "Soniq Voice Integrity Studio",
    generated_at: new Date().toISOString(),
    file_name: activePayload.file_name || "demo-tone.wav",
    interpretation: "Heuristic review only. The score is not a calibrated probability of cloned speech.",
    ...activePayload
  };
  const blob = new Blob([JSON.stringify(report, null, 2)], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = "soniq-voice-review.json";
  document.body.appendChild(anchor);
  anchor.click();
  anchor.remove();
  URL.revokeObjectURL(url);
  announce("Review report exported.");
}

document.getElementById("browseButton").addEventListener("click", event => {
  event.stopPropagation();
  fileInput.click();
});
dropZone.addEventListener("click", event => {
  if (!event.target.closest("button")) fileInput.click();
});
dropZone.addEventListener("keydown", event => {
  if (event.key === "Enter" || event.key === " ") {
    event.preventDefault();
    fileInput.click();
  }
});
fileInput.addEventListener("change", () => chooseFile(fileInput.files && fileInput.files[0]));
dropZone.addEventListener("dragover", event => {
  event.preventDefault();
  dropZone.classList.add("dragging");
});
dropZone.addEventListener("dragleave", event => {
  if (!dropZone.contains(event.relatedTarget)) dropZone.classList.remove("dragging");
});
dropZone.addEventListener("drop", event => {
  event.preventDefault();
  dropZone.classList.remove("dragging");
  chooseFile(event.dataTransfer && event.dataTransfer.files[0]);
});
analyzeButton.addEventListener("click", analyzeSelected);
document.getElementById("demoReference").addEventListener("click", () => loadDemo("reference"));
document.getElementById("demoProcessed").addEventListener("click", () => loadDemo("processed"));
document.getElementById("exportReport").addEventListener("click", exportReport);

window.addEventListener("beforeunload", () => {
  if (activeObjectUrl) URL.revokeObjectURL(activeObjectUrl);
});
