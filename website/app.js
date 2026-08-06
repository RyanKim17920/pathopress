/*
 * PathoPress static predictor. Its completion recipe is adapted from Microsoft
 * BenchPress (MIT); see NOTICE.txt and the repository THIRD_PARTY_NOTICES.md.
 */
"use strict";

const state = { data: null, modelIndex: 0, evaluationIndex: 0 };

function escapeHtml(value) {
  return String(value).replace(/[&<>'"]/g, character => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", "'": "&#39;", '"': "&quot;"
  })[character]);
}

function scoreText(value) { return Number.isFinite(value) ? value.toFixed(1) : "—"; }

async function loadData() {
  const response = await fetch("data.json");
  if (!response.ok) throw new Error(`data.json returned ${response.status}`);
  const data = await response.json();
  if (data.schema_version !== "pathopress-static-predictor-v1") {
    throw new Error("Unsupported generated website-data schema");
  }
  const rows = data.models.length, columns = data.evaluations.length;
  for (const key of ["observed", "predictions", "sources", "prediction_intervals"]) {
    if (!Array.isArray(data[key]) || data[key].length !== rows || data[key].some(row => row.length !== columns)) {
      throw new Error(`Invalid ${key} matrix dimensions`);
    }
  }
  if (!data.new_model_confidence || data.new_model_confidence.artifact_type !== "pathopress_new_model_group_conformal_v1") {
    throw new Error("Missing or unsupported new-model confidence artifact");
  }
  return data;
}

function setupLookup() {
  const { data } = state;
  const modelInput = document.getElementById("model-input");
  const evaluationInput = document.getElementById("evaluation-input");
  const modelOptions = document.getElementById("model-options");
  const evaluationOptions = document.getElementById("evaluation-options");
  data.models.forEach((model, index) => {
    const option = document.createElement("option");
    option.value = model.model_id;
    option.label = model.provider ? `${model.model_id} — ${model.provider}` : model.model_id;
    option.dataset.index = index;
    modelOptions.appendChild(option);
  });
  data.evaluations.forEach((evaluation, index) => {
    const option = document.createElement("option");
    option.value = evaluation.evaluation_id;
    option.label = `${evaluation.evaluation_id} — ${evaluation.suite_id}`;
    option.dataset.index = index;
    evaluationOptions.appendChild(option);
  });
  modelInput.value = data.models[0].model_id;
  evaluationInput.value = data.evaluations[0].evaluation_id;
  const commit = () => {
    const i = data.models.findIndex(model => model.model_id === modelInput.value.trim());
    const j = data.evaluations.findIndex(item => item.evaluation_id === evaluationInput.value.trim());
    if (i >= 0) state.modelIndex = i;
    if (j >= 0) state.evaluationIndex = j;
    renderLookup();
  };
  modelInput.addEventListener("change", commit);
  evaluationInput.addEventListener("change", commit);
  renderLookup();
}

function renderLookup() {
  const { data, modelIndex: i, evaluationIndex: j } = state;
  const model = data.models[i], evaluation = data.evaluations[j];
  const observed = data.observed[i][j], prediction = data.predictions[i][j];
  const interval = data.prediction_intervals[i][j], source = data.sources[i][j];
  const value = observed === null ? prediction : observed;
  const kind = observed === null ? "Predicted" : "Reported";
  const intervalLine = interval
    ? `<p>Calibrated 90% held-out-cell interval: <strong>${scoreText(interval[0])}–${scoreText(interval[1])}</strong></p>`
    : observed === null
      ? "<p class=\"meta\">No applicable confidence artifact.</p>"
      : "";
  const sourceLine = source?.url
    ? `<p><a href="${escapeHtml(source.url)}" target="_blank" rel="noopener">Open reported-score source ↗</a> · audit: ${escapeHtml(source.audit_status)}</p>`
    : "";
  document.getElementById("lookup-result").innerHTML = `
    <span class="tag">${kind}</span>
    <div class="score">${scoreText(value)}</div>
    <h3>${escapeHtml(model.model_id)} on ${escapeHtml(evaluation.evaluation_id)}</h3>
    <p class="meta">${escapeHtml(evaluation.suite_id)} · ${escapeHtml(evaluation.metric)} · normalized 0–100</p>
    ${intervalLine}${sourceLine}`;
  const ranked = data.models.map((item, index) => ({
    index, id: item.model_id,
    observed: data.observed[index][j], predicted: data.predictions[index][j]
  })).sort((left, right) => (right.observed ?? right.predicted) - (left.observed ?? left.predicted));
  document.getElementById("leaderboard-body").innerHTML = ranked.map((row, rank) => `
    <tr class="${row.index === i ? "current-row" : ""}">
      <td>${rank + 1}</td><td>${escapeHtml(row.id)}</td>
      <td>${scoreText(row.observed ?? row.predicted)}</td>
      <td>${row.observed === null ? "predicted" : "reported"}</td>
    </tr>`).join("");
}

function addKnownScoreRow(evaluationIndex = null) {
  const row = document.createElement("div");
  row.className = "known-row";
  const select = document.createElement("select");
  select.setAttribute("aria-label", "Known evaluation");
  state.data.evaluations.forEach((evaluation, index) => {
    const option = document.createElement("option");
    option.value = String(index);
    option.textContent = `${evaluation.evaluation_id} (${evaluation.suite_id})`;
    select.appendChild(option);
  });
  select.value = String(evaluationIndex ?? Math.min(document.querySelectorAll(".known-row").length, state.data.evaluations.length - 1));
  const input = document.createElement("input");
  input.type = "number"; input.min = "0"; input.max = "100"; input.step = "any";
  input.placeholder = "0–100"; input.setAttribute("aria-label", "Known normalized score");
  const remove = document.createElement("button");
  remove.type = "button"; remove.className = "remove-row"; remove.textContent = "Remove";
  remove.onclick = () => row.remove();
  row.append(select, input, remove);
  document.getElementById("known-score-rows").appendChild(row);
}

class NumpyRandomState {
  // MT19937 + legacy polar Gaussian, matching numpy.random.RandomState.
  constructor(seed) {
    this.state = new Uint32Array(624); this.index = 624;
    this.hasGaussian = false; this.gaussian = 0;
    this.state[0] = seed >>> 0;
    for (let i = 1; i < 624; i++) {
      const previous = this.state[i - 1];
      this.state[i] = (Math.imul(1812433253, previous ^ previous >>> 30) + i) >>> 0;
    }
  }
  twist() {
    for (let i = 0; i < 624; i++) {
      const value = (this.state[i] & 0x80000000) | (this.state[(i + 1) % 624] & 0x7fffffff);
      this.state[i] = this.state[(i + 397) % 624] ^ (value >>> 1) ^ ((value & 1) ? 0x9908b0df : 0);
    }
    this.index = 0;
  }
  uint32() {
    if (this.index >= 624) this.twist();
    let value = this.state[this.index++];
    value ^= value >>> 11;
    value ^= (value << 7) & 0x9d2c5680;
    value ^= (value << 15) & 0xefc60000;
    value ^= value >>> 18;
    return value >>> 0;
  }
  random() {
    const left = this.uint32() >>> 5, right = this.uint32() >>> 6;
    return (left * 67108864 + right) / 9007199254740992;
  }
  normal() {
    if (this.hasGaussian) { this.hasGaussian = false; return this.gaussian; }
    let first, second, radiusSquared;
    do {
      first = 2 * this.random() - 1;
      second = 2 * this.random() - 1;
      radiusSquared = first * first + second * second;
    } while (radiusSquared >= 1 || radiusSquared === 0);
    const scale = Math.sqrt(-2 * Math.log(radiusSquared) / radiusSquared);
    this.gaussian = first * scale; this.hasGaussian = true;
    return second * scale;
  }
}

function logitPercent(value) {
  const probability = Math.min(99.5, Math.max(.5, value)) / 100;
  return Math.log(probability / (1 - probability));
}
function inverseLogit(value) { return 100 / (1 + Math.exp(-value)); }

function newModelInterval(prediction, evaluation, knownIndexes) {
  const artifact = state.data.new_model_confidence;
  const supported = artifact.supported_probe_counts.filter(value => value <= knownIndexes.length);
  if (!supported.length) return { status: "abstained", reason: "at least one known score is required" };
  const k = Math.max(...supported);
  const entry = artifact.by_evaluation[evaluation.evaluation_id];
  const evaluationRisk = entry?.suite_id === evaluation.suite_id ? entry.by_k?.[String(k)] : null;
  if (!evaluationRisk?.supported) {
    return { status: "abstained", k, reason: "unsupported column: too few distinct calibration models" };
  }
  const sameSuite = knownIndexes.some(index => state.data.evaluations[index].suite_id === evaluation.suite_id);
  const choices = [
    ["evaluation+suite_same_probe", "suite_same_probe", `${k}|${evaluation.suite_id}|${sameSuite}`],
    ["evaluation+suite", "suite", `${k}|${evaluation.suite_id}`],
    ["evaluation+global_k", "global_k", String(k)]
  ];
  let context = null, scope = null;
  for (const [candidateScope, section, key] of choices) {
    const candidate = artifact.context_risk[section]?.[key];
    if (candidate?.supported) { context = candidate; scope = candidateScope; break; }
  }
  const scale = artifact.conformal_scale_by_k[String(k)]?.scale;
  if (!context || !Number.isFinite(scale)) return { status: "abstained", k, reason: "unsupported calibration context" };
  const risk = .5 * evaluationRisk.risk_median + .5 * context.risk_median;
  const radius = risk * scale;
  return {
    status: "calibrated", k, risk, scope,
    lower: Math.max(0, prediction - radius), upper: Math.min(100, prediction + radius),
    evaluationModels: evaluationRisk.n_models, evaluationPredictions: evaluationRisk.n_predictions,
    contextModels: context.n_models, contextPredictions: context.n_predictions
  };
}

function completeRank1(rawMatrix) {
  const nRows = rawMatrix.length, nColumns = rawMatrix[0].length;
  const observed = rawMatrix.map(row => row.map(Number.isFinite));
  const transformed = rawMatrix.map(row => row.slice());
  const means = Array(nColumns).fill(0), stds = Array(nColumns).fill(1);
  for (let j = 0; j < nColumns; j++) {
    const values = [];
    for (let i = 0; i < nRows; i++) if (observed[i][j]) values.push(logitPercent(rawMatrix[i][j]));
    means[j] = values.reduce((sum, value) => sum + value, 0) / values.length;
    const variance = values.reduce((sum, value) => sum + (value - means[j]) ** 2, 0) / values.length;
    stds[j] = Math.sqrt(variance) < 1e-12 ? 1 : Math.sqrt(variance);
    for (let i = 0; i < nRows; i++) if (observed[i][j]) transformed[i][j] = (logitPercent(rawMatrix[i][j]) - means[j]) / stds[j];
  }
  const predictions = Array.from({ length: nRows }, () => Array(nColumns).fill(0));
  for (let ensemble = 0; ensemble < 10; ensemble++) {
    const random = new NumpyRandomState(42 + ensemble);
    const rowBias = Array(nRows).fill(0), columnBias = Array(nColumns).fill(0);
    const rowFactor = Array.from({ length: nRows }, () => .01 * random.normal());
    const columnFactor = Array.from({ length: nColumns }, () => .01 * random.normal());
    let count = 0, mean = 0;
    for (let i = 0; i < nRows; i++) for (let j = 0; j < nColumns; j++) if (observed[i][j]) { mean += transformed[i][j]; count++; }
    mean /= count;
    for (let iteration = 0; iteration < 40; iteration++) {
      for (let i = 0; i < nRows; i++) {
        let n = 0, a01 = 0, a11 = .1, b0 = 0, b1 = 0;
        for (let j = 0; j < nColumns; j++) if (observed[i][j]) {
          const target = transformed[i][j] - mean - columnBias[j];
          n++; a01 += columnFactor[j]; a11 += columnFactor[j] ** 2;
          b0 += target; b1 += target * columnFactor[j];
        }
        if (n) {
          const a00 = n + .1, determinant = a00 * a11 - a01 * a01;
          rowBias[i] = (b0 * a11 - b1 * a01) / determinant;
          rowFactor[i] = (a00 * b1 - a01 * b0) / determinant;
        }
      }
      for (let j = 0; j < nColumns; j++) {
        let n = 0, a01 = 0, a11 = .1, b0 = 0, b1 = 0;
        for (let i = 0; i < nRows; i++) if (observed[i][j]) {
          const target = transformed[i][j] - mean - rowBias[i];
          n++; a01 += rowFactor[i]; a11 += rowFactor[i] ** 2;
          b0 += target; b1 += target * rowFactor[i];
        }
        if (n) {
          const a00 = n + .1, determinant = a00 * a11 - a01 * a01;
          columnBias[j] = (b0 * a11 - b1 * a01) / determinant;
          columnFactor[j] = (a00 * b1 - a01 * b0) / determinant;
        }
      }
      let residual = 0;
      for (let i = 0; i < nRows; i++) for (let j = 0; j < nColumns; j++) if (observed[i][j]) {
        residual += transformed[i][j] - rowBias[i] - columnBias[j] - rowFactor[i] * columnFactor[j];
      }
      mean = residual / count;
    }
    for (let i = 0; i < nRows; i++) for (let j = 0; j < nColumns; j++) {
      predictions[i][j] += (mean + rowBias[i] + columnBias[j] + rowFactor[i] * columnFactor[j]) / 10;
    }
  }
  return predictions.map((row, i) => row.map((value, j) => observed[i][j]
    ? rawMatrix[i][j]
    : Math.min(100, Math.max(0, inverseLogit(value * stds[j] + means[j])))));
}

async function predictNewModel() {
  const button = document.getElementById("predict-new-model");
  const status = document.getElementById("add-model-status");
  const known = new Map();
  try {
    for (const row of document.querySelectorAll(".known-row")) {
      const index = Number(row.querySelector("select").value);
      const raw = row.querySelector("input").value.trim();
      if (!raw) continue;
      const value = Number(raw);
      if (!Number.isFinite(value) || value < 0 || value > 100) throw new Error("Every known normalized score must be between 0 and 100.");
      if (known.has(index)) throw new Error("Each known evaluation may appear only once.");
      known.set(index, value);
    }
    if (!known.size) throw new Error("Enter at least one known score.");
    button.disabled = true; status.textContent = "Running deterministic rank-1 completion locally…";
    await new Promise(resolve => setTimeout(resolve, 20));
    const matrix = state.data.observed.map(row => row.map(value => value === null ? NaN : value));
    const newRow = Array(state.data.evaluations.length).fill(NaN);
    known.forEach((value, index) => { newRow[index] = value; });
    matrix.push(newRow);
    const prediction = completeRank1(matrix).at(-1);
    const knownIndexes = [...known.keys()];
    document.getElementById("new-model-body").innerHTML = state.data.evaluations.map((evaluation, index) => `
      <tr><td>${escapeHtml(evaluation.evaluation_id)}</td><td>${escapeHtml(evaluation.suite_id)}</td>
      <td>${scoreText(prediction[index])}</td><td>${known.has(index) ? "provided" : "predicted"}</td>
      <td>${known.has(index) ? "not applicable" : (() => {
        const result = newModelInterval(prediction[index], evaluation, knownIndexes);
        return result.status === "calibrated"
          ? `${scoreText(result.lower)}–${scoreText(result.upper)} (empirical 90%; k=${result.k}; risk ${result.risk.toFixed(2)}; ${result.scope}; eval ${result.evaluationModels} groups/${result.evaluationPredictions} predictions; context ${result.contextModels} groups/${result.contextPredictions} predictions)`
          : `abstained (${escapeHtml(result.reason)})`;
      })()}</td></tr>`).join("");
    document.getElementById("new-model-results").hidden = false;
    const calibrationK = Math.max(...state.data.new_model_confidence.supported_probe_counts.filter(value => value <= known.size));
    status.textContent = `${known.size} known score${known.size === 1 ? "" : "s"}; ${prediction.length - known.size} missing evaluations completed with the conservative k=${calibrationK} confidence bucket. Empirical intervals are retrospective, not clinical guarantees. Nothing was uploaded.`;
  } catch (error) {
    status.textContent = error.message;
  } finally { button.disabled = false; }
}

async function main() {
  try {
    state.data = await loadData();
    setupLookup();
    document.getElementById("matrix-summary").textContent = `${state.data.meta.models} supported models × ${state.data.meta.evaluations} evaluations; ${state.data.meta.observations} reported cells.`;
    document.getElementById("add-score-row").onclick = () => addKnownScoreRow();
    document.getElementById("predict-new-model").onclick = predictNewModel;
    addKnownScoreRow(0); addKnownScoreRow(1); addKnownScoreRow(2);
  } catch (error) {
    document.getElementById("lookup-result").innerHTML = `<p>Could not load generated data: ${escapeHtml(error.message)}</p>`;
  }
}

main();
