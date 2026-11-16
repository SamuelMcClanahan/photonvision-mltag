#!/usr/bin/env node
import { decode } from "@msgpack/msgpack";

const args = process.argv.slice(2);
const command = args[0] !== undefined ? args[0] : "measure";

function argValue(name, defaultValue = undefined) {
  const idx = args.indexOf(name);
  if (idx === -1) return defaultValue;
  return args[idx + 1] !== undefined ? args[idx + 1] : defaultValue;
}

function hasArg(name) {
  return args.includes(name);
}

function numericArg(name, defaultValue) {
  const parsed = Number(argValue(name, defaultValue));
  if (!Number.isFinite(parsed) || parsed < 0) {
    throw new Error(`${name} must be a non-negative number`);
  }
  return parsed;
}

function usage() {
  console.error(`Usage:
  node scripts/jetson-ws-tune.mjs snapshot [--host 192.168.4.25:5800]
  node scripts/jetson-ws-tune.mjs measure [--host 192.168.4.25:5800] [--duration 12]
  node scripts/jetson-ws-tune.mjs verify [--host 192.168.4.25:5800] [--duration 30] [--min-fps 118] [--min-cameras 2]

Options:
  --host               PhotonVision host:port. Default: 192.168.4.25:5800
  --duration           Measurement duration in seconds. Default: 12
  --settle             Snapshot wait in seconds. Default: 2
  --min-fps            For verify: required average FPS per camera. Default: 118
  --min-cameras        For verify: required active sampled cameras. Default: 2
  --min-target-frames  For verify: required frames with targets per camera. Default: 1
  --min-target-rate    For verify: required target-hit sample rate per camera. Default: 0
  --cameras            Comma-separated camera unique names. Default: all active connected cameras
  --raw                Include raw camera settings snapshots`);
}

if (hasArg("--help") || !["snapshot", "measure", "verify"].includes(command)) {
  usage();
  process.exit(hasArg("--help") ? 0 : 2);
}

const host = argValue("--host", "192.168.4.25:5800");
const durationMs = numericArg("--duration", "12") * 1000;
const settleMs = numericArg("--settle", "2") * 1000;
const verifyMinFps = numericArg("--min-fps", "118");
const verifyMinCameras = numericArg("--min-cameras", "2");
const verifyMinTargetFrames = numericArg("--min-target-frames", "1");
const verifyMinTargetRate = numericArg("--min-target-rate", "0");
if (verifyMinTargetRate > 1) {
  throw new Error("--min-target-rate must be between 0 and 1");
}

const requestedCameras = new Set(
  (argValue("--cameras", "") || "")
    .split(",")
    .map((s) => s.trim())
    .filter(Boolean)
);

function avg(values) {
  return values.length ? values.reduce((sum, value) => sum + value, 0) / values.length : 0;
}

function percentile(values, pct) {
  if (!values.length) return null;
  const sorted = [...values].sort((a, b) => a - b);
  const index = Math.min(sorted.length - 1, Math.max(0, Math.ceil((pct / 100) * sorted.length) - 1));
  return sorted[index];
}

function rounded(value) {
  return value === null || value === undefined ? null : Number(value.toFixed(2));
}

function finite(value) {
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

function targetCount(result) {
  if (Array.isArray(result && result.targets)) return result.targets.length;
  if (typeof (result && result.targetCount) === "number") return result.targetCount;
  if ((result && result.hasTarget) === true || (result && result.hasTargets) === true) return 1;
  if (result && result.bestTarget !== null && result.bestTarget !== undefined) return 1;
  return 0;
}

function resultFps(result) {
  if (!result) return null;
  return finite(result.fps !== undefined ? result.fps : result.framerate !== undefined ? result.framerate : result.FPS);
}

function resultLatency(result) {
  if (!result) return null;
  return finite(
    result.latencyMillis !== undefined ? result.latencyMillis : result.latency !== undefined ? result.latency : result.latencyMS
  );
}

function resultProcessing(result) {
  if (!result) return null;
  return finite(result.processingMs !== undefined ? result.processingMs : result.processingMillis);
}

function resultCaptureAgeAtPublish(result) {
  return finite(result && result.captureAgeAtPublishMs);
}

function resultCaptureEstimatedAge(result) {
  return finite(result && result.captureEstimatedAgeMs);
}

function numericSummary(values, suffix = "") {
  const filtered = values.filter((value) => value !== null);
  return {
    [`avg${suffix}`]: rounded(avg(filtered)),
    [`p50${suffix}`]: rounded(percentile(filtered, 50)),
    [`p95${suffix}`]: rounded(percentile(filtered, 95)),
    [`max${suffix}`]: filtered.length ? rounded(Math.max(...filtered)) : null
  };
}

function cameraSnapshot(camera) {
  const pipeline = camera.currentPipelineSettings || {};
  const videoMode = camera.videoFormatList ? camera.videoFormatList[pipeline.cameraVideoModeIndex] || null : null;
  return {
    nickname: camera.nickname,
    uniqueName: camera.uniqueName,
    enabled: camera.isEnabled,
    connected: camera.isConnected,
    path: camera.cameraPath,
    currentPipelineIndex: camera.currentPipelineIndex,
    cameraVideoModeIndex: pipeline.cameraVideoModeIndex,
    videoMode,
    exposure: pipeline.cameraExposureRaw,
    brightness: pipeline.cameraBrightness,
    gain: pipeline.cameraGain,
    decimate: pipeline.decimate,
    threads: pipeline.threads,
    refineEdges: pipeline.refineEdges,
    decisionMargin: pipeline.decisionMargin,
    useMLDetection: pipeline.useMLDetection,
    mlConfidenceThreshold: pipeline.mlConfidenceThreshold,
    mlNmsThreshold: pipeline.mlNmsThreshold,
    mlRoiPaddingPixels: pipeline.mlRoiPaddingPixels,
    model: pipeline.model ? pipeline.model.nickname || null : null,
    streamingFrameDivisor: pipeline.streamingFrameDivisor,
    inputShouldShow: pipeline.inputShouldShow,
    outputShouldShow: pipeline.outputShouldShow,
    outputShouldDraw: pipeline.outputShouldDraw
  };
}

function settingsSnapshot(settings) {
  const network = settings && settings.networkSettings;
  if (!network) return null;
  return {
    ntServerAddress: network.ntServerAddress,
    runNTServer: network.runNTServer,
    shouldPublishProto: network.shouldPublishProto,
    connectionType: network.connectionType,
    hostname: network.hostname,
    networkingDisabled: network.networkingDisabled
  };
}

function summarizeSamples(samples, cameraSettingsById, expectedCameraIds = []) {
  const summaries = {};
  const cameraIds = new Set([...expectedCameraIds, ...Object.keys(samples)]);
  for (const cameraId of cameraIds) {
    const cameraSamples = samples[cameraId] || [];
    const fpsValues = cameraSamples.map((sample) => sample.fps).filter((value) => value !== null);
    const latencyValues = cameraSamples.map((sample) => sample.latency).filter((value) => value !== null);
    const processingValues = cameraSamples.map((sample) => sample.processing).filter((value) => value !== null);
    const captureAgeAtPublishValues = cameraSamples
      .map((sample) => sample.captureAgeAtPublish)
      .filter((value) => value !== null);
    const captureEstimatedAgeValues = cameraSamples
      .map((sample) => sample.captureEstimatedAge)
      .filter((value) => value !== null);
    const targetValues = cameraSamples.map((sample) => sample.targets);
    const targetFrames = targetValues.filter((count) => count > 0).length;
    summaries[cameraId] = {
      nickname: cameraSettingsById[cameraId] ? cameraSettingsById[cameraId].nickname || cameraId : cameraId,
      resultSamples: cameraSamples.length,
      samples: fpsValues.length,
      avgFps: rounded(avg(fpsValues)),
      minFps: fpsValues.length ? rounded(Math.min(...fpsValues)) : null,
      p50Fps: rounded(percentile(fpsValues, 50)),
      p95Fps: rounded(percentile(fpsValues, 95)),
      maxFps: fpsValues.length ? rounded(Math.max(...fpsValues)) : null,
      ...numericSummary(latencyValues, "LatencyMs"),
      ...numericSummary(processingValues, "ProcessingMs"),
      ...numericSummary(captureAgeAtPublishValues, "CaptureAgeAtPublishMs"),
      ...numericSummary(captureEstimatedAgeValues, "CaptureEstimatedAgeMs"),
      avgTargets: rounded(avg(targetValues)),
      targetFrames,
      targetRate: cameraSamples.length ? Number((targetFrames / cameraSamples.length).toFixed(3)) : 0
    };
  }
  return summaries;
}

function verifySummaries(summaries) {
  const failures = [];
  const cameraIds = Object.keys(summaries);
  if (cameraIds.length < verifyMinCameras) {
    failures.push(`Only ${cameraIds.length} selected camera(s); required ${verifyMinCameras}`);
  }
  for (const [cameraId, summary] of Object.entries(summaries)) {
    const name = summary.nickname || cameraId;
    if (summary.samples <= 0) failures.push(`${name} produced no FPS samples`);
    if (!(summary.avgFps >= verifyMinFps)) failures.push(`${name} avg FPS ${summary.avgFps} < ${verifyMinFps}`);
    if (summary.targetFrames < verifyMinTargetFrames) {
      failures.push(`${name} target frames ${summary.targetFrames} < ${verifyMinTargetFrames}`);
    }
    if (summary.targetRate < verifyMinTargetRate) {
      failures.push(`${name} target rate ${summary.targetRate} < ${verifyMinTargetRate}`);
    }
  }
  return {
    ok: failures.length === 0,
    minFps: verifyMinFps,
    minCameras: verifyMinCameras,
    minTargetFrames: verifyMinTargetFrames,
    minTargetRate: verifyMinTargetRate,
    failures
  };
}

const ws = new WebSocket(`ws://${host}/websocket_data`);
ws.binaryType = "arraybuffer";

let cameraSettingsById = {};
let selectedCameraIds = [];
const samples = {};
let settings = null;
let ntConnectionInfo = null;
let metrics = null;
let measuring = false;
let done = false;
let measureTimer = null;
let snapshotTimer = null;

function finish(exitCode = 0) {
  if (done) return;
  done = true;
  clearTimeout(measureTimer);
  clearTimeout(snapshotTimer);

  const summary = summarizeSamples(samples, cameraSettingsById, selectedCameraIds);
  const verification = command === "verify" ? verifySummaries(summary) : undefined;
  const output = {
    command,
    host,
    network: settingsSnapshot(settings),
    ntConnectionInfo,
    metrics,
    activeCameras: selectedCameraIds.map((id) => cameraSnapshot(cameraSettingsById[id])),
    summary,
    ...(verification ? { verification } : {})
  };
  if (hasArg("--raw")) {
    output.rawCameraSettings = Object.values(cameraSettingsById);
  }
  console.log(JSON.stringify(output, null, 2));
  ws.close();
  process.exit(command === "verify" && !verification.ok ? 1 : exitCode);
}

function startMeasurement() {
  measuring = true;
  measureTimer = setTimeout(() => finish(0), durationMs);
}

ws.onmessage = (event) => {
  const data = decode(new Uint8Array(event.data));
  if (data.settings) settings = data.settings;
  if (data.ntConnectionInfo !== undefined) ntConnectionInfo = data.ntConnectionInfo;
  if (data.metrics !== undefined) metrics = data.metrics;

  if (data.cameraSettings) {
    cameraSettingsById = Object.fromEntries(data.cameraSettings.map((camera) => [camera.uniqueName, camera]));
    selectedCameraIds = data.cameraSettings
      .filter((camera) => camera.isEnabled && camera.isConnected)
      .filter((camera) => requestedCameras.size === 0 || requestedCameras.has(camera.uniqueName))
      .map((camera) => camera.uniqueName);

    if (command === "snapshot") {
      if (!snapshotTimer) snapshotTimer = setTimeout(() => finish(0), settleMs);
      return;
    }
    if (!measuring) startMeasurement();
  }

  if (measuring && data.updatePipelineResult) {
    for (const [cameraId, result] of Object.entries(data.updatePipelineResult)) {
      if (selectedCameraIds.length > 0 && !selectedCameraIds.includes(cameraId)) continue;
      if (!samples[cameraId]) samples[cameraId] = [];
      samples[cameraId].push({
        fps: resultFps(result),
        latency: resultLatency(result),
        processing: resultProcessing(result),
        captureAgeAtPublish: resultCaptureAgeAtPublish(result),
        captureEstimatedAge: resultCaptureEstimatedAge(result),
        targets: targetCount(result)
      });
    }
  }
};

ws.onerror = (error) => {
  if (done) return;
  console.error("WebSocket error", error && error.message !== undefined ? error.message : error);
  process.exit(1);
};

setTimeout(() => {
  if (!done) {
    console.error("Timed out waiting for PhotonVision websocket data");
    finish(2);
  }
}, durationMs + settleMs + 10000);
