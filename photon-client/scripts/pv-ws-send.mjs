#!/usr/bin/env node
import { encode } from "@msgpack/msgpack";

const args = process.argv.slice(2);

function argValue(name, defaultValue = undefined) {
  const idx = args.indexOf(name);
  if (idx === -1) return defaultValue;
  return args[idx + 1] !== undefined ? args[idx + 1] : defaultValue;
}

function usage() {
  console.error(`Usage:
  node scripts/pv-ws-send.mjs pipeline --host photonvision.local:5800 --camera CAMERA_ID --json '{"useMLDetection":true}'
  node scripts/pv-ws-send.mjs demand --host photonvision.local:5800 --camera CAMERA_ID --input false --output false`);
}

const command = args[0];
if (!["pipeline", "demand"].includes(command) || args.includes("--help")) {
  usage();
  process.exit(args.includes("--help") ? 0 : 2);
}

const host = argValue("--host", "photonvision.local:5800");
const cameraUniqueName = argValue("--camera", "");
if (!cameraUniqueName) {
  throw new Error("--camera is required");
}

function boolArg(name, defaultValue) {
  const raw = argValue(name, String(defaultValue));
  return !["0", "false", "no", "off"].includes(String(raw).toLowerCase());
}

const ws = new WebSocket(`ws://${host}/websocket_data`);
ws.binaryType = "arraybuffer";

ws.onopen = () => {
  let payload;
  if (command === "pipeline") {
    payload = JSON.parse(argValue("--json", "{}"));
    payload.cameraUniqueName = cameraUniqueName;
    payload = { changePipelineSetting: payload };
  } else {
    payload = {
      streamDemand: {
        cameraUniqueName,
        demandId: argValue("--demand-id", "codex-benchmark"),
        input: boolArg("--input", false),
        output: boolArg("--output", false)
      }
    };
  }

  ws.send(encode(payload));
  setTimeout(() => ws.close(), 200);
};

ws.onerror = (error) => {
  console.error("WebSocket error", error && error.message !== undefined ? error.message : error);
  process.exit(1);
};

ws.onclose = () => process.exit(0);
