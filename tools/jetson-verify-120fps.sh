#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'USAGE'
Usage:
  tools/jetson-verify-120fps.sh [options]

Verifies the live Jetson PhotonVision setup without changing quality-affecting
camera or pipeline settings. The check passes only when every sampled camera
meets the FPS threshold and produces target hits.

Options:
  --pv-host HOST:PORT       PhotonVision web host. If omitted, derive from SSH.
  --ssh-host HOST           Jetson SSH host used to derive PV host. Default: jetson.
  --duration SECONDS        Verification duration. Default: 30.
  --min-fps FPS             Required average FPS per camera. Default: 118.
  --min-cameras N           Required active connected sampled cameras. Default: 2.
  --min-target-frames N     Required frames with targets per camera. Default: 1.
  --min-target-rate RATE    Required target-hit sample rate per camera, 0-1. Default: 0.
  --cameras A,B             Optional comma-separated PhotonVision camera unique names.
  --skip-ssh-check          Do not SSH to Jetson before checking PV websocket.
  --help                    Show this help.

Environment:
  SSHPASS                   Optional sshpass password for the Jetson.

Examples:
  SSHPASS=... tools/jetson-verify-120fps.sh --ssh-host jetson
  tools/jetson-verify-120fps.sh --pv-host 192.168.4.24:5800 --duration 45
USAGE
}

pv_host=""
ssh_host="jetson"
duration="30"
min_fps="118"
min_cameras="2"
min_target_frames="1"
min_target_rate="0"
cameras=""
skip_ssh_check=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --pv-host) pv_host="${2:-}"; shift 2 ;;
        --ssh-host) ssh_host="${2:-}"; shift 2 ;;
        --duration) duration="${2:-}"; shift 2 ;;
        --min-fps) min_fps="${2:-}"; shift 2 ;;
        --min-cameras) min_cameras="${2:-}"; shift 2 ;;
        --min-target-frames) min_target_frames="${2:-}"; shift 2 ;;
        --min-target-rate) min_target_rate="${2:-}"; shift 2 ;;
        --cameras) cameras="${2:-}"; shift 2 ;;
        --skip-ssh-check) skip_ssh_check=true; shift ;;
        --help|-h) usage; exit 0 ;;
        *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
    esac
done

ssh_base=(ssh -o StrictHostKeyChecking=accept-new -o PreferredAuthentications=password -o PubkeyAuthentication=no -o NumberOfPasswordPrompts=1 -o ConnectTimeout=8 "$ssh_host")
if [[ -n "${SSHPASS:-}" ]] && command -v sshpass >/dev/null 2>&1; then
    ssh_base=(sshpass -e "${ssh_base[@]}")
fi

if [[ -z "$pv_host" && "$skip_ssh_check" == false ]]; then
    jetson_ip="$("${ssh_base[@]}" "python3 - <<'PY'
import socket

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
try:
    sock.connect(('8.8.8.8', 80))
    print(sock.getsockname()[0])
finally:
    sock.close()
PY")"
    if [[ -z "$jetson_ip" ]]; then
        echo "Could not derive Jetson IP from SSH host '$ssh_host'." >&2
        exit 2
    fi
    pv_host="${jetson_ip}:5800"
fi

if [[ -z "$pv_host" ]]; then
    echo "Provide --pv-host, or omit --skip-ssh-check so the script can derive it over SSH." >&2
    exit 2
fi

if [[ "$skip_ssh_check" == false ]]; then
    echo "== Jetson state =="
    "${ssh_base[@]}" "set -e
hostname
ip -br addr
echo
pgrep -af 'photonvision|photon-server' || true
echo
ls -l /dev/video* 2>/dev/null || true"
    echo
fi

verify_cmd=(node photon-client/scripts/jetson-ws-tune.mjs verify
    --host "$pv_host"
    --duration "$duration"
    --min-fps "$min_fps"
    --min-cameras "$min_cameras"
    --min-target-frames "$min_target_frames"
    --min-target-rate "$min_target_rate")

if [[ -n "$cameras" ]]; then
    verify_cmd+=(--cameras "$cameras")
fi

echo "== PhotonVision 120 FPS / target-hit verification =="
echo "PV host: $pv_host"
echo "Duration: ${duration}s"
echo "Minimum average FPS per camera: $min_fps"
echo "Minimum cameras: $min_cameras"
echo "Minimum target frames per camera: $min_target_frames"
echo "Minimum target-hit rate per camera: $min_target_rate"
echo

"${verify_cmd[@]}"
