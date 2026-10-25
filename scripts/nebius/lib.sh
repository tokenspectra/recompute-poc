# shellcheck shell=bash
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/config.env"
die() { echo "error: $*" >&2; exit 1; }
need() { command -v "$1" >/dev/null 2>&1 || die "missing command: $1"; }
ssh_vm() { local ip="$1"; shift
  ssh -o StrictHostKeyChecking=accept-new -o ServerAliveInterval=30 "$SSH_USER@$ip" "$@"; }
