#!/usr/bin/env bash
# download_ckpt.sh — pull a pi05_g1_sonic checkpoint from a remote training host via rsync.
#
# Usage examples:
#   export REMOTE_BASE=/path/to/openpi/checkpoints/pi05_g1_sonic   # required (or -r)
#   ./download_ckpt.sh -h user@1.2.3.4 5000              # 拉 step=5000
#   ./download_ckpt.sh -h user@1.2.3.4 latest            # 自动取最大 step
#   ./download_ckpt.sh -h user@1.2.3.4 --list            # 列出远程已有 step
#   ./download_ckpt.sh -h user@1.2.3.4 -p 30000          # 只下 params/，不要 optimizer/state
#   REMOTE_HOST=user@1.2.3.4 ./download_ckpt.sh 10000    # host 也可走环境变量
#
# Defaults assume the standard pi05_g1_sonic layout:
#   <REMOTE_BASE>/<exp>/<step>/{params, train_state, assets, ...}

set -euo pipefail

REMOTE_HOST="${REMOTE_HOST:-}"
REMOTE_PORT="${REMOTE_PORT:-22}"
REMOTE_BASE="${REMOTE_BASE:-}"
LOCAL_BASE="${LOCAL_BASE:-./checkpoints/pi05_g1_sonic}"
EXP="take_walk_put_v0"
PARAMS_ONLY=0
LIST_MODE=0
DRY_RUN=0
STEP=""

usage() {
    cat <<EOF
Usage: $0 [options] <step>

Pull a pi05_g1_sonic checkpoint from a remote host via rsync over ssh.

Positional:
  <step>             Checkpoint step number (e.g. 5000), or "latest" to fetch
                     the highest-numbered checkpoint on remote.

Options:
  -h, --host HOST    Remote ssh target, e.g. user\@10.1.2.3 (or set REMOTE_HOST env)
      --port PORT    Remote ssh port (default: 22; or set REMOTE_PORT env)
  -e, --exp NAME     Experiment subdir (default: take_walk_put_v0)
  -o, --out DIR      Local destination base (default: ./checkpoints/pi05_g1_sonic
                     or \$LOCAL_BASE). Final path = <out>/<exp>/<step>
  -r, --remote-base  Remote checkpoint base dir, e.g.
                     /path/to/openpi/checkpoints/pi05_g1_sonic (required;
                     or set REMOTE_BASE env)
  -p, --params-only  Download only params/ (≈ 1/3 the size; OK for inference,
                     not for resume-training)
  -l, --list         List remote step directories and exit
  -n, --dry-run      Show the rsync command without running it
      --help         Show this help

Environment overrides:
  REMOTE_HOST, REMOTE_PORT, REMOTE_BASE, LOCAL_BASE
EOF
}

while (( $# )); do
    case "$1" in
        -h|--host)        REMOTE_HOST="$2"; shift 2 ;;
        --port)           REMOTE_PORT="$2"; shift 2 ;;
        -e|--exp)         EXP="$2"; shift 2 ;;
        -o|--out)         LOCAL_BASE="$2"; shift 2 ;;
        -r|--remote-base) REMOTE_BASE="$2"; shift 2 ;;
        -p|--params-only) PARAMS_ONLY=1; shift ;;
        -l|--list)        LIST_MODE=1; shift ;;
        -n|--dry-run)     DRY_RUN=1; shift ;;
        --help)           usage; exit 0 ;;
        --)               shift; break ;;
        -*)               echo "error: unknown option: $1" >&2; usage; exit 1 ;;
        *)                STEP="$1"; shift ;;
    esac
done

if [[ -z "$REMOTE_HOST" ]]; then
    echo "error: --host or REMOTE_HOST env var is required" >&2
    usage
    exit 1
fi

if [[ -z "$REMOTE_BASE" ]]; then
    echo "error: --remote-base or REMOTE_BASE env var is required" >&2
    usage
    exit 1
fi

REMOTE_EXP_DIR="$REMOTE_BASE/$EXP"

SSH_CMD=(ssh -p "$REMOTE_PORT")
RSYNC_SSH="ssh -p $REMOTE_PORT"

# Helper: return sorted unique numeric step names found on remote.
list_steps() {
    "${SSH_CMD[@]}" "$REMOTE_HOST" "set -e; \
        if [ ! -d '$REMOTE_EXP_DIR' ]; then exit 0; fi; \
        ls -1 '$REMOTE_EXP_DIR' 2>/dev/null | awk '/^[0-9]+\$/' | sort -n"
}

if (( LIST_MODE )); then
    echo "remote: $REMOTE_HOST:$REMOTE_EXP_DIR"
    steps=$(list_steps || true)
    if [[ -z "$steps" ]]; then
        echo "  (no checkpoint steps found)"
    else
        echo "$steps"
    fi
    exit 0
fi

if [[ -z "$STEP" ]]; then
    echo "error: <step> required (use 'latest' or pass a number; '--list' to inspect)" >&2
    usage
    exit 1
fi

if [[ "$STEP" == "latest" ]]; then
    STEP=$(list_steps | tail -n1)
    if [[ -z "$STEP" ]]; then
        echo "error: no checkpoints found at $REMOTE_HOST:$REMOTE_EXP_DIR" >&2
        exit 1
    fi
    echo "resolved 'latest' -> step $STEP"
fi

# Validate the requested step exists on remote (gives a clearer error than rsync's).
if ! "${SSH_CMD[@]}" "$REMOTE_HOST" "test -d '$REMOTE_EXP_DIR/$STEP'"; then
    echo "error: $REMOTE_HOST:$REMOTE_EXP_DIR/$STEP does not exist" >&2
    echo "available steps:" >&2
    list_steps | sed 's/^/  /' >&2
    exit 1
fi

LOCAL_DIR="$LOCAL_BASE/$EXP/$STEP"

if (( PARAMS_ONLY )); then
    SRC="$REMOTE_HOST:$REMOTE_EXP_DIR/$STEP/params/"
    DST="$LOCAL_DIR/params/"
else
    SRC="$REMOTE_HOST:$REMOTE_EXP_DIR/$STEP/"
    DST="$LOCAL_DIR/"
fi

mkdir -p "$DST"

# rsync: -a archive, -h human sizes, -P partial+progress, --info=progress2 totals.
RSYNC_CMD=(rsync -ahP --info=progress2 -e "$RSYNC_SSH" "$SRC" "$DST")

echo "==> ${RSYNC_CMD[*]}"
if (( DRY_RUN )); then
    echo "(dry run; not executing)"
    exit 0
fi

"${RSYNC_CMD[@]}"

echo
echo "done: $LOCAL_DIR"
echo "use it like:"
echo "  uv run scripts/serve_policy.py policy:checkpoint \\"
echo "      --policy.config=pi05_g1_sonic \\"
echo "      --policy.dir=$LOCAL_DIR"
