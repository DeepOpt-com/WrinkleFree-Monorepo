#!/bin/bash
#
# Project Sync Script
# Syncs a local project repository to a remote machine and keeps it in sync.
#
# Dependencies:
#   - Local: rsync, inotifywait (from inotify-tools package)
#   - Remote: rsync, uv, tmux (auto-installed if missing)
#
# Configuration:
#   Create a .sync.conf file in this directory to define presets and project settings.
#   See --help for format.
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_NAME="$(basename "$SCRIPT_DIR")"
CONFIG_FILE=""
LOCK_DIR="$HOME/.cache/sync-sh"
mkdir -p "$LOCK_DIR"

# Check for config file (project-local only)
if [[ -f "$SCRIPT_DIR/.sync.conf" ]]; then
    CONFIG_FILE="$SCRIPT_DIR/.sync.conf"
fi

# Defaults
LOCAL_DIR="$SCRIPT_DIR"
REMOTE_HOST=""
REMOTE_PORT=""
REMOTE_DIR=""
REMOTE_USER=""
SSH_KEY=""
SKIP_DEPS=false
WATCH_MODE=true
SETUP_CREDS=false
STATUS_CHECK=false
CURRENT_PRESET=""

# Credential files to sync (from ~/.config/.env.global and GCP)
ENV_GLOBAL_FILE="$HOME/.config/.env.global"
GCP_CREDS_FILE="$HOME/.config/gcloud/application_default_credentials.json"
GCP_SERVICE_ACCOUNT=""  # Set in config if needed

# Project-specific settings (can be overridden in config)
UV_PROJECTS=""  # Space-separated list of subdirectories to run uv sync on
PATH_FIXES=()   # Array of path fix commands (set from config)

# Files/directories to exclude from sync
EXCLUDES=(
    '.git'
    '__pycache__'
    '*.pyc'
    '.venv'
    'venv'
    '.idea'
    '.vscode'
    '.DS_Store'
    'node_modules'
    '.sync.conf'
)

usage() {
    cat << 'EOF'
Usage: sync.sh [OPTIONS]

Sync project repository to a remote machine with live file watching.
Automatically sets up GCP, WANDB, and other credentials on remote.

Options:
  -h, --host HOST       Remote host (hostname or user@host)
  -p, --port PORT       SSH port (default: 22)
  -d, --dir DIR         Remote directory path
  -i, --identity FILE   SSH identity file (private key)
  -u, --user USER       Remote username (alternative to user@host)
  --preset NAME         Use a preset from config file
  --skip-deps           Skip dependency installation on remote
  --no-watch            Only do initial sync, don't watch for changes
  --setup-creds         Only setup credentials, don't sync files
  --status              Check if sync is running for a preset
  --list-presets        List available presets from config
  --help                Show this help message

Configuration File:
  Create .sync.conf in the project directory.

  Format:
    # Global project settings (optional)
    [project]
    uv_projects=SubProject1 SubProject2
    path_fix=sed -i 's|pattern|replacement|g' /path/to/file

    # Remote presets
    [preset-name]
    host=user@hostname
    port=22
    dir=/path/to/remote/dir
    key=~/.ssh/id_ed25519

  Project Settings:
    uv_projects     - Space-separated subdirs to run 'uv sync' on
    path_fix        - Commands to fix paths after sync (can have multiple)

  Example .sync.conf:
    [project]
    uv_projects=API-Manager Aider.PP OpenOrchestra
    path_fix=sed -i 's|file:\\.\\./API-Manager|file://$REMOTE_DIR/API-Manager|g' $REMOTE_DIR/OpenOrchestra/pyproject.toml

    [desktop]
    host=Desktop
    dir=/home/user/code/MyProject

    [gpu-server]
    host=user@192.168.1.100
    port=22
    dir=/workspace/MyProject
    key=~/.ssh/id_rsa

Examples:
  ./sync.sh --preset desktop              # Use 'desktop' preset with live sync
  ./sync.sh --status --preset desktop     # Check if sync is running
  ./sync.sh --setup-creds --preset gpu    # Only setup credentials
  ./sync.sh -h user@server -d /opt/code   # Custom remote
  ./sync.sh --preset gpu --skip-deps      # Skip dep installation
  ./sync.sh --preset desktop --no-watch   # One-time sync only
EOF
    exit 0
}

# Lock file management for status checks
get_lock_file() {
    local preset="$1"
    echo "$LOCK_DIR/sync-${PROJECT_NAME}-${preset}.lock"
}

write_lock() {
    local preset="$1"
    local lock_file
    lock_file=$(get_lock_file "$preset")
    echo "$$" > "$lock_file"
    echo "$REMOTE_HOST:$REMOTE_DIR" >> "$lock_file"
    echo "$(date -Iseconds)" >> "$lock_file"
}

remove_lock() {
    local preset="$1"
    local lock_file
    lock_file=$(get_lock_file "$preset")
    rm -f "$lock_file"
}

check_status() {
    local preset="$1"
    local lock_file
    lock_file=$(get_lock_file "$preset")

    if [[ ! -f "$lock_file" ]]; then
        echo '{"running": false, "preset": "'"$preset"'"}'
        return 1
    fi

    local pid remote started
    pid=$(sed -n '1p' "$lock_file")
    remote=$(sed -n '2p' "$lock_file")
    started=$(sed -n '3p' "$lock_file")

    # Check if process is still running
    if kill -0 "$pid" 2>/dev/null; then
        echo '{"running": true, "preset": "'"$preset"'", "pid": '"$pid"', "remote": "'"$remote"'", "started": "'"$started"'"}'
        return 0
    else
        # Stale lock file
        rm -f "$lock_file"
        echo '{"running": false, "preset": "'"$preset"'", "stale_lock": true}'
        return 1
    fi
}

# Cleanup on exit
cleanup() {
    if [[ -n "$CURRENT_PRESET" ]]; then
        remove_lock "$CURRENT_PRESET"
    fi
}
trap cleanup EXIT

list_presets() {
    if [[ -z "$CONFIG_FILE" || ! -f "$CONFIG_FILE" ]]; then
        echo "No config file found."
        echo "Create .sync.conf in the script directory"
        exit 1
    fi

    echo "Available presets from $CONFIG_FILE:"
    echo ""
    grep '^\[' "$CONFIG_FILE" | sed 's/\[//g; s/\]//g' | while read -r preset; do
        # Skip the special [project] section
        [[ "$preset" == "project" ]] && continue
        echo "  - $preset"
    done
    exit 0
}

load_project_config() {
    [[ -z "$CONFIG_FILE" || ! -f "$CONFIG_FILE" ]] && return

    # Parse the [project] section if it exists
    local in_section=false
    while IFS= read -r line || [[ -n "$line" ]]; do
        # Skip empty lines and comments
        [[ -z "$line" || "$line" =~ ^# ]] && continue

        # Check for section header
        if [[ "$line" =~ ^\[(.+)\]$ ]]; then
            if [[ "${BASH_REMATCH[1]}" == "project" ]]; then
                in_section=true
            else
                in_section=false
            fi
            continue
        fi

        # Parse key=value pairs in project section
        if $in_section && [[ "$line" =~ ^([^=]+)=(.*)$ ]]; then
            local key="${BASH_REMATCH[1]}"
            local value="${BASH_REMATCH[2]}"

            case "$key" in
                uv_projects)
                    UV_PROJECTS="$value"
                    ;;
                path_fix)
                    PATH_FIXES+=("$value")
                    ;;
            esac
        fi
    done < "$CONFIG_FILE"
}

load_preset() {
    local preset_name="$1"
    CURRENT_PRESET="$preset_name"

    if [[ -z "$CONFIG_FILE" || ! -f "$CONFIG_FILE" ]]; then
        echo "Error: No config file found. Create .sync.conf first."
        exit 1
    fi

    # Check if preset exists
    if ! grep -q "^\[$preset_name\]" "$CONFIG_FILE"; then
        echo "Error: Preset '$preset_name' not found in $CONFIG_FILE"
        echo "Use --list-presets to see available presets."
        exit 1
    fi

    # Parse the preset section
    local in_section=false
    while IFS= read -r line || [[ -n "$line" ]]; do
        # Skip empty lines and comments
        [[ -z "$line" || "$line" =~ ^# ]] && continue

        # Check for section header
        if [[ "$line" =~ ^\[(.+)\]$ ]]; then
            if [[ "${BASH_REMATCH[1]}" == "$preset_name" ]]; then
                in_section=true
            else
                in_section=false
            fi
            continue
        fi

        # Parse key=value pairs in our section
        if $in_section && [[ "$line" =~ ^([^=]+)=(.*)$ ]]; then
            local key="${BASH_REMATCH[1]}"
            local value="${BASH_REMATCH[2]}"
            # Expand ~ in paths
            value="${value/#\~/$HOME}"

            case "$key" in
                host) REMOTE_HOST="$value" ;;
                port) REMOTE_PORT="$value" ;;
                dir)  REMOTE_DIR="$value" ;;
                key)  SSH_KEY="$value" ;;
                user) REMOTE_USER="$value" ;;
            esac
        fi
    done < "$CONFIG_FILE"

    echo "Loaded preset: $preset_name"
}

# Setup credentials on remote machine
setup_credentials() {
    echo "[*] Setting up credentials on remote..."

    # Setup WANDB
    if [[ -f "$ENV_GLOBAL_FILE" ]]; then
        echo "    - Copying environment variables..."
        local wandb_key hf_token runpod_key
        wandb_key=$(grep '^WANDB_API_KEY=' "$ENV_GLOBAL_FILE" 2>/dev/null | cut -d= -f2 | tr -d '"' || true)
        hf_token=$(grep '^HUGGINGFACE_WRITE_TOKEN=' "$ENV_GLOBAL_FILE" 2>/dev/null | cut -d= -f2 | tr -d '"' || true)
        runpod_key=$(grep '^RUNPOD_API_KEY=' "$ENV_GLOBAL_FILE" 2>/dev/null | cut -d= -f2 | tr -d '"' || true)

        if [[ -n "$wandb_key" ]]; then
            echo "    - Setting up WANDB..."
            run_ssh "mkdir -p ~/.config && grep -q 'WANDB_API_KEY' ~/.bashrc 2>/dev/null || echo 'export WANDB_API_KEY=\"$wandb_key\"' >> ~/.bashrc"
            # Also setup netrc for wandb
            run_ssh "grep -q 'api.wandb.ai' ~/.netrc 2>/dev/null || echo -e 'machine api.wandb.ai\n  login user\n  password $wandb_key' >> ~/.netrc && chmod 600 ~/.netrc"
        fi

        if [[ -n "$hf_token" ]]; then
            echo "    - Setting up HuggingFace..."
            run_ssh "grep -q 'HUGGINGFACE_TOKEN' ~/.bashrc 2>/dev/null || echo 'export HUGGINGFACE_TOKEN=\"$hf_token\"' >> ~/.bashrc"
            run_ssh "grep -q 'HF_TOKEN' ~/.bashrc 2>/dev/null || echo 'export HF_TOKEN=\"$hf_token\"' >> ~/.bashrc"
        fi

        if [[ -n "$runpod_key" ]]; then
            echo "    - Setting up RunPod..."
            run_ssh "grep -q 'RUNPOD_API_KEY' ~/.bashrc 2>/dev/null || echo 'export RUNPOD_API_KEY=\"$runpod_key\"' >> ~/.bashrc"
        fi
    fi

    # Setup GCP
    if [[ -f "$GCP_CREDS_FILE" ]]; then
        echo "    - Setting up GCP credentials..."
        run_ssh "mkdir -p ~/.config/gcloud"
        rsync -az -e "$RSYNC_RSH" "$GCP_CREDS_FILE" "$REMOTE_HOST:~/.config/gcloud/application_default_credentials.json" 2>/dev/null || true
        run_ssh "chmod 600 ~/.config/gcloud/application_default_credentials.json" 2>/dev/null || true
    fi

    # Check/install gcloud CLI
    echo "    - Checking gcloud CLI..."
    if ! run_ssh 'which gcloud' &>/dev/null; then
        echo "    - Installing gcloud CLI (this may take a minute)..."
        run_ssh 'curl -sSL https://sdk.cloud.google.com > /tmp/install_gcloud.sh && bash /tmp/install_gcloud.sh --disable-prompts --install-dir=$HOME 2>/dev/null' || true
        run_ssh 'grep -q "google-cloud-sdk" ~/.bashrc 2>/dev/null || echo '\''export PATH="$HOME/google-cloud-sdk/bin:$PATH"'\'' >> ~/.bashrc'
    fi

    echo "[+] Credentials setup complete!"
}

# Load project config at startup
load_project_config

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--host)
            REMOTE_HOST="$2"
            shift 2
            ;;
        -p|--port)
            REMOTE_PORT="$2"
            shift 2
            ;;
        -d|--dir)
            REMOTE_DIR="$2"
            shift 2
            ;;
        -i|--identity)
            SSH_KEY="$2"
            shift 2
            ;;
        -u|--user)
            REMOTE_USER="$2"
            shift 2
            ;;
        --preset)
            load_preset "$2"
            shift 2
            ;;
        --skip-deps)
            SKIP_DEPS=true
            shift
            ;;
        --no-watch)
            WATCH_MODE=false
            shift
            ;;
        --setup-creds)
            SETUP_CREDS=true
            shift
            ;;
        --status)
            STATUS_CHECK=true
            shift
            ;;
        --list-presets)
            list_presets
            ;;
        --help)
            usage
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information."
            exit 1
            ;;
    esac
done

# Handle status check early (only needs preset)
if $STATUS_CHECK; then
    if [[ -z "$CURRENT_PRESET" ]]; then
        echo "Error: --status requires --preset"
        exit 1
    fi
    check_status "$CURRENT_PRESET"
    exit $?
fi

# Validate required parameters
if [[ -z "$REMOTE_HOST" ]]; then
    echo "Error: Remote host is required."
    echo "Use --host or --preset, or run --help for usage."
    exit 1
fi

if [[ -z "$REMOTE_DIR" ]]; then
    echo "Error: Remote directory is required."
    echo "Use --dir or --preset, or run --help for usage."
    exit 1
fi

# Prepend user if specified separately
if [[ -n "$REMOTE_USER" && ! "$REMOTE_HOST" =~ @ ]]; then
    REMOTE_HOST="$REMOTE_USER@$REMOTE_HOST"
fi

# Build SSH arguments
SSH_ARGS=()
if [[ -n "$SSH_KEY" ]]; then
    SSH_ARGS+=("-i" "$SSH_KEY")
fi
if [[ -n "$REMOTE_PORT" ]]; then
    SSH_ARGS+=("-p" "$REMOTE_PORT")
fi

# Build rsync SSH command
RSYNC_RSH="ssh"
if [[ -n "$SSH_KEY" ]]; then
    RSYNC_RSH="$RSYNC_RSH -i $SSH_KEY"
fi
if [[ -n "$REMOTE_PORT" ]]; then
    RSYNC_RSH="$RSYNC_RSH -p $REMOTE_PORT"
fi

# Build exclude arguments for rsync
RSYNC_EXCLUDES=()
for exclude in "${EXCLUDES[@]}"; do
    RSYNC_EXCLUDES+=("--exclude=$exclude")
done

# Helper function to run SSH commands
run_ssh() {
    ssh "${SSH_ARGS[@]}" "$REMOTE_HOST" "$@"
}

# Helper function to run rsync
do_sync() {
    rsync -avz --no-owner --no-group -e "$RSYNC_RSH" \
        "${RSYNC_EXCLUDES[@]}" \
        "$LOCAL_DIR/" "$REMOTE_HOST:$REMOTE_DIR/" || true
}

echo "========================================="
echo "$PROJECT_NAME Sync"
echo "========================================="
echo "Local:  $LOCAL_DIR"
echo "Remote: $REMOTE_HOST:$REMOTE_DIR"
echo ""

# Check local dependencies
if ! command -v rsync &> /dev/null; then
    echo "Error: rsync is not installed locally."
    echo "Install it with: apt install rsync (or brew install rsync on macOS)"
    exit 1
fi

if $WATCH_MODE && ! command -v inotifywait &> /dev/null; then
    echo "Error: inotifywait is not installed (required for watch mode)."
    echo "Install it with: apt install inotify-tools"
    echo "Or use --no-watch for one-time sync."
    exit 1
fi

# Create remote directory if it doesn't exist
echo "[*] Ensuring remote directory exists..."
run_ssh "mkdir -p $REMOTE_DIR"

if ! $SKIP_DEPS; then
    # Ensure rsync is installed on remote
    echo "[*] Ensuring rsync is installed on remote..."
    run_ssh "which rsync > /dev/null 2>&1 || (apt-get update && apt-get install -y rsync)" 2>/dev/null || true

    # Ensure uv is installed on remote
    echo "[*] Ensuring uv is installed on remote..."
    run_ssh 'which uv > /dev/null 2>&1 || (curl -LsSf https://astral.sh/uv/install.sh | sh && echo '\''export PATH="$HOME/.local/bin:$PATH"'\'' >> ~/.bashrc)' 2>/dev/null || true

    # Ensure tmux is installed on remote
    echo "[*] Ensuring tmux is installed on remote..."
    run_ssh "which tmux > /dev/null 2>&1 || (apt-get update && apt-get install -y tmux)" 2>/dev/null || true

    # Setup credentials (WANDB, GCP, HuggingFace, etc.)
    setup_credentials
fi

# Handle --setup-creds only mode
if $SETUP_CREDS; then
    if $SKIP_DEPS; then
        # If --skip-deps was also passed, still setup creds
        setup_credentials
    fi
    echo "Done! (credentials only mode)"
    exit 0
fi

# Initial sync
echo "[*] Performing initial sync..."
do_sync
echo "[+] Initial sync complete!"
echo ""

# Apply path fixes if configured
if [[ ${#PATH_FIXES[@]} -gt 0 ]]; then
    echo "[*] Applying path fixes for remote environment..."
    for fix_cmd in "${PATH_FIXES[@]}"; do
        # Expand variables in the command
        fix_cmd_expanded=$(eval echo "$fix_cmd")
        run_ssh "$fix_cmd_expanded" 2>/dev/null || echo "    Warning: Path fix failed: $fix_cmd"
    done
    echo "[+] Path fixes applied!"
    echo ""
fi

if ! $SKIP_DEPS && [[ -n "$UV_PROJECTS" ]]; then
    # Run uv sync on all projects
    echo "[*] Running uv sync on projects..."
    for project in $UV_PROJECTS; do
        if run_ssh "test -d $REMOTE_DIR/$project" 2>/dev/null; then
            echo "    - $project"
            run_ssh "export PATH=\"\$HOME/.local/bin:\$PATH\" && cd $REMOTE_DIR/$project && uv sync" 2>/dev/null || echo "      Warning: uv sync failed for $project"
        fi
    done
    echo "[+] Dependencies installed!"
    echo ""
fi

if ! $SKIP_DEPS; then
    # Run uv sync on the remote for the main project (monorepo root)
    echo "[*] Running uv sync on remote..."
    run_ssh "export PATH=\"\$HOME/.local/bin:\$PATH\" && cd $REMOTE_DIR && uv sync --all-packages" 2>/dev/null || echo "    Warning: uv sync failed"
    echo "[+] Dependencies synced!"
    echo ""
fi

if ! $WATCH_MODE; then
    echo "Done! (watch mode disabled)"
    exit 0
fi

# Write lock file for status checks
if [[ -n "$CURRENT_PRESET" ]]; then
    write_lock "$CURRENT_PRESET"
    echo "[*] Lock file written for preset: $CURRENT_PRESET"
fi

# Watch for changes
echo "[*] Watching for changes... (Press Ctrl+C to stop)"
echo ""

inotifywait -m -r -e modify,create,delete,move \
    --exclude '(\.git|__pycache__|\.venv|venv|node_modules)' \
    "$LOCAL_DIR" 2>/dev/null |
while read -r path action file; do
    # Skip sync.conf changes
    [[ "$file" == ".sync.conf" ]] && continue

    echo "[$(date '+%H:%M:%S')] $action: $file"
    do_sync 2>/dev/null
    echo "[$(date '+%H:%M:%S')] Synced"
done
