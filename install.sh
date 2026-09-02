#!/usr/bin/env bash
#
# install.sh - install the rlm-mcp MCP server from GitHub.
#
# rlm-mcp is distributed EXCLUSIVELY through its git repository
#   https://github.com/ProCleiton/rlm-mcp
# There is no PyPI package; this script never installs from PyPI and never
# uses `uvx rlm-mcp`.
#
# Installer preference: uv, then pipx, then pip. No sudo is ever used.
#
#   ./install.sh                        install the default branch (main)
#   ./install.sh --branch <ref>         install a specific git ref
#   ./install.sh --dev                  editable install from this checkout
#   ./install.sh --force                reinstall even if already installed
#   ./install.sh -h | --help            show this help

set -euo pipefail

GIT_BASE="git+https://github.com/ProCleiton/rlm-mcp"
BRANCH=""
DEV=0
FORCE=0
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

usage() {
    cat <<'EOF'
Usage: ./install.sh [options]

Install the rlm-mcp MCP server from GitHub:

    git+https://github.com/ProCleiton/rlm-mcp

Distribution is GitHub-only: nothing is published to PyPI, so this script
never installs from PyPI.

Options:
    --branch <ref>   Install the given git ref instead of the default branch
                     (main). Appended to the URL as "@<ref>", e.g.
                     --branch feat/rlm-over-mcp-core to try a feature branch
                     before it is merged.
    --dev            Editable install from the current local checkout
                     (uv sync --extra dev with uv, or
                     python3 -m pip install --user -e '.[dev]' otherwise).
                     Requires a git clone of the repository.
    --force          Reinstall even when rlm-mcp is already installed.
    -h, --help       Show this help and exit.

Tool manager preference: uv, then pipx, then pip. No sudo is used.
EOF
}

die() {
    echo "ERROR: $*" >&2
    exit 1
}

have() {
    command -v "$1" >/dev/null 2>&1
}

parse_args() {
    while [ $# -gt 0 ]; do
        case "$1" in
            -h | --help)
                usage
                exit 0
                ;;
            --branch)
                [ $# -ge 2 ] || die "--branch requires a value (e.g. --branch feat/rlm-over-mcp-core)"
                BRANCH="$2"
                shift 2
                ;;
            --branch=*)
                BRANCH="${1#--branch=}"
                [ -n "$BRANCH" ] || die "--branch requires a non-empty value"
                shift
                ;;
            --dev)
                DEV=1
                shift
                ;;
            --force)
                FORCE=1
                shift
                ;;
            *)
                die "unknown option '$1' (run ./install.sh --help)"
                ;;
        esac
    done
}

detect_manager() {
    if have uv; then
        printf 'uv\n'
    elif have pipx; then
        printf 'pipx\n'
    elif have python3 && python3 -m pip --version >/dev/null 2>&1; then
        printf 'pip\n'
    else
        printf 'none\n'
    fi
}

hint_install_uv() {
    cat >&2 <<'EOF'
No supported installer was found (uv, pipx, or python3 + pip).

Install uv, the recommended option:

    curl -LsSf https://astral.sh/uv/install.sh | sh

Then re-run this script. See https://docs.astral.sh/uv/ for alternatives.
EOF
}

pip_failure_hint() {
    cat >&2 <<'EOF'
ERROR: pip install failed. On Debian/Ubuntu 24.04+ the system Python is
externally managed (PEP 668), so pip refuses to install outside a virtual
environment. Prefer uv (https://docs.astral.sh/uv/) or pipx; if you accept
the risk, append --break-system-packages to the pip command above.
EOF
}

premerge_hint() {
    cat >&2 <<'EOF'
HINT: the install from the default branch (main) failed. This usually means
the rlm-mcp package has not been merged into the default branch yet. If you
are installing before that merge, re-run with the feature branch:

    ./install.sh --branch feat/rlm-over-mcp-core
EOF
}

install_failed() {
    # Targeted hint for the most likely cause when installing from the
    # default branch: the package still lives on a feature branch.
    if [ -z "$BRANCH" ]; then
        premerge_hint
    fi
    exit 1
}

install_package() {
    local git_url="$1"
    local manager
    manager="$(detect_manager)"
    case "$manager" in
        uv)
            echo "==> Installing with uv: uv tool install --from $git_url rlm-mcp"
            if [ "$FORCE" -eq 1 ]; then
                if ! uv tool install --force --from "$git_url" rlm-mcp; then
                    install_failed
                fi
            else
                if ! uv tool install --from "$git_url" rlm-mcp; then
                    install_failed
                fi
            fi
            ;;
        pipx)
            echo "==> Installing with pipx: pipx install $git_url"
            if [ "$FORCE" -eq 1 ]; then
                if ! pipx install --force "$git_url"; then
                    install_failed
                fi
            else
                if ! pipx install "$git_url"; then
                    install_failed
                fi
            fi
            ;;
        pip)
            echo "==> Installing with pip: python3 -m pip install --user $git_url"
            if [ "$FORCE" -eq 1 ]; then
                if ! python3 -m pip install --user --upgrade --force-reinstall "$git_url"; then
                    pip_failure_hint
                    install_failed
                fi
            else
                if ! python3 -m pip install --user "$git_url"; then
                    pip_failure_hint
                    install_failed
                fi
            fi
            ;;
        *)
            hint_install_uv
            exit 1
            ;;
    esac
}

install_dev() {
    [ -f "$SCRIPT_DIR/pyproject.toml" ] \
        || die "--dev requires a local checkout: run 'git clone https://github.com/ProCleiton/rlm-mcp && cd rlm-mcp' first"

    if have uv; then
        echo "==> uv sync --extra dev (editable install into $SCRIPT_DIR/.venv)"
        (cd "$SCRIPT_DIR" && uv sync --extra dev)
    elif have python3 && python3 -m pip --version >/dev/null 2>&1; then
        echo "==> python3 -m pip install --user -e '.[dev]'"
        if ! (cd "$SCRIPT_DIR" && python3 -m pip install --user -e '.[dev]'); then
            pip_failure_hint
            exit 1
        fi
    else
        hint_install_uv
        exit 1
    fi
}

find_binary() {
    if [ -x "$SCRIPT_DIR/.venv/bin/rlm-mcp" ]; then
        printf '%s\n' "$SCRIPT_DIR/.venv/bin/rlm-mcp"
        return 0
    fi
    if command -v rlm-mcp >/dev/null 2>&1; then
        command -v rlm-mcp
        return 0
    fi
    local cand
    for cand in "$HOME/.local/bin/rlm-mcp" "$HOME/bin/rlm-mcp"; do
        if [ -x "$cand" ]; then
            printf '%s\n' "$cand"
            return 0
        fi
    done
    return 1
}

print_next_steps() {
    local bin="$1"
    local version="$2"
    cat <<EOF

Installation finished.
    Binary : $bin
    Version: $version

Wire the server into your harness (it runs as a stdio child process and
never needs an API key). The blocks below already use the installed binary.
If that path is not on your harness's PATH, keep the absolute path as shown;
JSON config files do not expand '~'.

1) oh-my-pi / omp - add this entry under "mcpServers" in ~/.omp/agent/mcp.json
   (merge it into the existing "mcpServers" object if the file has one):

{
  "mcpServers": {
    "rlm-mcp": {
      "type": "stdio",
      "command": "$bin",
      "args": []
    }
  }
}

2) Claude Code - .mcp.json in the project root, or: claude mcp add rlm-mcp -- $bin

{
  "mcpServers": {
    "rlm-mcp": {
      "command": "$bin",
      "args": []
    }
  }
}

3) Cursor - .cursor/mcp.json:

{
  "mcpServers": {
    "rlm-mcp": {
      "command": "$bin",
      "args": []
    }
  }
}

4) VS Code / Copilot - .vscode/mcp.json (note the "servers" key):

{
  "servers": {
    "rlm-mcp": {
      "command": "$bin",
      "args": []
    }
  }
}

Optional server flags go in "args", e.g. ["--max-depth", "3", "--log-level",
"DEBUG"]. Logs go to stderr; stdout is the MCP channel. Restart the client
after adding the entry.

See docs/install-and-usage.md for the full guide (install, wiring and the
harness-side RLM loop), and docs/DESIGN.md for the protocol design.
EOF
}

main() {
    parse_args "$@"

    echo "rlm-mcp installer"
    echo "  source: $GIT_BASE (GitHub-only distribution; no PyPI)"
    echo

    if [ "$DEV" -eq 1 ]; then
        echo "Mode: --dev editable install from $SCRIPT_DIR"
        echo
        install_dev
    else
        local git_url="$GIT_BASE"
        if [ -n "$BRANCH" ]; then
            git_url="${GIT_BASE}@${BRANCH}"
        else
            echo "Mode: default branch (main) of $GIT_BASE"
            echo
        fi
        install_package "$git_url"
    fi
    echo

    echo "==> Locating the installed 'rlm-mcp' binary ..."
    local bin
    bin="$(find_binary)" || {
        echo "ERROR: could not locate the installed 'rlm-mcp' binary." >&2
        echo "       If uv/pipx installed it outside your PATH, add $HOME/.local/bin to PATH and re-run." >&2
        exit 1
    }

    echo "==> Verifying: $bin --version"
    local version
    version="$("$bin" --version 2>&1)" || {
        echo "ERROR: '$bin --version' failed after install." >&2
        exit 1
    }
    echo "    OK: $version"

    print_next_steps "$bin" "$version"
}

main "$@"
