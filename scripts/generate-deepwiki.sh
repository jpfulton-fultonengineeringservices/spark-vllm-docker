#!/bin/bash
# shellcheck shell=bash
#
# scripts/generate-deepwiki.sh
#
# Generates AI C4 architecture documentation with deepwiki-rs (Litho), backed
# by the local LiteLLM proxy. Output is committed under deepwiki-rs-docs/;
# the .litho/ cache is local-only (gitignored).
#
# Usage:
#   scripts/generate-deepwiki.sh [path] [options]
#
# Arguments:
#   path                        Subtree to analyze, relative to the repo root
#                               (default: . = whole repo)
#
# Shadow tree (default):
#   deepwiki-rs only honors gitignore for FILES in its later stages - its
#   directory walk and per-directory dossier reads see every gitignored
#   directory (bin/, build output, ...). To keep the analysis limited to
#   real source, the script materializes a shadow tree of exactly
#   `git ls-files` (hardlinks) at .litho/tree/<path> and points the tool at
#   it. Git resolves the full .gitignore cascade (root + folder-level +
#   negations + force-added files), so tracked sources inside ignored dirs
#   are kept. Built by scripts/python/shadow-tree.py; rebuilt on every run.
#   The python interpreter is selected by scripts/lib/toolchain.sh (pyenv
#   preferred over the system python3).
#
# Options:
#   --output DIR                Override output directory
#                               (default: deepwiki-rs-docs for root,
#                                deepwiki-rs-docs/<path> for a subtree)
#   --base-url URL              LLM API base URL (trailing slash optional -
#                               the tool trims it before appending
#                               /chat/completions)
#                               (default: https://dell.fulton-home.fultonengineeringservices.com/litellm
#                                env / .env: DEEPWIKI_API_BASE_URL)
#   --model-efficient MODEL     Efficient model slot
#                               (default: cluster/glm-5.3-flash;
#                                env / .env: DEEPWIKI_MODEL_EFFICIENT)
#   --model-powerful MODEL      Powerful model slot
#                               (default: cluster/glm-5.3-flash;
#                                env / .env: DEEPWIKI_MODEL_POWERFUL)
#   --max-parallels N           Max parallel LLM calls (default: 8;
#                               env / .env: DEEPWIKI_MAX_PARALLELS)
#   --context-length N          Model context window in tokens; bounds the
#                               compressor and the selection-index budget
#                               (default: 1000000 - glm-5.3-flash;
#                               env / .env: DEEPWIKI_CONTEXT_LENGTH)
#   --update-deepwiki           Install/update deepwiki-rs from the FSE fork
#                               (parallel directory-dossier build) via
#                               cargo install --git ... --force, then exit.
#                               (env: DEEPWIKI_GIT_URL / DEEPWIKI_GIT_REF;
#                                GIT_REF names a branch for cargo --branch)
#   --check-toolchain           Validate rust/rustup/python/pyenv presence and
#                               exit (delegates to scripts/lib/toolchain.sh)
#   --install-toolchain         Install missing toolchain pieces (macOS +
#                               Homebrew; rustup + pyenv), then exit
#   --no-shadow                 Analyze the working tree directly instead of
#                               the git-tracked shadow tree (not recommended)
#   --force-regenerate          Clear the LLM cache before running
#   --no-cache                  Disable the cache for this run
#   --skip-preprocessing        Skip the preprocessing stage
#   --skip-research             Skip the research stage
#   --skip-documentation        Skip final document generation
#   --disable-preset-tools      Disable the tool's preset agent tools
#   --verbose                   Verbose ReAct agent logging
#   -h, --help                  Show this help
#
# Secrets and configuration:
#   The LiteLLM key is read from $LITELLM_API_KEY, else parsed from the
#   repo-root .env file (never sourced). Copy .env.example to .env and fill
#   in the value. The key is exported as LITHO_LLM_API_KEY for the child
#   process only - it never appears on the command line.
#   .env may also set DEEPWIKI_API_BASE_URL, DEEPWIKI_MODEL_EFFICIENT,
#   DEEPWIKI_MODEL_POWERFUL, DEEPWIKI_MAX_PARALLELS, and
#   DEEPWIKI_CONTEXT_LENGTH. Precedence: CLI flag > process env > .env >
#   built-in default.
#
# Requirements:
#   - deepwiki-rs (FSE fork build; install with --update-deepwiki)
#   - mermaid-fixer (cargo install mermaid-fixer; deepwiki-rs bails at
#     startup without it)
#   - rust toolchain + python3 (validated by scripts/lib/toolchain.sh; pyenv
#     is used for python version selection, see scripts/lib/toolchain.sh)
#   - scripts/lib/common.sh and scripts/python/shadow-tree.py (both tracked)
#   - Generated markdown is normalized after generation (trailing whitespace
#     stripped, single trailing newline) so the pre-commit `whitespace` check
#     stays green on deepwiki-rs-docs/
#
# Compatible with: bash 3.2 + BSD coreutils (macOS stock /bin/bash)

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd -P)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." >/dev/null 2>&1 && pwd -P)"
LIB_DIR="${SCRIPT_DIR}/lib"
PY_SCRIPT="${SCRIPT_DIR}/python/shadow-tree.py"

# The lib scripts below are tracked in the repo; a missing lib is a broken
# checkout, so fail loudly rather than degrade to inline fallbacks.
if [ ! -f "${LIB_DIR}/common.sh" ] || [ ! -f "${LIB_DIR}/toolchain.sh" ]; then
    printf '[x] Missing lib scripts in %s (expected common.sh and toolchain.sh).\n' "${LIB_DIR}" >&2
    exit 1
fi
# shellcheck source=lib/common.sh disable=SC1091
source "${LIB_DIR}/common.sh"
# shellcheck source=lib/toolchain.sh disable=SC1091
source "${LIB_DIR}/toolchain.sh"

# ---------------------------------------------------------------------------
# Defaults (env-overridable)
# ---------------------------------------------------------------------------

# The tool trims trailing slashes before joining /chat/completions, so a
# trailing slash is safe here; the strip in "Configuration resolution" is
# kept defensively for consistency. (Starlette proxies 404 on a double-slash
# path, so the trim is belt-and-suspenders.)
DEFAULT_BASE_URL="https://dell.fulton-home.fultonengineeringservices.com/litellm"
DEFAULT_MODEL="cluster/glm-5.3-flash"
DEFAULT_MAX_PARALLELS=8
# glm-5.3-flash (cluster and openrouter) supports a 1M-token context window.
DEFAULT_CONTEXT_LENGTH=1000000

TARGET_PATH="."
OUTPUT_DIR=""
# Process env captured here; .env and built-in defaults are applied after
# REPO_ROOT is known (see "Configuration resolution" below).
BASE_URL="${DEEPWIKI_API_BASE_URL:-}"
MODEL_EFFICIENT="${DEEPWIKI_MODEL_EFFICIENT:-}"
MODEL_POWERFUL="${DEEPWIKI_MODEL_POWERFUL:-}"
MAX_PARALLELS="${DEEPWIKI_MAX_PARALLELS:-}"
CONTEXT_LENGTH="${DEEPWIKI_CONTEXT_LENGTH:-}"
USE_SHADOW=true
SHADOW_ROOT=".litho/tree"
# deepwiki-rs is installed from the FSE fork (carries the parallel
# directory-dossier patch), not from crates.io.
DEEPWIKI_GIT_URL="${DEEPWIKI_GIT_URL:-https://github.com/Fulton-Engineering-Services/deepwiki-rs.git}"
DEEPWIKI_GIT_REF="${DEEPWIKI_GIT_REF:-feat/parallel-directory-dossiers}"
UPDATE_DEEPWIKI=false
CHECK_TOOLCHAIN=false
INSTALL_TOOLCHAIN=false
EXTRA_ARGS=()
POSITIONAL_SEEN=false

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

while [ $# -gt 0 ]; do
    case $1 in
        --output)
            [ $# -ge 2 ] || { log_error "--output requires a value"; exit 1; }
            OUTPUT_DIR="$2"; shift 2 ;;
        --base-url)
            [ $# -ge 2 ] || { log_error "--base-url requires a value"; exit 1; }
            BASE_URL="$2"; shift 2 ;;
        --model-efficient)
            [ $# -ge 2 ] || { log_error "--model-efficient requires a value"; exit 1; }
            MODEL_EFFICIENT="$2"; shift 2 ;;
        --model-powerful)
            [ $# -ge 2 ] || { log_error "--model-powerful requires a value"; exit 1; }
            MODEL_POWERFUL="$2"; shift 2 ;;
        --max-parallels)
            [ $# -ge 2 ] || { log_error "--max-parallels requires a value"; exit 1; }
            MAX_PARALLELS="$2"; shift 2 ;;
        --context-length)
            [ $# -ge 2 ] || { log_error "--context-length requires a value"; exit 1; }
            CONTEXT_LENGTH="$2"; shift 2 ;;
        --no-shadow)            USE_SHADOW=false;                       shift ;;
        --update-deepwiki)      UPDATE_DEEPWIKI=true;                   shift ;;
        --check-toolchain)      CHECK_TOOLCHAIN=true;                   shift ;;
        --install-toolchain)    INSTALL_TOOLCHAIN=true;                 shift ;;
        --force-regenerate)     EXTRA_ARGS+=("--force-regenerate");     shift ;;
        --no-cache)             EXTRA_ARGS+=("--no-cache");             shift ;;
        --skip-preprocessing)   EXTRA_ARGS+=("--skip-preprocessing");   shift ;;
        --skip-research)        EXTRA_ARGS+=("--skip-research");        shift ;;
        --skip-documentation)   EXTRA_ARGS+=("--skip-documentation");   shift ;;
        --disable-preset-tools) EXTRA_ARGS+=("--disable-preset-tools"); shift ;;
        --verbose)              EXTRA_ARGS+=("--verbose");              shift ;;
        -h|--help)
            sed -n '/^# Usage:/,/^# Compatible with:/p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
            exit 0
            ;;
        -*)
            log_error "Unknown option: $1"
            printf '  Run with --help for usage.\n' >&2
            exit 1
            ;;
        *)
            if [ "${POSITIONAL_SEEN}" = "true" ]; then
                log_error "Unexpected extra argument: $1"
                printf '  Run with --help for usage.\n' >&2
                exit 1
            fi
            TARGET_PATH="$1"
            POSITIONAL_SEEN=true
            shift
            ;;
    esac
done

# ---------------------------------------------------------------------------
# Toolchain-only actions (exit before any config resolution / API key work)
# ---------------------------------------------------------------------------

if [ "${CHECK_TOOLCHAIN}" = "true" ]; then
    toolchain_check
    exit $?
fi

if [ "${INSTALL_TOOLCHAIN}" = "true" ]; then
    toolchain_install
    exit 0
fi

# ---------------------------------------------------------------------------
# Validate the target path
# ---------------------------------------------------------------------------

# Normalize: strip leading ./ and trailing /
TARGET_PATH="${TARGET_PATH#./}"
TARGET_PATH="${TARGET_PATH%/}"
[ -z "${TARGET_PATH}" ] && TARGET_PATH="."

if [ ! -d "${REPO_ROOT}/${TARGET_PATH}" ]; then
    log_error "Path not found: ${TARGET_PATH} (resolved against ${REPO_ROOT})"
    exit 1
fi

RESOLVED_PATH="$(cd -- "${REPO_ROOT}/${TARGET_PATH}" >/dev/null 2>&1 && pwd -P)"
case "${RESOLVED_PATH}" in
    "${REPO_ROOT}"|"${REPO_ROOT}"/*) ;;
    *)
        log_error "Path escapes the repo root: ${TARGET_PATH}"
        exit 1
        ;;
esac

# ---------------------------------------------------------------------------
# Configuration resolution: CLI flag > process env > .env > built-in default.
# CLI flags were applied during argument parsing above; fill anything still
# unset from .env, then from defaults.
# ---------------------------------------------------------------------------

# Read a KEY from the repo-root .env file (grep/cut - never `source` it).
# Prints the value (possibly empty) with optional surrounding quotes stripped.
# The `|| true` guards set -e/pipefail when grep finds no match.
read_env_file_value() {
    local key="$1"
    local env_file="${REPO_ROOT}/.env"
    [ -f "${env_file}" ] || return 0
    local value
    value="$(grep -E "^${key}=" "${env_file}" 2>/dev/null | head -1 | cut -d'=' -f2- || true)"
    value="${value%\"}"
    value="${value#\"}"
    value="${value%\'}"
    value="${value#\'}"
    printf '%s' "${value}"
}

if [ -z "${BASE_URL}" ]; then BASE_URL="$(read_env_file_value DEEPWIKI_API_BASE_URL)"; fi
if [ -z "${MODEL_EFFICIENT}" ]; then MODEL_EFFICIENT="$(read_env_file_value DEEPWIKI_MODEL_EFFICIENT)"; fi
if [ -z "${MODEL_POWERFUL}" ]; then MODEL_POWERFUL="$(read_env_file_value DEEPWIKI_MODEL_POWERFUL)"; fi
if [ -z "${MAX_PARALLELS}" ]; then MAX_PARALLELS="$(read_env_file_value DEEPWIKI_MAX_PARALLELS)"; fi
if [ -z "${CONTEXT_LENGTH}" ]; then CONTEXT_LENGTH="$(read_env_file_value DEEPWIKI_CONTEXT_LENGTH)"; fi
if [ -z "${BASE_URL}" ]; then BASE_URL="${DEFAULT_BASE_URL}"; fi
if [ -z "${MODEL_EFFICIENT}" ]; then MODEL_EFFICIENT="${DEFAULT_MODEL}"; fi
if [ -z "${MODEL_POWERFUL}" ]; then MODEL_POWERFUL="${DEFAULT_MODEL}"; fi
if [ -z "${MAX_PARALLELS}" ]; then MAX_PARALLELS="${DEFAULT_MAX_PARALLELS}"; fi
if [ -z "${CONTEXT_LENGTH}" ]; then CONTEXT_LENGTH="${DEFAULT_CONTEXT_LENGTH}"; fi

# Validate: positive integers
case "${MAX_PARALLELS}" in
    ''|*[!0-9]*|0)
        log_error "Invalid max-parallels value: '${MAX_PARALLELS}' (expected a positive integer)"
        exit 1
        ;;
esac
case "${CONTEXT_LENGTH}" in
    ''|*[!0-9]*|0)
        log_error "Invalid context-length value: '${CONTEXT_LENGTH}' (expected a positive integer)"
        exit 1
        ;;
esac

# Strip any trailing slash from the base URL - see the DEFAULT_BASE_URL note
BASE_URL="${BASE_URL%/}"

# ---------------------------------------------------------------------------
# Derive output dir, project name, cache location
# ---------------------------------------------------------------------------
# NOTE on cache: deepwiki-rs resolves its cache as .litho under the CWD, not
# under the target subtree (we cd to the repo root before exec). Keys are
# path-namespaced inside it (e.g. cache/directory_scoring_scripts/), so one
# root cache serves all subtree runs.

if [ "${TARGET_PATH}" = "." ]; then
    PROJECT_NAME="codekeeper"
    [ -z "${OUTPUT_DIR}" ] && OUTPUT_DIR="deepwiki-rs-docs"
else
    PROJECT_NAME="codekeeper/${TARGET_PATH}"
    [ -z "${OUTPUT_DIR}" ] && OUTPUT_DIR="deepwiki-rs-docs/${TARGET_PATH}"
fi
CACHE_DIR=".litho"

# ---------------------------------------------------------------------------
# Resolve the LiteLLM API key: process env first, else the repo-root .env.
# ---------------------------------------------------------------------------

API_KEY="${LITELLM_API_KEY:-}"

if [ -z "${API_KEY}" ]; then
    API_KEY="$(read_env_file_value LITELLM_API_KEY)"
fi

if [ -z "${API_KEY}" ] || [ "${API_KEY}" = "REPLACE_THIS_VALUE" ]; then
    log_error "LITELLM_API_KEY is not set and ${REPO_ROOT}/.env does not provide it."
    printf '       Remediation: copy .env.example to .env and fill in the value:\n' >&2
    printf '         cp .env.example .env && $EDITOR .env\n' >&2
    exit 1
fi

# ---------------------------------------------------------------------------
# Preconditions and summary
# ---------------------------------------------------------------------------

# Prepend the cargo bin dir to PATH for this process and its children.
# deepwiki-rs spawns mermaid-fixer as a bare command (PATH lookup), and
# cargo installs both binaries here; it is not always on the user's PATH.
CARGO_BIN_DIR="${CARGO_HOME:-${HOME}/.cargo}/bin"
if [ -d "${CARGO_BIN_DIR}" ]; then
    case ":${PATH}:" in
        *":${CARGO_BIN_DIR}:"*) ;;
        *) PATH="${CARGO_BIN_DIR}:${PATH}"; export PATH ;;
    esac
fi

# --update-deepwiki: install/refresh the fork build and exit. Handled before
# all other preconditions so it works on machines with no deepwiki-rs yet.
if [ "${UPDATE_DEEPWIKI}" = "true" ]; then
    toolchain_check_rust || die "A rust toolchain is required to build deepwiki-rs. Install: scripts/generate-deepwiki.sh --install-toolchain"
    log_step "Installing deepwiki-rs from ${DEEPWIKI_GIT_URL} (${DEEPWIKI_GIT_REF})..."
    log_info "Compiles from source; this can take several minutes."
    # Use the system git CLI for the fetch (honors gh's credential helper);
    # cargo's built-in libgit2 fetch can 404/auth-fail otherwise.
    CARGO_NET_GIT_FETCH_WITH_CLI=true \
        cargo install --git "${DEEPWIKI_GIT_URL}" --branch "${DEEPWIKI_GIT_REF}" --force deepwiki-rs
    log_success "deepwiki-rs updated: $(deepwiki-rs --version 2>/dev/null | head -1)"
    exit 0
fi

if ! command -v deepwiki-rs >/dev/null 2>&1; then
    log_error "Missing required command: deepwiki-rs"
    printf '       Install: scripts/generate-deepwiki.sh --update-deepwiki\n' >&2
    exit 1
fi

# deepwiki-rs bails at startup unless mermaid-fixer is on PATH - fail fast
# with an actionable message instead of the tool's bare error.
if ! command -v mermaid-fixer >/dev/null 2>&1; then
    log_error "Missing required command: mermaid-fixer"
    printf '       deepwiki-rs requires it at startup to validate Mermaid diagrams.\n' >&2
    printf '       Install: cargo install mermaid-fixer\n' >&2
    exit 1
fi

# deepwiki-rs discovers ./litho.toml from the CWD and resolves -p/-o relative
# to it, so everything below runs from the repo root.
cd "${REPO_ROOT}"

# ---------------------------------------------------------------------------
# Shadow tree: materialize exactly the git-tracked files of the target as
# hardlinks and point the tool at it. See the header for rationale. The
# python interpreter comes from scripts/lib/toolchain.sh (pyenv preferred).
# ---------------------------------------------------------------------------

EFFECTIVE_TARGET="${TARGET_PATH}"
SHADOW_COUNT=""

if [ "${USE_SHADOW}" = "true" ]; then
    if [ "${TARGET_PATH}" = "." ]; then
        SHADOW_DIR="${SHADOW_ROOT}/repo"
    else
        SHADOW_DIR="${SHADOW_ROOT}/${TARGET_PATH}"
    fi

    PY_BIN="$(toolchain_python_bin)" || die "No usable python3 found. Install: scripts/generate-deepwiki.sh --install-toolchain"
    [ -f "${PY_SCRIPT}" ] || die "Missing shadow-tree builder: ${PY_SCRIPT}"

    log_step "Building git-tracked shadow tree at ${SHADOW_DIR}..."
    # NOTE: rebuilding also drops the tool's internal_path cache
    # (<project_path>/.litho, i.e. <shadow>/.litho for a shadow run).
    # Currently inert because litho.toml omits the [knowledge] section; if
    # local-docs sync is ever enabled its cache would be wiped every run.
    rm -rf "${SHADOW_DIR}"
    mkdir -p "${SHADOW_DIR}"

    SHADOW_COUNT="$(REPO_ROOT="${REPO_ROOT}" TARGET="${TARGET_PATH}" SHADOW="${SHADOW_DIR}" "${PY_BIN}" "${PY_SCRIPT}")"

    if [ -z "${SHADOW_COUNT}" ] || [ "${SHADOW_COUNT}" -eq 0 ] 2>/dev/null; then
        log_error "No git-tracked files found under '${TARGET_PATH}' - nothing to analyze."
        exit 1
    fi

    log_success "Shadow tree: ${SHADOW_COUNT} tracked file(s) linked"
    EFFECTIVE_TARGET="${SHADOW_DIR}"
fi

log_header "deepwiki-rs architecture docs"
log_config "Target" "${TARGET_PATH}"
if [ "${USE_SHADOW}" = "true" ]; then
    log_config "Mode" "shadow (git-tracked files only)"
    log_config "Shadow" "${EFFECTIVE_TARGET}"
else
    log_config "Mode" "direct (working tree - includes gitignored content)"
fi
log_config "Project name" "${PROJECT_NAME}"
log_config "Output" "${OUTPUT_DIR}"
log_config "Base URL" "${BASE_URL}"
log_config "Model (efficient)" "${MODEL_EFFICIENT}"
log_config "Model (powerful)" "${MODEL_POWERFUL}"
log_config "Max parallels" "${MAX_PARALLELS}"
log_config "Context length" "${CONTEXT_LENGTH}"
log_config "Cache" "${CACHE_DIR}"
printf '\n'

# The key is exported for the child process only and never appears on the
# command line.
export LITHO_LLM_API_KEY="${API_KEY}"

# Run the tool without exec'ing so the output can be normalized below while
# preserving the tool's exit code.
set +e
deepwiki-rs \
    -p "${EFFECTIVE_TARGET}" \
    -o "${OUTPUT_DIR}" \
    --name "${PROJECT_NAME}" \
    --llm-context-length "${CONTEXT_LENGTH}" \
    --model-efficient "${MODEL_EFFICIENT}" \
    --model-powerful "${MODEL_POWERFUL}" \
    --max-parallels "${MAX_PARALLELS}" \
    --llm-api-base-url "${BASE_URL}" \
    ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}
TOOL_EXIT=$?
set -e

# Normalize the generated markdown: deepwiki-rs emits trailing whitespace and
# a blank line at EOF, which the pre-commit `whitespace` job rejects
# (`git diff --cached --check`). The docs are prettier-ignored, so this is the
# only whitespace gate they pass through. perl ships with the macOS CLT.
# NOTE: bash 3.2 `read` lacks -d, so use find -exec instead of a NUL loop.
if [ -d "${OUTPUT_DIR}" ]; then
    log_step "Normalizing generated markdown (trailing whitespace / EOF newline)..."
    find "${OUTPUT_DIR}" -name '*.md' -type f \
        -exec perl -0pi -e 's/[ \t]+$//mg; s/\n+\z/\n/' {} +
fi

exit "${TOOL_EXIT}"