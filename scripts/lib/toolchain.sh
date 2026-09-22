#!/bin/bash
# shellcheck shell=bash
#
# scripts/lib/toolchain.sh - Validates and installs the supporting toolchains
# for the codekeeper scripts:
#
#   rust / cargo / rustup   - required to build deepwiki-rs from the FSE fork
#   python3 + pyenv         - required for the shadow-tree builder; pyenv is
#                             used for python version selection/install
#
# Source this file from other scripts:
#   LIB_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../lib" >/dev/null 2>&1 && pwd -P)"
#   source "${LIB_DIR}/common.sh"
#   source "${LIB_DIR}/toolchain.sh"
#
# Or run standalone:
#   scripts/lib/toolchain.sh check          # validate presence, exit non-zero if incomplete
#   scripts/lib/toolchain.sh install        # install what's missing (macOS + Homebrew)
#   scripts/lib/toolchain.sh python-bin     # print the chosen python3 interpreter path
#
# The install path assumes macOS with Homebrew (see toolchain_install).
#
# Compatible with: bash 3.2 + BSD coreutils (macOS stock /bin/bash)

# Guard against double-sourcing
if [ -n "${_LOG_TOOLCHAIN_LOADED:-}" ]; then
    return 0
fi
_LOG_TOOLCHAIN_LOADED=1

# Source common.sh (log_*) when not already loaded (e.g. standalone runs).
if [ "${_LOG_COMMON_LOADED:-}" != "1" ]; then
    # shellcheck source=common.sh disable=SC1091
    source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd -P)/common.sh"
fi

# pyenv version slot used by toolchain_install_python (latest patch of this
# minor is installed when the exact version is absent).
TOOLCHAIN_PYTHON_VERSION="${TOOLCHAIN_PYTHON_VERSION:-3.12}"

# ---------------------------------------------------------------------------
# Python interpreter selection
# ---------------------------------------------------------------------------

# toolchain_python_bin
#   Prints a usable python3 interpreter path, preferring a pyenv-managed
#   python over the system python3. Returns 1 (prints nothing) if neither
#   exists. Suppress stderr from pyenv probing.
toolchain_python_bin() {
    local p
    if command -v pyenv >/dev/null 2>&1; then
        p="$(pyenv which python3 2>/dev/null)" || true
        if [ -n "${p}" ]; then
            printf '%s' "${p}"
            return 0
        fi
    fi
    if command -v python3 >/dev/null 2>&1; then
        printf '%s' "$(command -v python3)"
        return 0
    fi
    return 1
}

# ---------------------------------------------------------------------------
# Rust binary resolution
# ---------------------------------------------------------------------------

# toolchain_cargo_bin_dir
#   The cargo bin dir rustup installs into (~/.cargo/bin, honoring
#   CARGO_HOME). rustup-init runs with --no-modify-path from this lib, so
#   the tools land here even when they are not on the caller's PATH.
toolchain_cargo_bin_dir() {
    printf '%s' "${CARGO_HOME:-${HOME}/.cargo}/bin"
}

# toolchain_find_rust_bin <name>   (cargo | rustc | rustup | rustup-init)
#   Prints the first reachable path to <name>: the caller's PATH first, then
#   the cargo bin dir. Returns 1 (prints nothing) if neither yields it.
toolchain_find_rust_bin() {
    local name="$1" dir=""
    if command -v "${name}" >/dev/null 2>&1; then
        command -v "${name}"
        return 0
    fi
    dir="$(toolchain_cargo_bin_dir)"
    if [ -x "${dir}/${name}" ]; then
        printf '%s' "${dir}/${name}"
        return 0
    fi
    return 1
}

# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

# toolchain_check_rust
#   Validates cargo/rustc presence (PATH or ~/.cargo/bin); rustup is
#   recommended but not required. Returns 0 when a rust build is possible,
#   1 otherwise.
toolchain_check_rust() {
    local ok=true
    local cargo_bin="" rustc_bin=""
    cargo_bin="$(toolchain_find_rust_bin cargo 2>/dev/null)" || true
    rustc_bin="$(toolchain_find_rust_bin rustc 2>/dev/null)" || true
    if [ -n "${cargo_bin}" ] && [ -n "${rustc_bin}" ]; then
        log_info "  rust:     present (${cargo_bin})"
    else
        log_warn "  rust:     MISSING (cargo + rustc required)"
        ok=false
    fi
    if toolchain_find_rust_bin rustup >/dev/null 2>&1 \
            || toolchain_find_rust_bin rustup-init >/dev/null 2>&1; then
        log_info "  rustup:   present"
    else
        log_warn "  rustup:   missing (recommended for rust version management)"
    fi
    [ "${ok}" = "true" ]
}

# toolchain_check_python
#   Validates python3 presence and pyenv availability. Returns 0 when a
#   python3 interpreter exists (for the shadow-tree builder), 1 otherwise.
toolchain_check_python() {
    local ok=true
    local p="" pyenv_name=""
    p="$(toolchain_python_bin 2>/dev/null)" || true
    if [ -n "${p}" ]; then
        log_info "  python3:  present (${p} - $("${p}" --version 2>&1))"
    else
        log_warn "  python3:  MISSING (required for the shadow tree)"
        ok=false
    fi
    if command -v pyenv >/dev/null 2>&1; then
        pyenv_name="$(pyenv version-name 2>/dev/null)" || true
        [ -n "${pyenv_name}" ] || pyenv_name="no version set"
        log_info "  pyenv:    present (${pyenv_name})"
    else
        log_warn "  pyenv:    missing (recommended for python version selection/install)"
    fi
    [ "${ok}" = "true" ]
}

# toolchain_check
#   Full toolchain validation. Returns 0 when the critical pieces (rust +
#   python) are present, 1 otherwise.
toolchain_check() {
    log_header "Supporting toolchain check"
    local ok=true
    toolchain_check_rust || ok=false
    toolchain_check_python || ok=false
    if [ "${ok}" = "true" ]; then
        log_success "Toolchain ready."
        return 0
    fi
    log_error "Toolchain incomplete. Install path (macOS + Homebrew):"
    printf '       scripts/lib/toolchain.sh install\n' >&2
    return 1
}

# ---------------------------------------------------------------------------
# Installation (macOS + Homebrew)
# ---------------------------------------------------------------------------

toolchain_is_macos() {
    [ "$(uname -s)" = "Darwin" ]
}

# toolchain_install_rust
#   Installs rustup via Homebrew and, when cargo/rustc are absent, a stable
#   rust toolchain via rustup-init. Uses --no-modify-path so the caller
#   controls PATH (scripts/generate-deepwiki.sh prepends ~/.cargo/bin); the
#   check/install functions resolve rust bins against that dir too.
toolchain_install_rust() {
    log_step "Ensuring rust toolchain (cargo/rustc via rustup)..."
    if ! toolchain_find_rust_bin rustup >/dev/null 2>&1 \
            && ! toolchain_find_rust_bin rustup-init >/dev/null 2>&1; then
        log_info "Installing rustup via Homebrew (brew install rustup)..."
        brew install rustup || die "brew install rustup failed"
    fi
    if ! toolchain_find_rust_bin cargo >/dev/null 2>&1 \
            || ! toolchain_find_rust_bin rustc >/dev/null 2>&1; then
        log_info "Installing stable toolchain via rustup..."
        if toolchain_find_rust_bin rustup-init >/dev/null 2>&1; then
            "$(toolchain_find_rust_bin rustup-init)" \
                --profile minimal --default-toolchain stable --no-modify-path -y \
                || die "rustup-init failed to install the stable toolchain"
        else
            # Standalone rustup (no rustup-init): the flags above are
            # rustup-init-only, so drive rustup's own install+default.
            toolchain_find_rust_bin rustup >/dev/null 2>&1 \
                || die "neither rustup-init nor rustup found"
            "$(toolchain_find_rust_bin rustup)" default stable \
                || die "rustup failed to install the stable toolchain"
        fi
    fi
    log_success "rust: $(toolchain_find_rust_bin cargo 2>/dev/null) / $(toolchain_find_rust_bin rustc 2>/dev/null)"
}

# toolchain_install_python
#   Installs pyenv via Homebrew, then installs TOOLCHAIN_PYTHON_VERSION (used
#   for python version selection). Avoids clobbering an existing pyenv global:
#   it only takes over the global when none is set; otherwise it reports how
#   to select the version per-project (pyenv local).
toolchain_install_python() {
    log_step "Ensuring python3 (via pyenv, ${TOOLCHAIN_PYTHON_VERSION})..."
    if ! command -v pyenv >/dev/null 2>&1; then
        log_info "Installing pyenv via Homebrew (brew install pyenv)..."
        brew install pyenv || die "brew install pyenv failed"
    fi
    if ! pyenv versions --bare 2>/dev/null | grep -E "^${TOOLCHAIN_PYTHON_VERSION}(\\.|$)" >/dev/null; then
        log_info "Installing python ${TOOLCHAIN_PYTHON_VERSION} via pyenv (this can take a while)..."
        pyenv install "${TOOLCHAIN_PYTHON_VERSION}" || die "pyenv install ${TOOLCHAIN_PYTHON_VERSION} failed"
    fi
    local current=""
    current="$(pyenv global 2>/dev/null)" || true
    if [ -z "${current}" ] || [ "${current}" = "system" ]; then
        pyenv global "${TOOLCHAIN_PYTHON_VERSION}" 2>/dev/null || true
        log_info "pyenv global set to ${TOOLCHAIN_PYTHON_VERSION}."
    else
        log_info "pyenv global is already '${current}' - leaving it. To select ${TOOLCHAIN_PYTHON_VERSION} per-project: pyenv local ${TOOLCHAIN_PYTHON_VERSION}"
    fi
    log_success "python: $(toolchain_python_bin)"
}

# toolchain_install
#   Full install path: rust (cargo/rustc) + python (pyenv). macOS + Homebrew.
toolchain_install() {
    log_header "Supporting toolchain install"
    if ! toolchain_is_macos; then
        die "toolchain_install targets macOS + Homebrew (current system: $(uname -s))"
    fi
    require_command brew "Install Homebrew first: https://brew.sh"
    toolchain_install_rust
    toolchain_install_python
    log_success "Toolchain installed."
}

# ---------------------------------------------------------------------------
# Standalone entrypoint (only when executed directly)
# ---------------------------------------------------------------------------

if [ -n "${BASH_SOURCE[0]:-}" ] && [ "${BASH_SOURCE[0]}" = "$0" ]; then
    case "${1:-check}" in
        check)      toolchain_check ;;
        install)    toolchain_install ;;
        python-bin) toolchain_python_bin ;;
        *)
            printf 'Usage: %s {check|install|python-bin}\n' "$0" >&2
            exit 1
            ;;
    esac
fi