# scripts/ - repo tooling

Bash 3.2-safe helpers for local repo operations (macOS stock `/bin/bash` +
BSD coreutils).

## Layout

```
scripts/
  generate-deepwiki.sh       # AI C4 architecture docs via deepwiki-rs (Litho)
  lib/
    common.sh                # log_* logging helpers, die, require_command
    toolchain.sh             # rust/python toolchain validate + install
  python/
    shadow-tree.py           # git-tracked shadow-tree builder (std-lib only)
```

| Script                 | Description                                                                                      |
| ---------------------- | ------------------------------------------------------------------------------------------------ |
| `generate-deepwiki.sh` | Generate AI C4 architecture docs with deepwiki-rs (Litho), backed by the local LiteLLM proxy     |
| `lib/toolchain.sh`     | Validate / install the supporting toolchains (rust, rustup, python3 via pyenv); macOS + Homebrew |

## Toolchain

The generate script needs a rust toolchain (to build deepwiki-rs from the FSE
fork) and python3 (for the shadow-tree builder; pyenv is preferred for python
version selection).

```bash
# Check what's present
scripts/generate-deepwiki.sh --check-toolchain

# Install what's missing (macOS + Homebrew: rustup, pyenv, python 3.12).
# Takes over the pyenv global only when none is set; an existing global is
# left untouched (use `pyenv local` for a per-project pin).
scripts/generate-deepwiki.sh --install-toolchain

# Or call the lib directly
scripts/lib/toolchain.sh check
scripts/lib/toolchain.sh install
```

## Generate architecture docs

```bash
# Install/update deepwiki-rs from the FSE fork (one time bootstrap)
scripts/generate-deepwiki.sh --update-deepwiki
cargo install mermaid-fixer

# Generate docs for the whole repo (output committed under deepwiki-rs-docs/)
scripts/generate-deepwiki.sh

# Generate docs for a subtree
scripts/generate-deepwiki.sh src/mcp

# Full option reference
scripts/generate-deepwiki.sh --help
```

Prerequisites: `deepwiki-rs` (FSE fork), `mermaid-fixer`, a rust toolchain +
python3 (see Toolchain above), and a `LITELLM_API_KEY` in the environment or
the repo-root `.env` file (copy `.env.example` to `.env`). The `.litho/`
analysis cache is local-only (gitignored); generated docs are committed under
`deepwiki-rs-docs/` by convention. See the script header for the complete
option and config-reference documentation.
