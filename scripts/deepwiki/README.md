# scripts/deepwiki - AI architecture docs (deepwiki-rs / Litho)

Generate AI C4 architecture documentation with deepwiki-rs (Litho), backed by
the local LiteLLM proxy. Generated output is committed under
`deepwiki-rs-docs/`; the `.litho/` analysis cache is local-only (gitignored).

## Layout

```
scripts/deepwiki/
  generate-deepwiki.sh   # entry point: config, shadow tree, run, normalize
  shadow-tree.py         # git-tracked shadow-tree builder (std-lib only)
  README.md
  lib/
    common.sh            # log_* logging helpers, die, require_command
    toolchain.sh         # rust/python toolchain validate + install
```

## Bootstrap (one time)

```bash
# Validate or install the rust + python toolchains (macOS + Homebrew)
scripts/deepwiki/generate-deepwiki.sh --check-toolchain
scripts/deepwiki/generate-deepwiki.sh --install-toolchain

# Build deepwiki-rs from the FSE fork (feat/filter-explain branch) and
# install the Mermaid validator it requires at startup
scripts/deepwiki/generate-deepwiki.sh --update-deepwiki
cargo install mermaid-fixer
```

## Generate architecture docs

```bash
# Whole repo (output committed under deepwiki-rs-docs/)
scripts/deepwiki/generate-deepwiki.sh

# A subtree (output under deepwiki-rs-docs/<path>)
scripts/deepwiki/generate-deepwiki.sh packages/broker-db

# Full option reference
scripts/deepwiki/generate-deepwiki.sh --help
```

By default the script analyzes a **shadow tree** of exactly the git-tracked
files (hardlinks under `.litho/tree/`), because deepwiki-rs's directory walk
does not honor `.gitignore`. Pass `--no-shadow` to analyze the working tree
directly (not recommended).

## Prerequisites

- `deepwiki-rs` FSE fork build (`--update-deepwiki`) and `mermaid-fixer`
  (`cargo install mermaid-fixer`)
- rust toolchain + python3 (`--check-toolchain` / `--install-toolchain`)
- `LITELLM_API_KEY` in the environment or the repo-root `.env` file
  (`cp .env.example .env`); optional `DEEPWIKI_*` overrides live in the same
  file
- root `litho.toml` (analysis config; discovered from the repo root - never
  add an `[llm]` section to it, see the file header)

Precedence for all settings: CLI flag > process env > `.env` > built-in
default. See the script header for the complete option reference.
