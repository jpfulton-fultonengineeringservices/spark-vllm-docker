# Stub for the exllamav3 top-level package.
#
# The real __init__ imports torch and raises RuntimeError when torch is missing,
# and re-exports many runtime symbols (Model, Tokenizer, generators, etc.). The
# exl3-pack source never imports from the top-level package directly (it uses
# submodule paths such as `exllamav3.model.config` / `exllamav3.conversion`), so
# this stub stays intentionally empty to avoid triggering the torch import.
