# Developer Tooling Domain

*module · agent map*

Auxiliary scripts supporting developer workflows: evaluation runners and automated repository documentation generation via DeepWiki.

Evaluation Runner: fes-eval.sh pre-verifies staged FES weights, assembles read-only mount, verification mod, and offline hub shim, then executes run-recipe.sh with any additional arguments. Documentation Generator: generate-deepwiki.sh sets up a local LiteLLM proxy, runs…

**Location:** [Home](../../index.md) › **Developer Tooling Domain**

## Diagrams

- [Flowchart](flowchart.md)
- [Sequence](sequence.md)

## Source

- [`scripts/fes-eval.sh`](../../../../.litho/tree/repo/scripts/fes-eval.sh)
- [`scripts/deepwiki/generate-deepwiki.sh`](../../../../.litho/tree/repo/scripts/deepwiki/generate-deepwiki.sh)
- [`scripts/deepwiki/shadow-tree.py`](../../../../.litho/tree/repo/scripts/deepwiki/shadow-tree.py)

## Interaction

- The domain exposes two main entry points: the Evaluation Runner script (scripts/fes-eval.sh) which is a shell command that takes arguments for weights root, hub model, recipe, and FES slug, pre-verifies staged FES weights, assembles read-only mount, verification mod, and offline…
