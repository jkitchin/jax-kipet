#!/usr/bin/env bash
# Install the kipetax skill into a Claude Code skills directory so an agent can
# load kipetax guidance on demand.
#
# Usage:
#   bash kipetax/skills/install.sh           # install into ./.claude (project)
#   bash kipetax/skills/install.sh --user    # install into ~/.claude (global)
#   bash kipetax/skills/install.sh --force   # overwrite an existing copy
#
# Prefer knowhow, which also targets Codex, OpenCode, Gemini CLI, Cursor, ...
# and tracks upgrades:
#   uvx knowhow add jkitchin/jax-kipet --skill kipetax
#
# Idempotent: safely re-runnable. An existing SKILL.md is overwritten only with
# --force.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

DEST=".claude"
FORCE=0
for arg in "$@"; do
  case "$arg" in
    --user)  DEST="$HOME/.claude" ;;
    --force) FORCE=1 ;;
    *)       echo "unknown flag: $arg"; exit 1 ;;
  esac
done

SRC="$SCRIPT_DIR/kipetax/SKILL.md"
DST_DIR="$DEST/skills/kipetax"
DST="$DST_DIR/SKILL.md"

mkdir -p "$DST_DIR"
if [[ -e "$DST" && $FORCE -eq 0 ]]; then
  echo "  skip (exists): $DST   (re-run with --force to overwrite)"
else
  cp "$SRC" "$DST"
  echo "  installed: $DST"
fi

echo "Done. The 'kipetax' skill is available to Claude Code in $DEST/skills/."
