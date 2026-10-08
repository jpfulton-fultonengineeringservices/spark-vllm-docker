# Node map resolution: node aliases, SSH hosts, host list.
# Sourced by scripts/exl3-pack-drive.sh; defines functions only.
# Target: /bin/bash 3.2 (macOS). No top-level side effects.

# The node's ssh_config alias doubles as the node-map alias.
node_alias() {
  if [ -n "$NODE" ]; then
    printf '%s' "$NODE"
  else
    printf '%s' "$HOST"
  fi
}

# Resolve a --nodes slug to its ssh host alias via the node map.
# Falls back to the global --host when the slug is absent from the map.
node_ssh_host() {
  local slug="$1"
  local map="${NODE_MAP}"
  if [ -z "$map" ]; then
    map="${REPO_ROOT}/mods/exl3-pack/node-model-map.json"
  fi
  local alias=""
  if [ -n "$map" ] && [ -f "$map" ]; then
    alias="$(python3 - "$map" "$slug" <<'PY' 2>/dev/null
import sys, json
try:
    data = json.load(open(sys.argv[1]))
    print(data.get("nodes", {}).get(sys.argv[2], {}).get("alias", ""))
except Exception:
    pass
PY
    )"
  fi
  if [ -n "$alias" ]; then
    printf '%s' "$alias"
  else
    printf '%s' "$HOST"
  fi
}

# Unique ordered list of ssh hosts needed for a --nodes list plus the coordinator.
resolved_hosts() {
  local hosts="$HOST"
  local slug
  IFS=',' read -ra _nodes <<< "$NODES"
  for slug in "${_nodes[@]}"; do
    slug="${slug// /}"
    [ -n "$slug" ] || continue
    local h
    h="$(node_ssh_host "$slug")"
    case ",$hosts," in
      *",$h,"*) ;;
      *) hosts="$hosts,$h" ;;
    esac
  done
  printf '%s' "$hosts"
}
