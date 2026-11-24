#!/usr/bin/env bash
# VDS dagitimi (docs/deployment/ci-cd.md). GitHub Actions "Deploy" isi SSH ile cagirir; elle de calisir:
#
#   TAG=v2.24.0 bash scripts/deploy/vds_deploy.sh      # bir surumu ac (imaj GHCR'den)
#   bash scripts/deploy/vds_deploy.sh --rollback       # bir onceki basarili surume don
#
# Ortam: APP_DIR (varsayilan /opt/rltrading), GHCR_USER + GHCR_TOKEN (yalniz Actions'ta; gizli paket cekmek icin),
# BIND_ADDR (varsayilan 127.0.0.1: port yalniz tunel/vekile acik).
# Saglik: konteynerin HEALTHCHECK durumu 'healthy' olmazsa bir onceki surum geri acilir ve cikis 1.
set -euo pipefail

APP_DIR="${APP_DIR:-/opt/rltrading}"
cd "$APP_DIR"
STATE=.deploy-state
export BIND_ADDR="${BIND_ADDR:-127.0.0.1}"
IMAGE="${IMAGE:-ghcr.io/softdevcan/rl-trading-project}"
COMPOSE=(docker compose -f docker-compose.yml -f docker-compose.prod.yml)
# Ters vekil konteyneri (Nginx Proxy Manager vb.) kullaniliyorsa uygulama onun agina da baglanir (docker-compose.proxy.yml).
if [ -f .env ] && grep -Eq '^[[:space:]]*PROXY_NETWORK[[:space:]]*=[[:space:]]*[^[:space:]]' .env; then
  COMPOSE+=(-f docker-compose.proxy.yml)
fi

state() { [ -f "$STATE" ] && sed -n "s/^$1=//p" "$STATE" | tail -1 || true; }
prev="$(state IMAGE_TAG)"
if [ "${1:-}" = "--rollback" ]; then
  TAG="$(state PREV_TAG)"
  [ -n "$TAG" ] || { echo "geri donulecek onceki surum yok ($STATE)"; exit 1; }
fi
: "${TAG:?TAG gerekli (or. TAG=v2.24.0)}"

[ -f .env ] || { echo ".env yok: $APP_DIR/.env (.env.production.example'dan olusturun)"; exit 1; }
if grep -Eiq '^[[:space:]]*AUTH_ENABLED[[:space:]]*=[[:space:]]*"?false' .env; then
  echo "AUTH_ENABLED=False ile sunucuya cikilmaz (pano ve API herkese acik kalir)"; exit 1
fi
# Konteyner uid 10001 ile calisir; bind mount dizinleri yoksa Docker root:root acar ve uygulama yazamaz.
mkdir -p models workspaces
chown 10001:10001 workspaces 2>/dev/null || true

if [ -n "${GHCR_TOKEN:-}" ]; then
  echo "$GHCR_TOKEN" | docker login ghcr.io -u "${GHCR_USER:?}" --password-stdin >/dev/null
  trap 'docker logout ghcr.io >/dev/null 2>&1 || true' EXIT
fi

up() {
  IMAGE_TAG="$1" "${COMPOSE[@]}" pull rltrading || docker image inspect "$IMAGE:$1" >/dev/null
  IMAGE_TAG="$1" "${COMPOSE[@]}" up -d --no-build --remove-orphans
}

healthy() {
  local s
  for _ in $(seq 1 40); do
    s="$(docker inspect -f '{{.State.Health.Status}}' rltrading 2>/dev/null || echo yok)"
    [ "$s" = healthy ] && return 0
    sleep 5
  done
  echo "son saglik durumu: $s"
  return 1
}

echo "dagitiliyor: $TAG (onceki: ${prev:-yok}, bag: $BIND_ADDR)"
up "$TAG"
if healthy; then
  new_prev="$prev"; [ "$prev" = "$TAG" ] && new_prev="$(state PREV_TAG)"
  printf 'IMAGE_TAG=%s\nPREV_TAG=%s\nDEPLOYED_AT=%s\n' "$TAG" "$new_prev" "$(date -Is)" > "$STATE"
  docker image prune -f >/dev/null || true
  echo "dagitildi: $TAG"
  exit 0
fi

echo "saglik denetimi gecmedi: $TAG"
docker logs --tail 80 rltrading 2>&1 || true
if [ -n "$prev" ] && [ "$prev" != "$TAG" ]; then
  echo "geri donuluyor: $prev"
  up "$prev"
  if healthy; then echo "geri donuldu: $prev"; else echo "geri donus da saglikli degil: $prev"; fi
fi
exit 1
