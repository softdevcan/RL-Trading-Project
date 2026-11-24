# CI/CD: GitHub Actions → GHCR → VDS

**Sürüm:** v0.1 · 2026-11-24 · Kullanıcı kararları (2026-11-24): imaj GitHub'da derlenir, GHCR'ye itilir, SSH ile
dağıtılır; M8 ileri veri defteri şimdilik yerelde kalır; VDS Linux + Docker hazır; dış erişim Cloudflare
üzerinden `rl.softdevcan.com`. Elle kurulum ayrıntıları [docker.md](docker.md).

## 1. Akış

```
PR açılır / main'e push      ──► CI (.github/workflows/ci.yml)
                                  imaj derlenir → konteyner açılır → /health 200 → oturumsuz /api 401

PR birleşir → sürüm etiketi   ──► Deploy (.github/workflows/deploy.yml)
  git tag -a vX.Y.Z              1. build: etiket = app/__init__.py __version__ mi? → imaj → GHCR
  git push origin vX.Y.Z            ghcr.io/softdevcan/rl-trading-project:vX.Y.Z (+ latest)
                                 2. deploy ('production' ortamı): compose dosyaları + betik scp ile VDS'ye,
                                    SSH ile scripts/deploy/vds_deploy.sh:
                                    pull → up -d --no-build → HEALTHCHECK 'healthy' bekle (≤ 200 sn)
                                    geçmezse bir önceki sürüm geri açılır, iş kırmızı biter
```

- VDS'de derleme yok; sunucuda yalnız `docker compose pull/up` çalışır (torch kurulumu GitHub'da).
- Tetik yalnız `v*.*.*` kalıbındaki etiket. Araştırma etiketleri (`m8-prereg-v1`, yerel `m8-forward-annex-v1`)
  dağıtım başlatmaz. Etiketleri tek tek it; `git push --tags` yasak (yerel etiket sızar).
- Depo herkese açık. Bu yüzden VDS'ye self-hosted runner kurulmaz: dışarıdan gelen bir PR o makinede kod
  çalıştırabilirdi. Dağıtım işi PR'larda çalışmaz, gizliler yalnız `production` ortamında.

## 2. Tek seferlik kurulum

### 2.1 VDS

```bash
# Dağıtım kullanıcısı (docker grubunda; root değil)
sudo adduser --disabled-password deploy
sudo usermod -aG docker deploy
sudo mkdir -p /opt/rltrading && sudo chown deploy:deploy /opt/rltrading

# Actions için ayrı SSH anahtarı (yerelde üret, özel anahtar yalnız GitHub gizlisine gider)
ssh-keygen -t ed25519 -f rltrading_deploy -C "github-actions-deploy" -N ""
# rltrading_deploy.pub içeriğini VDS'de ~deploy/.ssh/authorized_keys'e ekle
```

`/opt/rltrading/.env`'i [`.env.production.example`](../../.env.production.example)'dan oluştur (ilk dağıtım
örnek dosyayı da kopyalar). Bu kurulum için zorunlu değerler:

| Anahtar | Değer |
|---|---|
| `JWT_SECRET_KEY` | `python3 -c "import secrets; print(secrets.token_urlsafe(64))"` |
| `AUTH_ENABLED` | `True` (betik `False`'ta dağıtımı reddeder) |
| `COOKIE_SECURE` | `True` (Cloudflare HTTPS sonlandırır, tarayıcı HTTPS görür) |
| `CORS_ORIGINS` | `["https://rl.softdevcan.com"]` |
| `EVDS_API_KEY` | TCMB EVDS anahtarı (makro veri) |
| `BOOTSTRAP_ADMIN_EMAIL` / `_PASSWORD` | İlk açılışta admin; hesap açılınca SİL. Alternatif: `docker compose exec rltrading python scripts/create_admin.py` |

Eğitilmiş modeller imaja girmez: `models/` dizinini yerelden `/opt/rltrading/models/`'e kopyala (docker.md §4).
Piyasa verisi named volume'lerde boş başlar; uygulama kendisi çeker.

### 2.2 Cloudflare

**Bu VDS'de (2026-11-24): Cloudflare DNS → Nginx Proxy Manager (konteyner, 80/443) → uygulama.**
- `.env`'e `PROXY_NETWORK=<NPM'in Docker ağı>` yazılır; betik `docker-compose.proxy.yml`'i ekler ve uygulama o ağa
  da bağlanır. Ağ adı: `docker inspect nginx-proxy-manager-app-1 -f '{{range $k,$v := .NetworkSettings.Networks}}{{$k}} {{end}}'`.
- NPM → Proxy Hosts → Add: domain `rl.softdevcan.com`, scheme `http`, forward hostname `rltrading`, port `8000`,
  "Websockets Support" açık; SSL sekmesinde sertifika (diğer sitelerdeki gibi).
- Host portu yine `127.0.0.1:8000`'e bağlı kalır (sunucuda `curl` için); dışarıdan yalnız NPM üzerinden erişilir.
- `X-Forwarded-For`'u NPM yazar; Cloudflare'in eklediği istemci IP'si başta kalır.

Diğer yol, **Cloudflare Tunnel** (`cloudflared` host'ta servis): public hostname `rl.softdevcan.com` →
servis `http://localhost:8000` (sunucuda 8000 doluysa `.env`'de `HOST_PORT` ve tünel hedefi birlikte değişir).
Aynı VDS'de başka projeler varsa mevcut tünele yeni bir hostname eklemek yeterli; yeni tünel gerekmez.
- Betik portu `127.0.0.1:8000`'e bağlar (`BIND_ADDR`); uygulama dışarıya yalnız tünelden açılır. Giriş
  hız sınırı `X-Forwarded-For`'a bakar (`app/auth/routes.py::client_ip`); port dışarıya açık olsaydı bu başlık
  sahte gönderilebilirdi.
- `cloudflared` bir konteynerde çalışıyorsa `localhost` o konteynerin kendisidir: ya host ağında çalıştır
  (`network_mode: host`) ya da `BIND_ADDR`'ı docker köprü adresine ver.
- Tünel yerine turuncu bulut (DNS vekili) kullanılıyorsa 8000 portu Cloudflare'in desteklediği bir port
  değildir: önüne Caddy/nginx (443) gerekir (docker.md §9).

### 2.3 GitHub

Settings → Environments → `production` (isteğe bağlı: "Required reviewers" ile her dağıtım onay ister).

| Tür | Ad | Değer |
|---|---|---|
| Secret | `VDS_HOST` | VDS IP ya da SSH alan adı |
| Secret | `VDS_USER` | `deploy` |
| Secret | `VDS_SSH_KEY` | `rltrading_deploy` özel anahtarının tamamı |
| Secret (isteğe bağlı) | `VDS_PORT` | SSH portu (varsayılan 22) |
| Variable (isteğe bağlı) | `VDS_PATH` | `/opt/rltrading` |

- GitHub'ın çalıştırıcıları VDS'nin SSH portuna **internetten** erişebilmeli. SSH yalnız Cloudflare Access
  arkasındaysa bu iş çalışmaz; o durumda iş `cloudflared access ssh` adımıyla genişletilmeli (açık madde).
- GHCR paketi ilk itişte oluşur ve gizli başlar. VDS çekmeyi iş sırasında `GITHUB_TOKEN` ile yapar (token işin
  sonunda geçersizleşir, `docker logout` çalışır); paketi herkese açmak gerekmez.

## 3. Günlük kullanım

- **Yeni sürüm:** PR birleşir → `git tag -a vX.Y.Z -m "..."` → `git push origin vX.Y.Z` →
  `gh release create vX.Y.Z --generate-notes` ([versioning.md](../development/versioning.md)). Dağıtım etiket
  itilince kendiliğinden başlar.
- **Geri dönüş:** Actions → Deploy → Run workflow → eski etiket. Ya da VDS'de
  `bash /opt/rltrading/scripts/deploy/vds_deploy.sh --rollback` (bir önceki başarılı sürüm, `.deploy-state`).
- **Durum:** `cat /opt/rltrading/.deploy-state` (çalışan sürüm, önceki sürüm, zaman);
  `docker compose -f docker-compose.yml -f docker-compose.prod.yml logs -f rltrading`.

## 4. Kapsam dışı ve açık maddeler

- **M8 ileri veri defteri VDS'de çalışmaz** (kullanıcı kararı 2026-11-24). İmajda `data/marl` yok
  (`.dockerignore`), `MARL_FORWARD_AUTORUN` varsayılan kapalı; VDS panosunda simülasyon sayfası arşiv yok der.
  Defteri VDS'ye taşımak ayrı paket: 2,8 GB arşiv kopyası + SHA doğrulaması + tek yazar devri (yerelde iş kapanır).
- **Python testleri CI'da yok.** Çoğu git dışı veriye bağlı (panel, arşiv); veri gerektirmeyen alt küme temiz
  bir klonda ölçülüp ayrı iş olarak eklenecek.
- SSH yalnız Cloudflare Access arkasındaysa §2.3 notu.
- Eski imajlar VDS'de birikir (geri dönüş için tutuluyor); disk dolarsa `docker image prune -a` ile temizlenir.

## 5. Doğrulama (2026-11-24)

- Workflow ve compose dosyaları YAML olarak ayrıştırıldı; birleşik compose (`docker compose config`, IMAGE_TAG
  `v9.9.9`) imajı `ghcr.io/softdevcan/rl-trading-project:v9.9.9`, portu `127.0.0.1:8000` olarak çözüyor.
- `vds_deploy.sh` sahte `docker` ile denendi: ilk dağıtım, ikinci dağıtım (önceki sürüm kaydı), sağlıksız
  sürümde otomatik geri dönüş (çıkış 1), aynı sürümü yeniden açma, `--rollback`, `AUTH_ENABLED=false` reddi.
- İmaj derlemesi ve konteyner sağlık denetimi bu PR'ın CI koşusunda (yerelde Docker servisi kapalıydı).
- İlk gerçek dağıtım VDS gizlileri girilince ilk sürüm etiketiyle.
