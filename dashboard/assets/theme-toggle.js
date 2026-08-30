/* ============================================================================
 * Tema anahtari — Faz 8, B.4
 * ----------------------------------------------------------------------------
 * Uc durum: light | dark | system.
 *   light/dark -> <html data-theme="...">  damgasi
 *   system     -> damga YOK, karari @media (prefers-color-scheme) verir
 *
 * Ilk boyama <head>'deki senkron script tarafindan zaten dogru yapiliyor
 * (bkz. dashboard/app.py, INDEX_TEMPLATE). Bu dosya sonrasini yonetir:
 * tiklama, isletim sistemi degisimi, Plotly yeniden boyama ve sunucuya kayit.
 *
 * Renk degerleri BURADA YOK. Plotly'ye verilecek hex'ler
 * getComputedStyle ile tokens.css'ten okunur — tek kaynak orasi kalsin diye.
 * ========================================================================== */
(function () {
  "use strict";

  var THEME_COOKIE = "rlt_theme";
  var RESOLVED_COOKIE = "rlt_theme_r";
  var ORDER = ["light", "dark", "system"];
  var LABELS = { light: "Aydinlik", dark: "Koyu", system: "Sistem" };
  var ICONS = { light: "bi-sun", dark: "bi-moon-stars", system: "bi-circle-half" };

  // ── Cerez yardimcilari ───────────────────────────────────────────────────
  function readCookie(name) {
    var m = document.cookie.match(new RegExp("(?:^|; )" + name + "=([^;]*)"));
    return m ? decodeURIComponent(m[1]) : "";
  }

  function writeCookie(name, value) {
    document.cookie =
      name + "=" + encodeURIComponent(value) + ";path=/;max-age=31536000;samesite=lax";
  }

  function prefersDark() {
    return !!(window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)").matches);
  }

  function getPreference() {
    var p = readCookie(THEME_COOKIE);
    return ORDER.indexOf(p) === -1 ? "system" : p;
  }

  function resolve(pref) {
    if (pref === "light" || pref === "dark") return pref;
    return prefersDark() ? "dark" : "light";
  }

  // ── DOM damgasi ──────────────────────────────────────────────────────────
  function stamp(pref) {
    var root = document.documentElement;
    if (pref === "light" || pref === "dark") {
      root.setAttribute("data-theme", pref);
    } else {
      // "system" = damga yok; media sorgusu devreye girer
      root.removeAttribute("data-theme");
    }
  }

  // ── Plotly ───────────────────────────────────────────────────────────────
  function token(name) {
    return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  }

  /* Cizili figurleri yeniden boyar.
   * Not: trace renkleri (kar yesili vb.) relayout ile DEGISMEZ; onlar bir
   * sonraki callback yenilemesinde sunucudan dogru palette gelir. Iki temada
   * da AA gectikleri icin aradaki tur sorun degil. */
  function repaintPlots() {
    if (!window.Plotly) return;
    var surface = token("--surface");
    var grid = token("--border");
    var line = token("--border-strong");
    var text = token("--text");
    var muted = token("--muted");
    var surface2 = token("--surface-2");
    if (!surface || !text) return;

    var patch = {
      paper_bgcolor: surface,
      plot_bgcolor: surface,
      "font.color": text,
      "xaxis.gridcolor": grid,
      "xaxis.linecolor": line,
      "xaxis.zerolinecolor": grid,
      "xaxis.tickfont.color": muted,
      "yaxis.gridcolor": grid,
      "yaxis.linecolor": line,
      "yaxis.zerolinecolor": grid,
      "yaxis.tickfont.color": muted,
      "legend.font.color": muted,
      "legend.bordercolor": grid,
      "hoverlabel.bgcolor": surface2,
      "hoverlabel.bordercolor": line,
      "hoverlabel.font.color": text
    };

    document.querySelectorAll(".js-plotly-plot").forEach(function (el) {
      try {
        window.Plotly.relayout(el, patch);
      } catch (e) {
        /* figure henuz hazir degil — sonraki yenilemede duzelir */
      }
    });
  }

  // ── Sunucuya kayit ───────────────────────────────────────────────────────
  function persist(pref, resolved) {
    var csrf = readCookie("rlt_csrf");
    if (!csrf) return; // auth kapali ya da oturum yok: cerez yeterli
    fetch("/auth/preferences", {
      method: "PATCH",
      credentials: "same-origin",
      headers: {
        "Content-Type": "application/json",
        "X-CSRF-Token": csrf,
        "X-Theme-Resolved": resolved,
        "X-Requested-With": "XMLHttpRequest"
      },
      body: JSON.stringify({ theme: pref })
    }).catch(function () {
      /* Cevrimdisi olabilir; cerez yazildi, bir sonraki giriste DB tazelenir */
    });
  }

  // ── Arayuz senkronu ──────────────────────────────────────────────────────
  function syncControls(pref) {
    var btn = document.getElementById("theme-toggle");
    if (btn) {
      btn.setAttribute("title", "Gorunum: " + LABELS[pref] + " (degistirmek icin tikla)");
      var icon = btn.querySelector("i");
      if (icon) icon.className = "bi " + ICONS[pref];
      var label = btn.querySelector(".theme-label");
      if (label) label.textContent = LABELS[pref];
    }
    document.querySelectorAll("[data-theme-set]").forEach(function (el) {
      var val = el.getAttribute("data-theme-set");
      if (val === "__cycle__") return;
      el.classList.toggle("active", val === pref);
      el.setAttribute("aria-pressed", val === pref ? "true" : "false");
    });
  }

  // ── Uygulama ─────────────────────────────────────────────────────────────
  function apply(pref, options) {
    options = options || {};
    if (ORDER.indexOf(pref) === -1) pref = "system";
    var resolved = resolve(pref);

    stamp(pref);
    writeCookie(THEME_COOKIE, pref);
    writeCookie(RESOLVED_COOKIE, resolved);
    syncControls(pref);
    repaintPlots();

    if (options.persist !== false) persist(pref, resolved);
  }

  // ── Olaylar ──────────────────────────────────────────────────────────────
  // Dash sayfayi her gezinmede yeniden ciziyor; tek tek baglamak yerine
  // belge duzeyinde delegasyon kullaniliyor.
  document.addEventListener("click", function (ev) {
    var el = ev.target.closest("[data-theme-set]");
    if (!el) return;
    ev.preventDefault();
    var value = el.getAttribute("data-theme-set");
    if (value === "__cycle__") {
      var next = ORDER[(ORDER.indexOf(getPreference()) + 1) % ORDER.length];
      apply(next);
    } else {
      apply(value);
    }
  });

  // "system" seciliyken isletim sistemi temasi degisirse takip et
  if (window.matchMedia) {
    var mq = window.matchMedia("(prefers-color-scheme: dark)");
    var onChange = function () {
      if (getPreference() !== "system") return;
      writeCookie(RESOLVED_COOKIE, resolve("system"));
      repaintPlots();
    };
    if (mq.addEventListener) mq.addEventListener("change", onChange);
    else if (mq.addListener) mq.addListener(onChange);
  }

  /* Dash arayuzu React ile DOMContentLoaded'dan SONRA ciziliyor: kenar
   * cubugu ve tema dugmeleri o an henuz yok. Bu yuzden body'nin tamami
   * izleniyor ve dugmeler belirdiginde isaretleri tazeleniyor. */
  var syncQueued = false;
  var observer = new MutationObserver(function () {
    if (syncQueued) return;
    syncQueued = true;
    requestAnimationFrame(function () {
      syncQueued = false;
      syncControls(getPreference());
    });
  });

  function start() {
    // Damga <head>'de zaten kondu; burada yalnizca arayuzu esitliyoruz.
    // persist:false — acilista sunucuya gereksiz istek atmayalim.
    apply(getPreference(), { persist: false });
    observer.observe(document.body, { childList: true, subtree: true });
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", start);
  } else {
    start();
  }
})();
