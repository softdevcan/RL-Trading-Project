/* Takvim (dcc.DatePickerSingle / DatePickerRange) ay ve gun adlari Turkce (Faz 11 §3.16).
 *
 * Dash 4 takvimi Ingilizce adlari SABIT ciziyor ("November", "Su Mo Tu"); bilesende dil secenegi yok ve tarayici
 * dili (tr-TR ile denendi) etkilemiyor. Burada yalniz takvim kapsamindaki, TAM eslesen metin dugumleri cevrilir:
 *   - .dash-datepicker-calendar-wrapper icindeki ay adi ve gun basliklari,
 *   - ay secicinin acilir listesi (.dash-dropdown-content), ancak listede bir ay adi varsa.
 * Grafik eksenleri (SVG) ve sayfanin geri kalani kapsam disi. React ayni dugumu yeniden yazarsa gozlemci yeniden cevirir.
 */
(function () {
  "use strict";
  var MONTHS = {
    January: "Ocak", February: "Şubat", March: "Mart", April: "Nisan", May: "Mayıs", June: "Haziran",
    July: "Temmuz", August: "Ağustos", September: "Eylül", October: "Ekim", November: "Kasım", December: "Aralık"
  };
  // Uc harf: ceviri hedefi kaynak anahtarlarla CAKISMAMALI. Iki harfte Sali "Sa" = Ingilizce Saturday "Sa" idi ve
  // gozlemcinin ikinci turu onu "Ct" yapiyordu (tarayicida yakalandi).
  var DAYS = { Su: "Paz", Mo: "Pzt", Tu: "Sal", We: "Çar", Th: "Per", Fr: "Cum", Sa: "Cmt" };

  function translate(root, map) {
    var walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT, null);
    var node;
    while ((node = walker.nextNode())) {
      var t = node.nodeValue.trim();
      if (map.hasOwnProperty(t) && node.nodeValue !== map[t]) node.nodeValue = map[t];
    }
  }

  function hasMonth(el) {
    var txt = el.textContent || "";
    for (var m in MONTHS) if (txt.indexOf(m) !== -1) return true;
    return false;
  }

  function apply() {
    var wraps = document.querySelectorAll(".dash-datepicker-calendar-wrapper");
    for (var i = 0; i < wraps.length; i++) {
      translate(wraps[i], MONTHS);
      var heads = wraps[i].querySelectorAll(".dash-datepicker-calendar thead");
      for (var j = 0; j < heads.length; j++) translate(heads[j], DAYS);
    }
    var lists = document.querySelectorAll(".dash-dropdown-content");
    for (var k = 0; k < lists.length; k++) if (hasMonth(lists[k])) translate(lists[k], MONTHS);
  }

  var pending = false;
  function schedule() {
    if (pending) return;
    pending = true;
    window.requestAnimationFrame(function () { pending = false; apply(); });
  }

  function start() {
    new MutationObserver(schedule).observe(document.body, { childList: true, subtree: true, characterData: true });
    apply();
  }
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", start);
  else start();
})();
