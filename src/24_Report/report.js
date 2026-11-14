(function () {
  "use strict";
  var root = document.documentElement;
  root.classList.add("po-js");
  var entries = Array.prototype.slice.call(document.querySelectorAll(".po-entry"));
  var byName = {};
  entries.forEach(function (e) { byName[e.getAttribute("data-name")] = e; });
  var select = document.getElementById("po-select");
  var hidden = {};

  function groupOf(e) { return e.getAttribute("data-group"); }
  function isVisible(e) { var g = groupOf(e); return !(g !== null && hidden[g]); }
  function order() {
    if (!select) { return entries; }
    return Array.prototype.map.call(select.options, function (o) {
      return document.getElementById(o.value);
    });
  }
  function show(id) {
    var target = document.getElementById(id);
    if (!target || !target.classList.contains("po-entry") || !isVisible(target)) {
      target = order().filter(isVisible)[0] || entries[0];
    }
    entries.forEach(function (e) { e.classList.toggle("po-active", e === target); });
    if (select && target) { select.value = target.id; }
  }
  function go(id) {
    try { history.replaceState(null, "", "#" + id); } catch (err) { location.hash = id; }
    show(id);
  }
  function active() {
    return entries.filter(function (e) { return e.classList.contains("po-active"); })[0];
  }
  function step(d) {
    var seq = order().filter(isVisible);
    var i = seq.indexOf(active());
    var j = i + d;
    if (j >= 0 && j < seq.length) { go(seq[j].id); }
  }

  if (select) {
    select.addEventListener("change", function () { go(select.value); });
  }
  window.addEventListener("hashchange", function () { show(location.hash.slice(1)); });
  document.addEventListener("keydown", function (ev) {
    var t = ev.target && ev.target.tagName;
    if (t === "INPUT" || t === "SELECT" || t === "TEXTAREA" || t === "BUTTON") { return; }
    if (ev.altKey || ev.ctrlKey || ev.metaKey || ev.shiftKey) { return; }
    if (ev.key === "ArrowRight") { step(1); ev.preventDefault(); }
    if (ev.key === "ArrowLeft") { step(-1); ev.preventDefault(); }
  });

  function sortBy(table, i, th) {
    var dir = th.getAttribute("aria-sort") === "ascending" ? "descending" : "ascending";
    Array.prototype.forEach.call(table.tHead.rows[0].cells, function (c) {
      c.removeAttribute("aria-sort");
    });
    th.setAttribute("aria-sort", dir);
    var sign = dir === "ascending" ? 1 : -1;
    var body = table.tBodies[0];
    var rows = Array.prototype.slice.call(body.rows);
    rows.sort(function (a, b) {
      var ca = a.cells[i], cb = b.cells[i];
      var va = ca ? ca.getAttribute("data-v") : null;
      var vb = cb ? cb.getAttribute("data-v") : null;
      if (va === null || vb === null) {
        return va === vb ? 0 : (va === null ? 1 : -1);
      }
      var num = ca.classList.contains("po-num") && cb.classList.contains("po-num");
      var c = num ? Number(va) - Number(vb) : va.localeCompare(vb);
      return sign * c;
    });
    rows.forEach(function (r) { body.appendChild(r); });
  }
  Array.prototype.forEach.call(document.querySelectorAll("table.po-sortable"), function (t) {
    Array.prototype.forEach.call(t.tHead.rows[0].cells, function (th, i) {
      th.tabIndex = 0;
      th.addEventListener("click", function () { sortBy(t, i, th); });
      th.addEventListener("keydown", function (ev) {
        if (ev.key === "Enter" || ev.key === " ") { sortBy(t, i, th); ev.preventDefault(); }
      });
    });
  });
  Array.prototype.forEach.call(document.querySelectorAll("tr[data-entry]"), function (r) {
    r.addEventListener("click", function () {
      var e = byName[r.getAttribute("data-entry")];
      if (e) { go(e.id); }
    });
  });

  function applyGroups() {
    entries.forEach(function (e) { e.classList.toggle("po-hidden", !isVisible(e)); });
    Array.prototype.forEach.call(document.querySelectorAll("tr[data-entry]"), function (r) {
      var e = byName[r.getAttribute("data-entry")];
      r.classList.toggle("po-hidden", !!e && !isVisible(e));
    });
    if (select) {
      Array.prototype.forEach.call(select.options, function (o) {
        o.hidden = !isVisible(document.getElementById(o.value));
      });
      Array.prototype.forEach.call(select.querySelectorAll("optgroup"), function (g) {
        g.hidden = !!hidden[g.getAttribute("label")];
      });
    }
    var a = active();
    if (!a || !isVisible(a)) { show(""); }
  }
  Array.prototype.forEach.call(document.querySelectorAll(".po-groups input"), function (c) {
    c.addEventListener("change", function () {
      hidden[c.getAttribute("data-group")] = !c.checked;
      applyGroups();
    });
  });

  var toggle = document.getElementById("po-theme");
  if (toggle) {
    toggle.addEventListener("click", function () {
      var cur = root.getAttribute("data-theme");
      if (!cur) {
        var dark = window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)").matches;
        cur = dark ? "dark" : "light";
      }
      root.setAttribute("data-theme", cur === "dark" ? "light" : "dark");
    });
  }

  show(location.hash.slice(1));
})();
