/* eval report explorer — vanilla JS, no network access. */
(function () {
  "use strict";

  var dataEl = document.getElementById("eval-data");
  var data = dataEl ? JSON.parse(dataEl.textContent) : { run: {}, tasks: [] };

  var PAGE = 25;
  var state = { task: "all", outcome: "all", query: "", page: 0, activeTab: "overview" };

  function esc(s) {
    return String(s).replace(/[&<>"']/g, function (c) {
      return { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c];
    });
  }

  function highlight(text, query) {
    var safe = esc(text);
    if (!query) return safe;
    var idx = text.toLowerCase().indexOf(query.toLowerCase());
    if (idx < 0) return safe;
    var before = esc(text.slice(0, idx));
    var hit = esc(text.slice(idx, idx + query.length));
    var after = esc(text.slice(idx + query.length));
    return before + "<mark>" + hit + "</mark>" + after;
  }

  function matches(sample, taskName) {
    if (state.task !== "all" && taskName !== state.task) return false;
    if (state.outcome !== "all" && sample.outcome !== state.outcome) return false;
    if (!state.query) return true;
    var q = state.query.toLowerCase();
    return (
      sample.prompt.toLowerCase().indexOf(q) >= 0 ||
      sample.output.toLowerCase().indexOf(q) >= 0 ||
      sample.reference.toLowerCase().indexOf(q) >= 0 ||
      String(sample.id).toLowerCase().indexOf(q) >= 0
    );
  }

  function choicesTable(sample) {
    if (!sample.choices || !sample.choices.length) return "";
    var rows = sample.choices
      .map(function (c, i) {
        var cls = i === sample.picked ? ' class="picked"' : "";
        var score = c[1] === null || c[1] === undefined ? "" : Number(c[1]).toFixed(3);
        return "<tr" + cls + "><td>" + i + "</td><td>" + esc(c[0]) + "</td><td>" + score + "</td></tr>";
      })
      .join("");
    return (
      '<table class="choices"><thead><tr><th>i</th><th>choice</th><th>score</th></tr></thead>' +
      "<tbody>" + rows + "</tbody></table>"
    );
  }

  function sampleCard(sample, taskName, query) {
    query = query || "";
    var scores = Object.keys(sample.scores)
      .map(function (k) { return k + "=" + sample.scores[k]; })
      .join(" ");
    return (
      '<article class="sample-card outcome-' + esc(sample.outcome) + '">' +
      '<div class="sample-head">' +
      '<span class="badge badge-' + esc(sample.outcome) + '">' + esc(sample.outcome) + "</span>" +
      '<span class="sample-id">' + highlight(String(sample.id), query) + "</span>" +
      '<span class="sample-scores">' + esc(scores) + "</span>" +
      '<span class="muted">' + esc(taskName) + "</span>" +
      "</div>" +
      choicesTable(sample) +
      "<details><summary>prompt</summary><pre>" + highlight(sample.prompt, query) + "</pre></details>" +
      "<details open><summary>output</summary><pre>" + highlight(sample.output, query) + "</pre></details>" +
      "<details><summary>reference</summary><pre>" + highlight(sample.reference, query) + "</pre></details>" +
      "</article>"
    );
  }

  function collect() {
    var out = [];
    data.tasks.forEach(function (task) {
      task.samples.forEach(function (s) {
        if (matches(s, task.task)) out.push([s, task.task]);
      });
    });
    return out;
  }

  function render() {
    var list = document.getElementById("explorer-list");
    var count = document.getElementById("count");
    if (!list) return;
    var rows = collect();
    var pages = Math.max(Math.ceil(rows.length / PAGE), 1);
    if (state.page >= pages) state.page = pages - 1;
    var start = state.page * PAGE;
    var slice = rows.slice(start, start + PAGE);
    if (!rows.length) {
      var empty = data.tasks.length
        ? "no samples match the current filter"
        : "this run has no samples";
      list.innerHTML = '<p class="muted">' + empty + "</p>";
    } else {
      list.innerHTML = slice
        .map(function (r) { return sampleCard(r[0], r[1], state.query); })
        .join("");
    }
    if (count) {
      count.textContent = rows.length
        ? "showing " + (start + 1) + "–" + (start + slice.length) + " of " + rows.length
        : "0 samples";
    }
  }

  function buildTaskFilter() {
    var sel = document.getElementById("task-filter");
    if (!sel) return;
    var opts = ['<option value="all">all tasks</option>'];
    data.tasks.forEach(function (t) {
      opts.push('<option value="' + esc(t.task) + '">' + esc(t.task) + "</option>");
    });
    sel.innerHTML = opts.join("");
  }

  function renderTaskPanel(task) {
    var panel = document.getElementById("tab-" + task.slug);
    if (!panel) return;
    var list = panel.querySelector(".task-explorer-list");
    var tools = panel.querySelector(".task-explorer-tools");
    if (!list || !tools) return;
    var search = tools.querySelector("input");
    var outcome = tools.querySelector("select");
    var query = (search ? search.value : "").trim().toLowerCase();
    var filter = outcome ? outcome.value : "all";
    var matched = task.samples.filter(function (sample) {
      if (filter !== "all" && sample.outcome !== filter) return false;
      if (!query) return true;
      return [sample.prompt, sample.output, sample.reference, String(sample.id)]
        .some(function (v) { return v.toLowerCase().indexOf(query) >= 0; });
    });
    var page = Number(panel.dataset.page || 0);
    var pages = Math.max(Math.ceil(matched.length / PAGE), 1);
    page = Math.min(page, pages - 1);
    panel.dataset.page = String(page);
    var start = page * PAGE;
    var slice = matched.slice(start, start + PAGE);
    list.innerHTML = slice.length
      ? slice.map(function (s) { return sampleCard(s, task.task, query); }).join("")
      : '<p class="muted">' + esc(task.sample_note || "no samples match the current filter") + "</p>";
    var count = tools.querySelector(".count");
    if (count) count.textContent = matched.length
      ? "showing " + (start + 1) + "–" + (start + slice.length) + " of " + matched.length
      : "0 samples";
  }

  function renderTaskPanels() {
    data.tasks.forEach(renderTaskPanel);
  }

  function wireTabs() {
    document.querySelectorAll(".task-tab").forEach(function (button) {
      button.addEventListener("click", function () {
        var id = button.getAttribute("data-tab");
        state.activeTab = id;
        document.querySelectorAll(".task-tab").forEach(function (b) {
          b.classList.toggle("active", b === button);
        });
        document.querySelectorAll(".tab-panel").forEach(function (panel) {
          var show = panel.id === (id === "overview" ? "overview-tab" : "tab-" + id);
          panel.hidden = !show;
          panel.classList.toggle("active", show);
        });
        document.querySelectorAll(".overview-only").forEach(function (panel) {
          panel.hidden = id !== "overview";
        });
        if (id !== "overview") {
          var target = data.tasks.filter(function (t) { return t.slug === id; })[0];
          if (target) renderTaskPanel(target);
        }
      });
    });
  }

  function hookKeyboard() {
    document.addEventListener("keydown", function (ev) {
      var search = state.activeTab === "overview"
        ? document.getElementById("search")
        : document.querySelector("#tab-" + state.activeTab + " .task-explorer-tools input");
      if (!search) return;
      var typing = /^(INPUT|TEXTAREA|SELECT)$/.test(document.activeElement.tagName);
      if (ev.key === "/" && !typing) {
        ev.preventDefault();
        search.focus();
      } else if (ev.key === "Escape" && typing) {
        search.value = "";
        if (state.activeTab === "overview") {
          state.query = "";
          state.page = 0;
          render();
        } else {
          var current = data.tasks.filter(function (t) { return t.slug === state.activeTab; })[0];
          if (current) {
            var panel = document.getElementById("tab-" + current.slug);
            panel.dataset.page = "0";
            renderTaskPanel(current);
          }
        }
      } else if ((ev.key === "j" || ev.key === "k") && !typing) {
        ev.preventDefault();
        var cardSelector = state.activeTab === "overview"
          ? "#explorer-list .sample-card"
          : "#tab-" + state.activeTab + " .task-explorer-list .sample-card";
        var cards = document.querySelectorAll(cardSelector);
        if (!cards.length) return;
        var cur = Array.prototype.indexOf.call(cards, document.activeElement);
        var next = ev.key === "j" ? Math.min(cur + 1, cards.length - 1) : Math.max(cur - 1, 0);
        if (cur === -1) next = 0;
        cards[next].setAttribute("tabindex", "-1");
        cards[next].focus();
      }
    });
  }

  function hookTheme() {
    var btn = document.getElementById("theme-toggle");
    function apply(theme) {
      if (theme) document.documentElement.setAttribute("data-theme", theme);
      else document.documentElement.removeAttribute("data-theme");
    }
    try {
      var saved = window.localStorage.getItem("evalReportTheme");
      if (saved) apply(saved);
    } catch (e) { /* file:// restrictions */ }
    if (!btn) return;
    btn.addEventListener("click", function () {
      var cur = document.documentElement.getAttribute("data-theme");
      var next = cur === "dark" ? "light" : "dark";
      apply(next);
      try { window.localStorage.setItem("evalReportTheme", next); } catch (e) { /* ignore */ }
    });
  }

  function bind() {
    var search = document.getElementById("search");
    var outcome = document.getElementById("outcome-filter");
    var task = document.getElementById("task-filter");
    var prev = document.getElementById("prev");
    var next = document.getElementById("next");
    if (search) {
      search.addEventListener("input", function () {
        state.query = search.value.trim();
        state.page = 0;
        render();
      });
    }
    if (outcome) {
      outcome.addEventListener("change", function () {
        state.outcome = outcome.value;
        state.page = 0;
        render();
      });
    }
    if (task) {
      task.addEventListener("change", function () {
        state.task = task.value;
        state.page = 0;
        render();
      });
    }
    if (prev) prev.addEventListener("click", function () { state.page = Math.max(state.page - 1, 0); render(); });
    if (next) next.addEventListener("click", function () { state.page += 1; render(); });
    document.querySelectorAll(".task-explorer-tools").forEach(function (tools) {
      var panel = tools.closest(".tab-panel");
      var t = data.tasks.filter(function (item) { return "tab-" + item.slug === panel.id; })[0];
      if (!t) return;
      var search = tools.querySelector("input");
      var outcome = tools.querySelector("select");
      var prev = tools.querySelector(".prev");
      var next = tools.querySelector(".next");
      if (search) search.addEventListener("input", function () {
        panel.dataset.page = "0";
        renderTaskPanel(t);
      });
      if (outcome) outcome.addEventListener("change", function () {
        panel.dataset.page = "0";
        renderTaskPanel(t);
      });
      if (prev) prev.addEventListener("click", function () {
        panel.dataset.page = String(Math.max(Number(panel.dataset.page || 0) - 1, 0));
        renderTaskPanel(t);
      });
      if (next) next.addEventListener("click", function () {
        panel.dataset.page = String(Number(panel.dataset.page || 0) + 1);
        renderTaskPanel(t);
      });
    });
  }

  document.addEventListener("DOMContentLoaded", function () {
    try {
      // Tabs only make sense with JS; without it the panels stay visible
      // so the whole report reads as one stacked document.
      document.documentElement.classList.add("js");
      var overviewPanel = document.getElementById("overview-tab");
      if (overviewPanel) overviewPanel.classList.add("active");
      document.querySelectorAll(".task-tab-panel").forEach(function (panel) {
        panel.classList.remove("active");
      });
      buildTaskFilter();
      wireTabs();
      hookTheme();
      hookKeyboard();
      bind();
      render();
      renderTaskPanels();
    } catch (e) {
      var list = document.getElementById("explorer-list");
      if (list) list.innerHTML = '<p class="muted">explorer init failed: ' + esc(e.message) + "</p>";
    }
  });
})();
