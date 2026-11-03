(function () {
  "use strict";

  const $ = (id) => document.getElementById(id);

  function setStatus(el, msg, kind) {
    el.textContent = msg || "";
    el.className = "status-line " + (kind || "");
  }

  document.querySelectorAll(".tab").forEach((btn) => {
    btn.addEventListener("click", () => {
      document.querySelectorAll(".tab").forEach((b) => b.classList.remove("active"));
      document.querySelectorAll(".panel").forEach((p) => p.classList.remove("active"));
      btn.classList.add("active");
      const panel = $("panel-" + btn.dataset.tab);
      if (panel) panel.classList.add("active");
    });
  });

  function switchTab(name) {
    const btn = document.querySelector('.tab[data-tab="' + name + '"]');
    if (btn) btn.click();
  }

  async function getJSON(url, options) {
    const res = await fetch(url, options || {});
    const text = await res.text();
    let data;
    try {
      data = JSON.parse(text);
    } catch {
      throw new Error(text.slice(0, 200) || res.statusText || "Invalid server response");
    }
    if (!res.ok) throw new Error(data.error || data.detail || res.statusText);
    return data;
  }

  let lastOriginalText = "";
  let lastFilename = "";
  let lastCheckpoint = "";

  async function boot() {
    try {
      const health = await getJSON("/api/health");
      const modelReady = health.model_loaded ? "model ready" : "model loads on first recognize";
      const acc =
        health.heldout_char_accuracy_pct != null
          ? health.heldout_char_accuracy_pct
          : health.heldout_char_accuracy_pct_base;
      $("healthBanner").innerHTML =
        `<span class="ok">Service online</span> · ` +
        `testing workspace · ${modelReady} · held-out accuracy ` +
        `<strong>${acc}%</strong>`;
      const ck = await getJSON("/api/checkpoint");
      $("checkpointInfo").textContent =
        "Active model: " +
        (ck.selected || "") +
        " · " +
        (ck.reason || "") +
        " · comparisons: " +
        JSON.stringify(ck.comparisons || {});
      $("metricActive").textContent =
        (ck.active_heldout_char_accuracy_pct != null
          ? ck.active_heldout_char_accuracy_pct
          : acc) + "%";
    } catch (err) {
      $("healthBanner").innerHTML =
        `<span class="warn">Service not reachable.</span> Start with ` +
        `<code>.\\.venv\\Scripts\\python.exe scripts\\run_judge_demo_server.py --port 8766 --preload-model</code>`;
    }
    loadSamples();
  }

  async function loadSamples() {
    const row = $("sampleRow");
    if (!row) return;
    try {
      const data = await getJSON("/api/samples");
      row.innerHTML = "";
      if (!data.samples || !data.samples.length) {
        row.innerHTML = '<p class="muted small">No sample images found under demo/approved.</p>';
        return;
      }
      const label = document.createElement("span");
      label.className = "sample-label";
      label.textContent = "Quick samples:";
      row.appendChild(label);
      data.samples.slice(0, 8).forEach((s) => {
        const btn = document.createElement("button");
        btn.type = "button";
        btn.className = "sample-chip";
        btn.textContent = s.label.replace(/^demo \d+ /i, "").slice(0, 28);
        btn.title = s.id;
        btn.addEventListener("click", () => useSample(s));
        row.appendChild(btn);
      });
    } catch (err) {
      row.innerHTML = '<p class="muted small">Samples unavailable.</p>';
    }
  }

  async function useSample(sample) {
    try {
      const res = await fetch(sample.url);
      if (!res.ok) throw new Error("Could not load sample");
      const blob = await res.blob();
      const file = new File([blob], sample.id, { type: "image/png" });
      const dt = new DataTransfer();
      dt.items.add(file);
      $("recFile").files = dt.files;
      showPreview(file);
      setStatus($("recStatus"), "Sample loaded: " + sample.id, "ok");
      if ($("recAuto").checked) runRecognize(file);
    } catch (err) {
      setStatus($("recStatus"), "Sample error: " + err.message, "warn");
    }
  }

  let recObjectUrl = null;
  let recognizing = false;

  function showPreview(f) {
    const box = $("recPreview");
    if (recObjectUrl) URL.revokeObjectURL(recObjectUrl);
    if (!f) {
      box.innerHTML = '<p class="muted">Preview appears here.</p>';
      return;
    }
    recObjectUrl = URL.createObjectURL(f);
    box.innerHTML = "";
    const img = document.createElement("img");
    img.alt = "Upload preview";
    img.src = recObjectUrl;
    box.appendChild(img);
  }

  $("recFile").addEventListener("change", () => {
    const f = $("recFile").files && $("recFile").files[0];
    showPreview(f);
    if (!f) return;
    setStatus($("recStatus"), "Image ready.", "ok");
    if ($("recAuto").checked) runRecognize(f);
  });

  function setRecognizing(on, label) {
    recognizing = on;
    const btn = $("btnRecognize");
    const prog = $("recProgress");
    btn.disabled = on;
    btn.textContent = on ? "Working…" : "Recognize";
    if (on) {
      prog.classList.remove("hidden");
      $("recProgressText").textContent = label || "Recognizing handwriting…";
    } else {
      prog.classList.add("hidden");
    }
  }

  async function runRecognize(file) {
    if (recognizing) return;
    const f = file || ($("recFile").files && $("recFile").files[0]);
    if (!f) {
      setStatus($("recStatus"), "Choose an image first.", "warn");
      return;
    }
    const ext = (f.name.split(".").pop() || "").toLowerCase();
    if (!["png", "jpg", "jpeg"].includes(ext)) {
      setStatus($("recStatus"), "Only PNG/JPG/JPEG allowed.", "warn");
      return;
    }

    $("recText").value = "";
    $("recMeta").textContent = "";
    $("recTime").textContent = "";
    $("recNote").textContent = "";
    lastOriginalText = "";
    lastFilename = f.name;
    lastCheckpoint = "";
    setRecognizing(true, "Uploading image…");
    setStatus($("recStatus"), "Recognizing…", "");

    const fd = new FormData();
    fd.append("image", f, f.name);
    fd.append("force_single_line", $("recForceSingle").checked ? "true" : "false");

    const controller = new AbortController();
    const timeoutMs = 180000;
    const timer = setTimeout(() => controller.abort(), timeoutMs);

    try {
      setTimeout(() => {
        if (recognizing) {
          $("recProgressText").textContent = "Running recognition…";
        }
      }, 2000);

      const data = await getJSON("/api/recognize", {
        method: "POST",
        body: fd,
        signal: controller.signal,
      });

      const text = (data.text || "").trim();
      lastOriginalText = text;
      lastCheckpoint = data.checkpoint || "";
      $("recText").value =
        text ||
        "(No readable text returned. Try a clearer single-line crop, or uncheck single-line for full pages.)";
      $("recTime").textContent =
        data.inference_time_s != null ? `(${data.inference_time_s}s)` : "";
      $("recNote").textContent =
        (data.recognition && data.recognition.segmentation_note) || data.storage_note || "";
      $("recMeta").textContent = JSON.stringify(
        {
          checkpoint: data.checkpoint,
          selection: data.selection,
          mode: data.recognition && data.recognition.mode,
          n_lines: data.recognition && data.recognition.n_lines_detected,
          lines: (data.recognition && data.recognition.lines) || [],
        },
        null,
        2
      );
      setStatus(
        $("recStatus"),
        text ? "Recognition finished — edit the draft if needed, then save correction." : "Finished, but no usable text — try another crop.",
        text ? "ok" : "warn"
      );
    } catch (err) {
      const aborted = err && err.name === "AbortError";
      setStatus(
        $("recStatus"),
        aborted
          ? "Timed out after 3 minutes. Restart the server with --preload-model and retry a smaller/single-line image."
          : "Error: " + (err.message || String(err)),
        "warn"
      );
      $("recText").value = "";
    } finally {
      clearTimeout(timer);
      setRecognizing(false);
    }
  }

  $("btnRecognize").addEventListener("click", () => runRecognize());

  $("btnCopyDraft").addEventListener("click", async () => {
    const text = $("recText").value.trim();
    if (!text) {
      setStatus($("recStatus"), "Nothing to copy.", "warn");
      return;
    }
    try {
      await navigator.clipboard.writeText(text);
      setStatus($("recStatus"), "Draft copied.", "ok");
    } catch {
      $("recText").select();
      setStatus($("recStatus"), "Select-all ready — press Ctrl+C.", "");
    }
  });

  $("btnSaveCorrection").addEventListener("click", async () => {
    const corrected = $("recText").value.trim();
    if (!corrected) {
      setStatus($("recStatus"), "Edit or recognize a draft before saving a correction.", "warn");
      return;
    }
    try {
      const data = await getJSON("/api/corrections", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          originalText: lastOriginalText,
          correctedText: corrected,
          filename: lastFilename,
          checkpoint: lastCheckpoint,
        }),
      });
      setStatus($("recStatus"), "Correction saved locally as " + data.record.id, "ok");
      refreshRecords();
    } catch (err) {
      setStatus($("recStatus"), "Save failed: " + err.message, "warn");
    }
  });

  let legObjectUrl = null;
  $("legFile").addEventListener("change", () => {
    const f = $("legFile").files && $("legFile").files[0];
    const box = $("legPreview");
    if (legObjectUrl) URL.revokeObjectURL(legObjectUrl);
    if (!f) {
      box.innerHTML = '<p class="muted">Preview appears here.</p>';
      return;
    }
    legObjectUrl = URL.createObjectURL(f);
    box.innerHTML = "";
    const img = document.createElement("img");
    img.alt = "Paper preview";
    img.src = legObjectUrl;
    box.appendChild(img);
  });

  $("btnLegacySave").addEventListener("click", async () => {
    const f = $("legFile").files && $("legFile").files[0];
    if (!f) {
      $("legResult").textContent = "Choose a paper image first.";
      return;
    }
    const fd = new FormData();
    fd.append("image", f, f.name);
    fd.append("doctor_id", $("legDoctor").value.trim());
    fd.append("patient_id", $("legPatient").value.trim());
    fd.append("note", $("legNote").value.trim());
    try {
      const data = await getJSON("/api/legacy/save", { method: "POST", body: fd });
      $("legResult").textContent = JSON.stringify(data.record, null, 2);
      refreshRecords();
    } catch (err) {
      $("legResult").textContent = "Error: " + err.message;
    }
  });

  $("btnLegacyRecognize").addEventListener("click", () => {
    const f = $("legFile").files && $("legFile").files[0];
    if (!f) {
      $("legResult").textContent = "Choose a paper image first.";
      return;
    }
    const dt = new DataTransfer();
    dt.items.add(f);
    $("recFile").files = dt.files;
    showPreview(f);
    $("recForceSingle").checked = false;
    switchTab("recognize");
    runRecognize(f);
  });

  async function refreshRecords() {
    const list = $("recordList");
    try {
      const data = await getJSON("/api/records");
      list.innerHTML = "";
      if (!data.records || !data.records.length) {
        list.innerHTML = '<p class="muted">No local testing records yet.</p>';
        return;
      }
      data.records.forEach((rec) => {
        const row = document.createElement("div");
        row.className = "record-item";
        const left = document.createElement("div");
        const summary =
          rec.type === "transcription_correction"
            ? escapeHtml((rec.correctedText || "").slice(0, 80))
            : `<span class="muted">${escapeHtml(rec.doctorId || "")} · ${escapeHtml(
                rec.patientId || rec.patientName || ""
              )}</span>`;
        left.innerHTML =
          `<strong>${escapeHtml(rec.type || "record")}</strong> · ${escapeHtml(rec.id)}<br/>` +
          `<span class="muted">${escapeHtml(rec.createdAt || "")}</span><br/>` +
          summary;
        const actions = document.createElement("div");
        const btnOpen = document.createElement("button");
        btnOpen.type = "button";
        btnOpen.className = "btn";
        btnOpen.textContent = rec.type === "transcription_correction" ? "View" : "Open";
        btnOpen.addEventListener("click", () => openRecordOriginal(rec));
        actions.appendChild(btnOpen);
        row.appendChild(left);
        row.appendChild(actions);
        list.appendChild(row);
      });
    } catch (err) {
      list.innerHTML = `<p class="warn">${escapeHtml(err.message)}</p>`;
    }
  }

  async function openRecordOriginal(rec) {
    const box = $("recordPreview");
    if (!box) return;
    box.classList.remove("hidden");
    if (rec.type === "transcription_correction") {
      box.innerHTML =
        `<p><strong>Correction</strong> · ${escapeHtml(rec.id)}</p>` +
        `<p class="muted">Original</p><pre class="mono">${escapeHtml(rec.originalText || "")}</pre>` +
        `<p class="muted">Corrected</p><pre class="mono">${escapeHtml(rec.correctedText || "")}</pre>`;
      return;
    }
    box.innerHTML = "<p class='muted'>Loading…</p>";
    try {
      const imgUrl = "/api/records/" + encodeURIComponent(rec.id) + "/file?kind=png";
      box.innerHTML =
        `<p><strong>${escapeHtml(rec.type || "")}</strong> · ${escapeHtml(rec.id)}</p>` +
        `<img alt="Original record" src="${imgUrl}" style="max-width:100%;border:1px solid #e5e7eb;border-radius:8px" />`;
    } catch (err) {
      box.innerHTML = `<p class="warn">${escapeHtml(err.message)}</p>`;
    }
  }

  function escapeHtml(s) {
    return String(s || "")
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;");
  }

  $("btnRefreshRecords").addEventListener("click", refreshRecords);
  boot();
  refreshRecords();
})();
