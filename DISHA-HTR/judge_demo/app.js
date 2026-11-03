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

  async function boot() {
    try {
      const health = await getJSON("/api/health");
      const modelReady = health.model_loaded ? "model ready" : "model loads on first recognize";
      $("healthBanner").innerHTML =
        `<span class="ok">Service online</span> · ` +
        `testing workspace · ${modelReady} · held-out accuracy ` +
        `<strong>${health.heldout_char_accuracy_pct_base}%</strong>`;
      const ck = await getJSON("/api/checkpoint");
      $("checkpointInfo").textContent =
        "Model: " + (ck.selected || "") + " · comparisons: " + JSON.stringify(ck.comparisons || {});
      $("metricBase").textContent = (ck.comparisons && ck.comparisons.base != null ? ck.comparisons.base : 14.01) + "%";
    } catch (err) {
      $("healthBanner").innerHTML =
        `<span class="warn">Service not reachable.</span> Start with ` +
        `<code>.\\.venv\\Scripts\\python.exe scripts\\run_judge_demo_server.py --port 8766 --preload-model</code>`;
    }
  }

  let recObjectUrl = null;
  let recognizing = false;

  $("recFile").addEventListener("change", () => {
    const f = $("recFile").files && $("recFile").files[0];
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
    setStatus($("recStatus"), "Image ready. Click Recognize.", "ok");
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

  $("btnRecognize").addEventListener("click", async () => {
    if (recognizing) return;
    const f = $("recFile").files && $("recFile").files[0];
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
    setRecognizing(true, "Uploading image…");
    setStatus($("recStatus"), "Please wait — first run can take 30–90s while the model loads.", "");

    const fd = new FormData();
    fd.append("image", f, f.name);
    fd.append("force_single_line", $("recForceSingle").checked ? "true" : "false");

    const controller = new AbortController();
    const timeoutMs = 180000;
    const timer = setTimeout(() => controller.abort(), timeoutMs);

    try {
      setTimeout(() => {
        if (recognizing) {
          $("recProgressText").textContent = "Running recognition (model may still be loading)…";
        }
      }, 2500);

      const data = await getJSON("/api/recognize", {
        method: "POST",
        body: fd,
        signal: controller.signal,
      });

      const text = (data.text || "").trim();
      $("recText").value = text || "(No readable text returned. Try a clearer single-line crop, or uncheck single-line for full pages.)";
      $("recTime").textContent =
        data.inference_time_s != null ? `(${data.inference_time_s}s)` : "";
      $("recNote").textContent =
        (data.recognition && data.recognition.segmentation_note) || data.storage_note || "";
      $("recMeta").textContent = JSON.stringify(
        {
          checkpoint: data.checkpoint,
          mode: data.recognition && data.recognition.mode,
          n_lines: data.recognition && data.recognition.n_lines_detected,
          lines: (data.recognition && data.recognition.lines) || [],
        },
        null,
        2
      );
      setStatus(
        $("recStatus"),
        text ? "Recognition finished." : "Finished, but no usable text — try another crop.",
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
        left.innerHTML =
          `<strong>${escapeHtml(rec.type || "record")}</strong> · ${escapeHtml(rec.id)}<br/>` +
          `<span class="muted">${escapeHtml(rec.createdAt || "")} · ${escapeHtml(
            rec.doctorId || ""
          )} · ${escapeHtml(rec.patientId || rec.patientName || "")}</span>`;
        const actions = document.createElement("div");
        const btnOpen = document.createElement("button");
        btnOpen.type = "button";
        btnOpen.className = "btn";
        btnOpen.textContent = "Open";
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
