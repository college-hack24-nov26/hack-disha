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
  let lastSampleId = "";
  let lastSampleExpected = "";
  let lastSampleLabel = "";
  let lastSampleSource = "";
  let selectedSampleToken = 0;

  function normalizeCompare(s) {
    return (s || "").replace(/\s+/g, " ").trim();
  }

  function clearSampleSelection() {
    lastSampleId = "";
    lastSampleExpected = "";
    lastSampleLabel = "";
    lastSampleSource = "";
    document.querySelectorAll(".sample-card.selected").forEach((el) => {
      el.classList.remove("selected");
    });
    if ($("recSampleId")) $("recSampleId").textContent = "";
    if ($("recReference")) $("recReference").textContent = "—";
    if ($("recPrediction")) $("recPrediction").textContent = "—";
    if ($("recDiffRow")) $("recDiffRow").classList.add("hidden");
    if ($("recConfidenceRow")) $("recConfidenceRow").classList.add("hidden");
  }

  function renderArchiveSummary(record) {
    const view = $("legResultView");
    const raw = $("legResult");
    if (!view || !raw) return;
    if (!record) {
      view.className = "archive-empty";
      view.innerHTML =
        `<p class="empty-title">No archive save yet</p>` +
        `<p class="muted">After you save, clinician, patient, and file details show here in a clear summary.</p>`;
      raw.classList.add("hidden");
      raw.textContent = "";
      return;
    }
    view.className = "archive-summary";
    view.innerHTML =
      `<div class="summary-row"><dt>Status</dt><dd>Original saved</dd></div>` +
      `<div class="summary-row"><dt>Record ID</dt><dd>${escapeHtml(record.id || "")}</dd></div>` +
      `<div class="summary-row"><dt>Clinician</dt><dd>${escapeHtml(
        [record.doctorName, record.doctorId].filter(Boolean).join(" · ")
      )}</dd></div>` +
      `<div class="summary-row"><dt>Patient</dt><dd>${escapeHtml(
        [record.patientName, record.patientId].filter(Boolean).join(" · ")
      )}</dd></div>` +
      `<div class="summary-row"><dt>File</dt><dd>${escapeHtml(record.originalFilename || "")}</dd></div>` +
      `<div class="summary-row"><dt>Note</dt><dd>${escapeHtml(record.note || "—")}</dd></div>` +
      `<div class="summary-row"><dt>Storage</dt><dd>${escapeHtml(record.storageNote || "Local workspace only")}</dd></div>`;
    raw.classList.remove("hidden");
    raw.textContent = JSON.stringify(record, null, 2);
  }

  async function boot() {
    try {
      const health = await getJSON("/api/health");
      const modelReady = health.model_loaded ? "Model ready" : "Model loads on first recognize";
      const acc =
        health.heldout_char_accuracy_pct != null
          ? health.heldout_char_accuracy_pct
          : health.heldout_char_accuracy_pct_base;
      const wer =
        health.heldout_mean_wer_pct != null ? health.heldout_mean_wer_pct : null;
      const wordAcc =
        wer != null ? Math.round((100 - Number(wer)) * 100) / 100 : acc;
      $("healthBanner").innerHTML =
        `<span class="ok">Online</span> · ${modelReady} · held-out WER ` +
        `<strong>${wer != null ? wer : "—"}%</strong>` +
        (wordAcc != null ? ` · word acc <strong>${wordAcc}%</strong>` : "");
      if ($("livePillText")) {
        $("livePillText").textContent =
          wer != null ? `Live · WER ${wer}%` : `Live · ${acc}% held-out`;
      }
      const ck = await getJSON("/api/checkpoint");
      if ($("checkpointInfo")) {
        $("checkpointInfo").textContent =
          "Active model: " + (ck.selected || "") + " · " + (ck.reason || "");
      }
      if ($("metricActive")) {
        $("metricActive").textContent = (wordAcc != null ? wordAcc : acc) + "%";
      }
      if ($("metricWer")) {
        $("metricWer").textContent = wer != null ? wer + "%" : "—";
      }
    } catch (err) {
      $("healthBanner").innerHTML =
        `<span class="warn">Service not reachable.</span> Start with ` +
        `<code>.\\.venv\\Scripts\\python.exe scripts\\run_judge_demo_server.py --port 8766 --preload-model</code>`;
      if ($("livePillText")) {
        $("livePillText").textContent = "Offline";
      }
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
      window.__samplePackSource = data.pack_source || "iam";
      const head = document.createElement("div");
      head.className = "sample-pack-head";
      const verifiedN =
        data.verified_count != null ? data.verified_count : data.samples.length;
      head.innerHTML =
        `<strong>${escapeHtml(data.pack_title || "Verified IAM samples")}</strong>` +
        `<span class="badge-verified">${escapeHtml(String(verifiedN))} with reference · Hugging Face</span>`;
      row.appendChild(head);
      if (data.story) {
        const story = document.createElement("p");
        story.className = "sample-story muted small";
        story.textContent = data.story;
        row.appendChild(story);
      }
      const grid = document.createElement("div");
      grid.className = "sample-showcase";
      data.samples.forEach((s) => {
        const card = document.createElement("button");
        card.type = "button";
        card.className = "sample-card";
        card.dataset.sampleId = s.sample_id || s.id;
        card.title = s.expected_text || s.id;
        const img = document.createElement("img");
        img.src = s.url;
        img.alt = s.label || s.id;
        img.loading = "lazy";
        const meta = document.createElement("div");
        meta.className = "sample-card-meta";
        const ref = s.expected_text || s.verified_reference || "";
        const verifiedTag = ref
          ? `<span class="sample-card-tag">Verified reference</span>`
          : `<span class="sample-card-tag unverified">No reference</span>`;
        meta.innerHTML =
          `<span class="sample-card-label">${escapeHtml(s.label || s.id)}</span>` +
          `<span class="sample-card-text">${escapeHtml(ref.slice(0, 72))}${
            ref.length > 72 ? "..." : ""
          }</span>` +
          verifiedTag;
        card.appendChild(img);
        card.appendChild(meta);
        card.addEventListener("click", () => useSample(s, data.pack_source || "iam"));
        grid.appendChild(card);
      });
      row.appendChild(grid);
    } catch (err) {
      row.innerHTML = '<p class="muted small">Samples unavailable.</p>';
    }
  }

  async function useSample(sample, packSource) {
    const token = ++selectedSampleToken;
    try {
      const res = await fetch(sample.url);
      if (!res.ok) throw new Error("Could not load sample");
      const blob = await res.blob();
      if (token !== selectedSampleToken) return;
      const fileName = sample.id || (sample.sample_id || "sample") + ".png";
      const file = new File([blob], fileName, { type: "image/png" });
      const dt = new DataTransfer();
      dt.items.add(file);
      $("recFile").files = dt.files;
      showPreview(file);

      document.querySelectorAll(".sample-card.selected").forEach((el) => {
        el.classList.remove("selected");
      });
      const sid = sample.sample_id || PathStem(sample.id);
      document.querySelectorAll(".sample-card").forEach((el) => {
        if (el.dataset.sampleId === sid) el.classList.add("selected");
      });

      lastSampleId = sid;
      lastSampleExpected = (sample.expected_text || sample.verified_reference || "").trim();
      lastSampleLabel = sample.label || sample.id;
      lastSampleSource = packSource || window.__samplePackSource || "iam";

      // Clear previous prediction/reference immediately so samples never mix.
      $("recText").value = "";
      lastOriginalText = "";
      if ($("recPrediction")) $("recPrediction").textContent = "—";
      if ($("recReference")) {
        $("recReference").textContent = lastSampleExpected || "—";
      }
      if ($("recSampleId")) {
        $("recSampleId").textContent = lastSampleId ? " · " + lastSampleId : "";
      }
      if ($("recDiffRow")) $("recDiffRow").classList.add("hidden");
      if ($("recConfidenceRow")) $("recConfidenceRow").classList.add("hidden");
      if ($("recLines")) {
        $("recLines").innerHTML = "";
        $("recLines").classList.add("hidden");
      }
      if ($("recMeta")) $("recMeta").textContent = "";
      if ($("recNote")) $("recNote").textContent = "";
      if ($("recCerBox")) $("recCerBox").classList.add("hidden");

      setStatus(
        $("recStatus"),
        lastSampleExpected
          ? "Sample " + lastSampleId + " loaded with verified reference."
          : "Sample " + lastSampleId + " loaded (no verified reference).",
        "ok"
      );
      if ($("recAuto").checked) runRecognize(file, token);
    } catch (err) {
      setStatus($("recStatus"), "Sample error: " + err.message, "warn");
    }
  }

  function PathStem(name) {
    const n = String(name || "");
    const i = n.lastIndexOf(".");
    return i > 0 ? n.slice(0, i) : n;
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
    const n = String(f.name || "").toLowerCase();
    if (!n.startsWith("iam") && !n.startsWith("online_")) {
      clearSampleSelection();
    }
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

  function renderCompactResult(prediction, reference, lines, sampleId) {
    if ($("recPrediction")) {
      $("recPrediction").textContent = prediction || "—";
    }
    if ($("recReference")) {
      $("recReference").textContent = reference || "—";
    }
    if ($("recSampleId")) {
      $("recSampleId").textContent = sampleId ? " · " + sampleId : "";
    }
    const diffRow = $("recDiffRow");
    const diffEl = $("recDiff");
    if (diffRow && diffEl) {
      if (reference && prediction) {
        const matched =
          normalizeCompare(prediction) === normalizeCompare(reference);
        if (matched) {
          diffRow.classList.remove("hidden");
          diffEl.className = "recog-v ok";
          diffEl.textContent = "Prediction matches verified reference";
        } else {
          diffRow.classList.remove("hidden");
          diffEl.className = "recog-v warn";
          diffEl.textContent =
            "Prediction differs from verified reference (shown above separately)";
        }
      } else {
        diffRow.classList.add("hidden");
      }
    }
    const confRow = $("recConfidenceRow");
    const confEl = $("recConfidence");
    if (confRow && confEl) {
      const low = (lines || []).filter((ln) => ln.status === "low_confidence");
      const ok = (lines || []).filter((ln) => ln.status === "ok");
      // Confidence reflects model output only — never the verified reference.
      if (low.length) {
        confRow.classList.remove("hidden");
        confEl.className = "recog-v warn";
        confEl.textContent =
          low.length +
          " low-confidence model line(s)" +
          (ok.length ? " · " + ok.length + " ok" : "");
      } else if ((lines || []).length) {
        confRow.classList.remove("hidden");
        confEl.className = "recog-v ok";
        confEl.textContent = "Model lines reported ok";
      } else {
        confRow.classList.add("hidden");
      }
    }
  }

  async function runRecognize(file, selectionToken) {
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

    const requestSampleId = lastSampleId;
    const requestExpected = lastSampleExpected;

    $("recText").value = "";
    $("recMeta").textContent = "";
    $("recTime").textContent = "";
    $("recNote").textContent = "";
    if ($("recPrediction")) $("recPrediction").textContent = "—";
    if ($("recDiffRow")) $("recDiffRow").classList.add("hidden");
    if ($("recConfidenceRow")) $("recConfidenceRow").classList.add("hidden");
    lastOriginalText = "";
    lastFilename = f.name;
    lastCheckpoint = "";
    setRecognizing(true, "Uploading image…");
    setStatus($("recStatus"), "Recognizing…", "");

    const fd = new FormData();
    fd.append("image", f, f.name);
    fd.append("force_single_line", $("recForceSingle").checked ? "true" : "false");
    fd.append(
      "force_printed",
      $("recForcePrinted") && $("recForcePrinted").checked ? "true" : "false"
    );
    fd.append(
      "auto_printed",
      !$("recAutoPrinted") || $("recAutoPrinted").checked ? "true" : "false"
    );
    const fname = String(f.name || "").toLowerCase();
    const useIam =
      lastSampleSource === "iam" ||
      lastSampleSource === "online_iam" ||
      fname.startsWith("iam") ||
      fname.startsWith("online_");
    if (useIam) {
      fd.append("sample_source", "iam");
      fd.set("force_single_line", "true");
      fd.set("auto_printed", "false");
      fd.set("force_printed", "false");
    }
    if (requestSampleId) {
      fd.append("sample_id", requestSampleId);
    }

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

      // Drop stale responses if the user clicked another sample mid-request.
      if (
        selectionToken != null &&
        selectionToken !== selectedSampleToken
      ) {
        return;
      }
      if (requestSampleId && lastSampleId && requestSampleId !== lastSampleId) {
        return;
      }

      const lines = (data.recognition && data.recognition.lines) || [];
      // Prefer combined draft; if sanitizer cleared it, still surface raw line texts
      // so the UI never substitutes the verified reference for the model output.
      let text = (data.text || "").trim();
      if (!text && lines.length) {
        text = lines
          .map((ln) => (ln.text || "").trim())
          .filter(Boolean)
          .join("\n")
          .trim();
      }
      lastOriginalText = text;
      lastCheckpoint = data.checkpoint || "";
      $("recText").value = text;
      $("recTime").textContent =
        data.inference_time_s != null ? `(${data.inference_time_s}s)` : "";
      $("recNote").textContent =
        (data.recognition && data.recognition.segmentation_note) || data.storage_note || "";
      // Keep per-line panel compact: only open when model reported low-confidence lines.
      const lowLines = lines.filter((ln) => ln.status === "low_confidence");
      if (lowLines.length) {
        renderLines(lowLines);
      } else if ($("recLines")) {
        $("recLines").innerHTML = "";
        $("recLines").classList.add("hidden");
      }
      renderCompactResult(text, requestExpected, lines, requestSampleId || lastSampleId);
      $("recMeta").textContent = JSON.stringify(
        {
          sample_id: requestSampleId || null,
          verified_reference: requestExpected || null,
          model_prediction: text,
          checkpoint: data.checkpoint,
          selection: data.selection,
          mode: data.recognition && data.recognition.mode,
          n_lines: data.recognition && data.recognition.n_lines_detected,
          lines: lines,
        },
        null,
        2
      );
      const contentType =
        (data.recognition && data.recognition.content_type) || "handwriting";
      const selSrc = (data.selection && data.selection.source) || "";
      let matchNote = "";
      if (requestExpected) {
        matchNote =
          normalizeCompare(text) === normalizeCompare(requestExpected)
            ? " · prediction matches reference"
            : " · prediction differs from reference";
      }
      setStatus(
        $("recStatus"),
        text
          ? `Done (${contentType}` +
              (selSrc ? ` · ${selSrc}` : "") +
              ")" +
              matchNote +
              "."
          : "No model prediction — check options or try a tighter crop.",
        text ? "ok" : "warn"
      );
    } catch (err) {
      if (
        selectionToken != null &&
        selectionToken !== selectedSampleToken
      ) {
        return;
      }
      const aborted = err && err.name === "AbortError";
      setStatus(
        $("recStatus"),
        aborted
          ? "Timed out after 3 minutes. Restart the server with --preload-model and retry a smaller/single-line image."
          : "Error: " + (err.message || String(err)),
        "warn"
      );
      $("recText").value = "";
      if ($("recPrediction")) $("recPrediction").textContent = "—";
    } finally {
      clearTimeout(timer);
      setRecognizing(false);
    }
  }

  $("btnRecognize").addEventListener("click", () => runRecognize());

  $("btnCopyDraft").addEventListener("click", async () => {
    const text =
      ($("recPrediction") && $("recPrediction").textContent.trim()) ||
      $("recText").value.trim();
    if (!text || text === "—") {
      setStatus($("recStatus"), "Nothing to copy.", "warn");
      return;
    }
    try {
      await navigator.clipboard.writeText(text);
      setStatus($("recStatus"), "Model prediction copied.", "ok");
    } catch {
      $("recText").select();
      setStatus($("recStatus"), "Select-all ready — press Ctrl+C.", "");
    }
  });

  if ($("btnCopyReference")) {
    $("btnCopyReference").addEventListener("click", async () => {
      const text = (lastSampleExpected || "").trim();
      if (!text) {
        setStatus($("recStatus"), "No verified reference for this selection.", "warn");
        return;
      }
      try {
        await navigator.clipboard.writeText(text);
        setStatus($("recStatus"), "Verified reference copied.", "ok");
      } catch {
        setStatus($("recStatus"), "Copy failed — select reference text manually.", "warn");
      }
    });
  }

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
      const m = (data.record && data.record.metrics) || {};
      const cerPct = m.cer_pct != null ? m.cer_pct : null;
      const werPct = m.wer_pct != null ? m.wer_pct : null;
      const werNote =
        werPct != null
          ? ` WER ${werPct}%` +
            (cerPct != null ? ` · CER ${cerPct}%` : "") +
            (m.wer_worse_than_cer ? " — word errors dominate" : "")
          : "";
      setStatus($("recStatus"), "Saved " + data.record.id + "." + werNote, "ok");
      if ($("recCerBox")) {
        $("recCerBox").classList.remove("hidden");
        $("recCerBox").innerHTML =
          werPct != null
            ? `<strong>WER ${werPct}%</strong>` +
              (cerPct != null ? ` · CER ${cerPct}%` : "") +
              ` · word acc ${
                m.word_accuracy_pct != null ? m.word_accuracy_pct : "—"
              }%` +
              `<span class="muted"> · primary = WER</span>`
            : "";
      }
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
      renderArchiveSummary(null);
      if ($("legResultView")) {
        $("legResultView").innerHTML =
          `<p class="empty-title">Choose a paper image first</p>` +
          `<p class="muted">Select a scan or photo, then fill clinician and patient details.</p>`;
      }
      return;
    }
    const fd = new FormData();
    fd.append("image", f, f.name);
    fd.append("doctor_id", $("legDoctor").value.trim());
    fd.append("patient_id", $("legPatient").value.trim());
    fd.append("doctor_name", ($("legDoctorName") && $("legDoctorName").value.trim()) || "");
    fd.append("patient_name", ($("legPatientName") && $("legPatientName").value.trim()) || "");
    fd.append("note", $("legNote").value.trim());
    try {
      const data = await getJSON("/api/legacy/save", { method: "POST", body: fd });
      renderArchiveSummary(data.record);
      refreshRecords();
    } catch (err) {
      if ($("legResultView")) {
        $("legResultView").className = "archive-empty";
        $("legResultView").innerHTML =
          `<p class="empty-title">Save failed</p>` +
          `<p class="warn">${escapeHtml(err.message)}</p>`;
      }
      if ($("legResult")) {
        $("legResult").classList.add("hidden");
        $("legResult").textContent = "";
      }
    }
  });

  $("btnLegacyRecognize").addEventListener("click", () => {
    const f = $("legFile").files && $("legFile").files[0];
    if (!f) {
      if ($("legResultView")) {
        $("legResultView").className = "archive-empty";
        $("legResultView").innerHTML =
          `<p class="empty-title">Choose a paper image first</p>` +
          `<p class="muted">Then use Recognize this image to open the Recognition tab.</p>`;
      }
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
        const clinicianLabel = [rec.doctorName, rec.doctorId].filter(Boolean).join(" · ");
        const patientLabel = [rec.patientName, rec.patientId].filter(Boolean).join(" · ");
        const typeLabel =
          rec.type === "legacy_paper"
            ? "Paper archive"
            : rec.type === "transcription_correction"
              ? "Correction"
              : rec.type === "digital_rx"
                ? "Prescription pad"
                : rec.type || "Record";
        const summary =
          rec.type === "transcription_correction"
            ? escapeHtml((rec.correctedText || "").slice(0, 80))
            : `<span class="muted">Clinician: ${escapeHtml(clinicianLabel || "—")} · Patient: ${escapeHtml(
                patientLabel || "—"
              )}</span>`;
        left.innerHTML =
          `<strong>${escapeHtml(typeLabel)}</strong> · ${escapeHtml(rec.id)}<br/>` +
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

  function renderLines(lines) {
    const box = $("recLines");
    if (!box) return;
    box.innerHTML = "";
    if (!lines || !lines.length) {
      box.classList.add("hidden");
      return;
    }
    box.classList.remove("hidden");
    const title = document.createElement("p");
    title.className = "muted small";
    title.textContent = "Low-confidence model lines:";
    box.appendChild(title);
    lines.forEach((ln) => {
      const row = document.createElement("div");
      const st = ln.status || "unknown";
      row.className = "rec-line-item " + st;
      const shown = (ln.text || "").trim() || "(empty)";
      row.innerHTML =
        `<span class="tag">L${escapeHtml(ln.line_index)} · ${escapeHtml(st)}</span>` +
        `<span>${escapeHtml(shown)}</span>`;
      box.appendChild(row);
    });
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
