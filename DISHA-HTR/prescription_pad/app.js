/**
 * DISHA digital handwriting prescription pad (testing workspace).
 * Pointer Events for mouse / touch / future stylus.
 * Preserves stroke coordinates, order, colour, width, optional pressure/tilt.
 * Does NOT claim mouse pressure is real stylus pressure.
 * Local persistence only — not the production DISHA database.
 */
(function () {
  "use strict";

  const STORAGE_KEY = "disha_rx_pad_records_v1";
  const SCHEMA_VERSION = 1;

  /** Synthetic clinicians / patients for testing only. */
  const TEST_DOCTORS = [
    {
      id: "clinician_mehta",
      name: "Dr. A. Mehta",
      qualifications: "MBBS, MD (Medicine)",
      registration: "TEST-REG-1042",
      department: "General Medicine",
      address: "12 Test Road, Example City 560001",
      contact: "ext. 210",
    },
    {
      id: "clinician_rao",
      name: "Dr. S. Rao",
      qualifications: "MBBS, MS (Ortho)",
      registration: "TEST-REG-2218",
      department: "Orthopaedics",
      address: "12 Test Road, Example City 560001",
      contact: "ext. 334",
    },
  ];

  const TEST_PATIENTS = [
    { name: "Test Patient One", id: "TEST-P-001", age: "14" },
    { name: "Test Patient Two", id: "TEST-P-002", age: "58" },
  ];

  const canvas = document.getElementById("rxCanvas");
  const ctx = canvas.getContext("2d", { willReadFrequently: true });
  const wrap = document.getElementById("canvasWrap");
  const penColor = document.getElementById("penColor");
  const penWidth = document.getElementById("penWidth");
  const penWidthLabel = document.getElementById("penWidthLabel");
  const doctorSelect = document.getElementById("doctorSelect");
  const doctorMeta = document.getElementById("doctorMeta");
  const patientName = document.getElementById("patientName");
  const patientId = document.getElementById("patientId");
  const patientAge = document.getElementById("patientAge");
  const visitDate = document.getElementById("visitDate");
  const draftDiagnosis = document.getElementById("draftDiagnosis");
  const draftMedicines = document.getElementById("draftMedicines");
  const draftDosage = document.getElementById("draftDosage");
  const draftFrequency = document.getElementById("draftFrequency");
  const draftDuration = document.getElementById("draftDuration");
  const draftInstructions = document.getElementById("draftInstructions");
  const draftTranscription = document.getElementById("draftTranscription");

  /** @type {{ id: string, points: object[], color: string, width: number, pointerType: string, startedAt: string, endedAt: string|null }[]} */
  let strokes = [];
  /** @type {typeof strokes[0] | null} */
  let current = null;
  let drawing = false;
  let pointerId = null;
  /** Prior saved record id when creating a new version (append-only; never overwrites). */
  let versionOfId = null;

  function todayISODate() {
    const d = new Date();
    const m = String(d.getMonth() + 1).padStart(2, "0");
    const day = String(d.getDate()).padStart(2, "0");
    return `${d.getFullYear()}-${m}-${day}`;
  }

  function selectedDoctor() {
    return TEST_DOCTORS.find((d) => d.id === doctorSelect.value) || TEST_DOCTORS[0];
  }

  function fillDoctors() {
    doctorSelect.innerHTML = "";
    TEST_DOCTORS.forEach((d) => {
      const opt = document.createElement("option");
      opt.value = d.id;
      opt.textContent = d.name;
      doctorSelect.appendChild(opt);
    });
    updateDoctorMeta();
  }

  function updateDoctorMeta() {
    const d = selectedDoctor();
    doctorMeta.textContent = [
      d.qualifications,
      `Reg. No.: ${d.registration}`,
      d.address || "",
      `${d.department} · ${d.contact}`,
    ]
      .filter(Boolean)
      .join("\n");
  }

  function fillTestPatient(index) {
    const p = TEST_PATIENTS[index % TEST_PATIENTS.length];
    patientName.value = p.name;
    patientId.value = p.id;
    patientAge.value = p.age;
    visitDate.value = todayISODate();
    syncFooter();
  }

  function syncFooter() {
    document.getElementById("footerPatient").textContent = patientName.value || "—";
    document.getElementById("footerId").textContent = patientId.value || "—";
    document.getElementById("footerAge").textContent = patientAge.value || "—";
    document.getElementById("footerDate").textContent = visitDate.value || "—";
  }

  function cssSize() {
    const rect = canvas.getBoundingClientRect();
    return { w: rect.width, h: rect.height, left: rect.left, top: rect.top };
  }

  function resizeCanvas() {
    const snapshot = canvas.toDataURL("image/png");
    const ratio = Math.max(window.devicePixelRatio || 1, 1);
    const { w, h } = cssSize();
    // Keep logical drawing size stable-ish; backstore for sharpness
    const cssW = Math.max(320, Math.floor(w));
    const cssH = Math.max(280, Math.floor(cssW * 0.56));
    canvas.style.width = "100%";
    canvas.width = Math.floor(cssW * ratio);
    canvas.height = Math.floor(cssH * ratio);
    ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
    ctx.lineCap = "round";
    ctx.lineJoin = "round";
    // restore ink after resize
    if (strokes.length) {
      redrawAll();
    } else if (snapshot && snapshot !== "data:,") {
      const img = new Image();
      img.onload = () => ctx.drawImage(img, 0, 0, cssW, cssH);
      img.src = snapshot;
    }
  }

  function pointerToPoint(evt) {
    const rect = canvas.getBoundingClientRect();
    const ratio = window.devicePixelRatio || 1;
    // Convert to CSS-pixel space used with ctx.setTransform(ratio, ...)
    const px = ((evt.clientX - rect.left) * (canvas.width / rect.width)) / ratio;
    const py = ((evt.clientY - rect.top) * (canvas.height / rect.height)) / ratio;

    const point = {
      x: Number(px.toFixed(3)),
      y: Number(py.toFixed(3)),
      t: Date.now(),
    };

    // Only record hardware-reported pressure / tilt (stylus). Mouse often reports 0.5 — do not claim it.
    if (evt.pointerType === "pen") {
      if (typeof evt.pressure === "number" && evt.pressure > 0) {
        point.pressure = evt.pressure;
      }
      if (typeof evt.tiltX === "number") point.tiltX = evt.tiltX;
      if (typeof evt.tiltY === "number") point.tiltY = evt.tiltY;
    } else if (
      evt.pointerType === "touch" &&
      typeof evt.pressure === "number" &&
      evt.pressure > 0 &&
      evt.pressure !== 0.5
    ) {
      point.pressure = evt.pressure;
    }
    // Mouse: intentionally omit pressure so JSON does not imply stylus fidelity.

    return point;
  }

  function setDrawingLock(on) {
    document.body.classList.toggle("drawing-lock", on);
  }

  function beginStroke(evt) {
    if (drawing) return;
    drawing = true;
    pointerId = evt.pointerId;
    canvas.setPointerCapture(pointerId);
    setDrawingLock(true);
    const point = pointerToPoint(evt);
    current = {
      id: `s_${Date.now()}_${Math.random().toString(36).slice(2, 8)}`,
      color: penColor.value,
      width: Number(penWidth.value),
      pointerType: evt.pointerType || "unknown",
      startedAt: new Date().toISOString(),
      endedAt: null,
      points: [point],
    };
    strokes.push(current);
    ctx.strokeStyle = current.color;
    ctx.lineWidth = current.width;
    ctx.beginPath();
    ctx.moveTo(point.x, point.y);
  }

  function extendStroke(evt) {
    if (!drawing || evt.pointerId !== pointerId || !current) return;
    const point = pointerToPoint(evt);
    const prev = current.points[current.points.length - 1];
    current.points.push(point);
    ctx.strokeStyle = current.color;
    ctx.lineWidth = current.width;
    ctx.beginPath();
    ctx.moveTo(prev.x, prev.y);
    ctx.lineTo(point.x, point.y);
    ctx.stroke();
  }

  function endStroke(evt) {
    if (!drawing || (evt && evt.pointerId !== pointerId)) return;
    if (current) {
      current.endedAt = new Date().toISOString();
      if (current.points.length === 1) {
        // tap/dot
        const p = current.points[0];
        ctx.fillStyle = current.color;
        ctx.beginPath();
        ctx.arc(p.x, p.y, current.width / 2, 0, Math.PI * 2);
        ctx.fill();
      }
    }
    drawing = false;
    current = null;
    pointerId = null;
    setDrawingLock(false);
  }

  function redrawAll() {
    const ratio = window.devicePixelRatio || 1;
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
    ctx.lineCap = "round";
    ctx.lineJoin = "round";
    for (const stroke of strokes) {
      if (!stroke.points.length) continue;
      ctx.strokeStyle = stroke.color;
      ctx.lineWidth = stroke.width;
      ctx.beginPath();
      ctx.moveTo(stroke.points[0].x, stroke.points[0].y);
      for (let i = 1; i < stroke.points.length; i++) {
        ctx.lineTo(stroke.points[i].x, stroke.points[i].y);
      }
      if (stroke.points.length === 1) {
        ctx.fillStyle = stroke.color;
        ctx.beginPath();
        ctx.arc(stroke.points[0].x, stroke.points[0].y, stroke.width / 2, 0, Math.PI * 2);
        ctx.fill();
      } else {
        ctx.stroke();
      }
    }
  }

  function undo() {
    if (!strokes.length) return;
    strokes.pop();
    redrawAll();
  }

  function clearCanvas() {
    if (!strokes.length) return;
    const ok = window.confirm("Clear the entire prescription handwriting? This cannot be undone.");
    if (!ok) return;
    strokes = [];
    redrawAll();
  }

  function canvasLogicalSize() {
    const ratio = window.devicePixelRatio || 1;
    return {
      width: canvas.width / ratio,
      height: canvas.height / ratio,
      devicePixelRatio: ratio,
    };
  }

  function readClinicalDraft() {
    return {
      diagnosis: (draftDiagnosis && draftDiagnosis.value.trim()) || "",
      medicines: (draftMedicines && draftMedicines.value.trim()) || "",
      dosage: (draftDosage && draftDosage.value.trim()) || "",
      frequency: (draftFrequency && draftFrequency.value.trim()) || "",
      duration: (draftDuration && draftDuration.value.trim()) || "",
      instructions: (draftInstructions && draftInstructions.value.trim()) || "",
      transcription: (draftTranscription && draftTranscription.value.trim()) || "",
      note: "Draft layer only — editing these fields never rewrites stroke coordinates or the handwriting PNG.",
    };
  }

  function applyClinicalDraft(draft) {
    const d = draft || {};
    if (draftDiagnosis) draftDiagnosis.value = d.diagnosis || "";
    if (draftMedicines) draftMedicines.value = d.medicines || "";
    if (draftDosage) draftDosage.value = d.dosage || "";
    if (draftFrequency) draftFrequency.value = d.frequency || "";
    if (draftDuration) draftDuration.value = d.duration || "";
    if (draftInstructions) draftInstructions.value = d.instructions || "";
    if (draftTranscription) {
      draftTranscription.value = d.transcription || "";
    }
  }

  function buildStrokeDocument() {
    const doctor = selectedDoctor();
    const size = canvasLogicalSize();
    return {
      schemaVersion: SCHEMA_VERSION,
      recordType: "digital_handwriting_prescription",
      createdAt: new Date().toISOString(),
      doctor: { ...doctor },
      patient: {
        name: patientName.value.trim(),
        id: patientId.value.trim(),
        age: patientAge.value.trim(),
        visitDate: visitDate.value,
        note: "Synthetic patient fields only — not a real clinical record.",
      },
      clinicalDraft: readClinicalDraft(),
      canvas: size,
      inputNotes: {
        pressure: "Recorded only when pointerType is pen (or non-default touch). Mouse input does not provide real stylus pressure.",
        tilt: "Recorded only when the device reports tiltX/tiltY (typically stylus).",
        futureStylus: "Same Pointer Events path accepts a real tablet stylus without changing stroke schema.",
      },
      legacyDistinction: {
        legacyPaper: "Original uploaded image stored unchanged; OCR optional later; never replaces original.",
        digitalNew: "Original stroke JSON kept independently from rendered PNG; OCR/HTR must not mutate stroke source.",
      },
      strokes: strokes.map((s) => ({
        id: s.id,
        color: s.color,
        width: s.width,
        pointerType: s.pointerType,
        startedAt: s.startedAt,
        endedAt: s.endedAt,
        points: s.points,
      })),
    };
  }

  function restoreStrokesFromDocument(doc) {
    const incoming = (doc && doc.strokes) || [];
    strokes = incoming.map((s) => ({
      id: s.id || `s_${Math.random().toString(36).slice(2, 8)}`,
      color: s.color || "#1a1a1a",
      width: Number(s.width) || 3,
      pointerType: s.pointerType || "unknown",
      startedAt: s.startedAt || null,
      endedAt: s.endedAt || null,
      points: Array.isArray(s.points) ? s.points : [],
    }));
    redrawAll();
  }

  function downloadBlob(filename, blob) {
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    a.remove();
    setTimeout(() => URL.revokeObjectURL(url), 1500);
  }

  function exportPng(filename) {
    return new Promise((resolve) => {
      canvas.toBlob((blob) => {
        if (!blob) {
          resolve(null);
          return;
        }
        if (filename) downloadBlob(filename, blob);
        resolve(blob);
      }, "image/png");
    });
  }

  function exportJson(filename) {
    const doc = buildStrokeDocument();
    const blob = new Blob([JSON.stringify(doc, null, 2)], { type: "application/json" });
    if (filename) downloadBlob(filename, blob);
    return doc;
  }

  function loadRecords() {
    try {
      const raw = localStorage.getItem(STORAGE_KEY);
      if (!raw) return [];
      const parsed = JSON.parse(raw);
      return Array.isArray(parsed) ? parsed : [];
    } catch {
      return [];
    }
  }

  function persistRecords(records) {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(records));
  }

  async function savePrescription() {
    if (!strokes.length) {
      window.alert("Write something on the canvas before saving.");
      return;
    }
    if (!patientName.value.trim() || !patientId.value.trim()) {
      window.alert("Fill patient name and ID before saving.");
      return;
    }

    const stamp = new Date();
    const id = `rx_${stamp.getTime()}`;
    const base = `${id}_${selectedDoctor().id}_${patientId.value.trim().replace(/\s+/g, "_")}`;
    const strokeDoc = exportJson(`${base}_strokes.json`);
    const pngBlob = await exportPng(`${base}.png`);
    let pngDataUrl = canvas.toDataURL("image/png");
    if (pngBlob) {
      pngDataUrl = await new Promise((resolve) => {
        const reader = new FileReader();
        reader.onload = () => resolve(String(reader.result));
        reader.readAsDataURL(pngBlob);
      });
    }

    const clinicalDraft = readClinicalDraft();
    const record = {
      id,
      createdAt: stamp.toISOString(),
      doctorId: selectedDoctor().id,
      doctorName: selectedDoctor().name,
      patientName: patientName.value.trim(),
      patientId: patientId.value.trim(),
      patientAge: patientAge.value.trim(),
      visitDate: visitDate.value,
      clinicalDraft,
      transcriptionDraft: clinicalDraft.transcription,
      approvalStatus: "draft_unapproved",
      versionOf: versionOfId,
      storageNote: "Local testing only — not saved to the production DISHA database.",
      strokeDocument: strokeDoc,
      pngDataUrl,
      strokeCount: strokes.length,
    };

    const records = loadRecords();
    records.unshift(record); // never overwrite earlier entries — version history via append
    while (records.length > 20) records.pop();
    try {
      persistRecords(records);
    } catch (err) {
      window.alert(
        "Could not write localStorage (quota). PNG/JSON downloads still happened. " +
          String(err && err.message ? err.message : err)
      );
    }

    let serverMsg = "";
    try {
      const res = await fetch("/api/records/digital", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(record),
      });
      const text = await res.text();
      let data = null;
      try {
        data = JSON.parse(text);
      } catch {
        data = null;
      }
      if (res.ok && data && data.record) {
        record.serverId = data.record.id;
        serverMsg = `\nAlso written to local testing storage as ${data.record.id}.`;
      } else if (!res.ok) {
        serverMsg =
          "\nDisk API unavailable — browser localStorage + downloads still OK.";
      }
    } catch {
      serverMsg =
        "\nTesting server not reachable — browser localStorage + downloads still OK.";
    }

    versionOfId = record.serverId || record.id;
    renderSavedList();
    showPreview(record);
    window.alert(
      "Saved locally for testing (browser + downloaded files)." +
        serverMsg +
        "\nThis record was NOT saved to the production DISHA database.\n" +
        "Clinical draft fields are unapproved drafts and do not alter handwriting."
    );
  }

  function reopenRecord(rec) {
    if (!rec) return;
    if (rec.doctorId) {
      doctorSelect.value = rec.doctorId;
      updateDoctorMeta();
    }
    patientName.value = rec.patientName || "";
    patientId.value = rec.patientId || "";
    patientAge.value = rec.patientAge || "";
    visitDate.value = rec.visitDate || todayISODate();
    syncFooter();
    applyClinicalDraft(rec.clinicalDraft || { transcription: rec.transcriptionDraft });
    if (rec.strokeDocument) {
      restoreStrokesFromDocument(rec.strokeDocument);
    } else if (rec.pngDataUrl) {
      strokes = [];
      const img = new Image();
      img.onload = () => {
        const ratio = window.devicePixelRatio || 1;
        const cssW = canvas.width / ratio;
        const cssH = canvas.height / ratio;
        ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
        ctx.clearRect(0, 0, cssW, cssH);
        ctx.drawImage(img, 0, 0, cssW, cssH);
      };
      img.src = rec.pngDataUrl;
    }
    versionOfId = rec.serverId || rec.id || null;
    showPreview(rec);
    window.alert(
      "Reopened. Original strokes restored when stroke JSON was present.\n" +
        "Editing draft fields or saving creates a new version — prior records are kept."
    );
  }

  function renderSavedList() {
    const list = document.getElementById("savedList");
    const records = loadRecords();
    list.innerHTML = "";
    if (!records.length) {
      list.innerHTML = "<p class='saved-note'>No local testing records yet.</p>";
      return;
    }
    records.forEach((rec) => {
      const row = document.createElement("div");
      row.className = "saved-item";
      const left = document.createElement("div");
      left.innerHTML =
        `<strong>${escapeHtml(rec.doctorName)}</strong>` +
        `<span class="saved-sub">${escapeHtml(rec.patientName)} · ${escapeHtml(rec.patientId)}</span>`;
      const actions = document.createElement("div");
      actions.className = "saved-actions";
      const btnPreview = document.createElement("button");
      btnPreview.type = "button";
      btnPreview.className = "btn btn-sm";
      btnPreview.textContent = "Preview";
      btnPreview.addEventListener("click", () => showPreview(rec));
      const btnReopen = document.createElement("button");
      btnReopen.type = "button";
      btnReopen.className = "btn btn-primary btn-sm";
      btnReopen.textContent = "Reopen";
      btnReopen.addEventListener("click", () => reopenRecord(rec));
      actions.appendChild(btnPreview);
      actions.appendChild(btnReopen);
      row.appendChild(left);
      row.appendChild(actions);
      list.appendChild(row);
    });
  }

  function showPreview(rec) {
    const box = document.getElementById("preview");
    const img = document.getElementById("previewImg");
    const meta = document.getElementById("previewMeta");
    box.classList.remove("hidden");
    img.src = rec.pngDataUrl || "";
    meta.textContent = JSON.stringify(
      {
        id: rec.id,
        serverId: rec.serverId || null,
        createdAt: rec.createdAt,
        doctorId: rec.doctorId,
        doctorName: rec.doctorName,
        patientName: rec.patientName,
        patientId: rec.patientId,
        visitDate: rec.visitDate,
        strokeCount: rec.strokeCount,
        clinicalDraft: rec.clinicalDraft || null,
        approvalStatus: rec.approvalStatus || "draft_unapproved",
        versionOf: rec.versionOf || null,
        storageNote: rec.storageNote,
      },
      null,
      2
    );
  }

  function escapeHtml(s) {
    return String(s || "")
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;");
  }

  // Events
  canvas.addEventListener("pointerdown", (e) => {
    e.preventDefault();
    beginStroke(e);
  });
  canvas.addEventListener("pointermove", (e) => {
    if (!drawing) return;
    e.preventDefault();
    extendStroke(e);
  });
  canvas.addEventListener("pointerup", (e) => {
    e.preventDefault();
    endStroke(e);
  });
  canvas.addEventListener("pointercancel", (e) => endStroke(e));
  canvas.addEventListener("lostpointercapture", () => {
    if (drawing) endStroke({ pointerId });
  });

  // Prevent scroll/gesture while drawing on wrap
  wrap.addEventListener(
    "touchmove",
    (e) => {
      if (drawing) e.preventDefault();
    },
    { passive: false }
  );

  penWidth.addEventListener("input", () => {
    penWidthLabel.textContent = penWidth.value;
  });
  doctorSelect.addEventListener("change", updateDoctorMeta);
  [patientName, patientId, patientAge, visitDate].forEach((el) =>
    el.addEventListener("input", syncFooter)
  );

  async function recognizeCanvasDraft() {
    const status = document.getElementById("padRecStatus");
    if (!strokes.length) {
      if (status) status.textContent = "Write something before recognizing.";
      return;
    }
    if (status) status.textContent = "Recognizing handwriting…";
    const blob = await new Promise((resolve) => {
      canvas.toBlob((b) => resolve(b), "image/png");
    });
    if (!blob) {
      if (status) status.textContent = "Could not export canvas.";
      return;
    }
    const fd = new FormData();
    fd.append("image", blob, `pad_${Date.now()}.png`);
    fd.append("force_single_line", "false");
    try {
      const res = await fetch("/api/recognize", { method: "POST", body: fd });
      const text = await res.text();
      let data;
      try {
        data = JSON.parse(text);
      } catch {
        throw new Error(text.slice(0, 160) || res.statusText);
      }
      if (!res.ok) throw new Error(data.error || data.detail || res.statusText);
      const draft = (data.text || "").trim();
      draftTranscription.value = draft;
      if (status) {
        status.textContent = draft
          ? `Draft filled (${data.inference_time_s != null ? data.inference_time_s + "s" : "done"}). Review before saving.`
          : "No text returned — try larger writing or a clearer line.";
      }
    } catch (err) {
      if (status) {
        status.textContent =
          "Recognition unavailable (" +
          (err && err.message ? err.message : String(err)) +
          "). Start the testing server if needed.";
      }
    }
  }

  document.getElementById("btnUndo").addEventListener("click", undo);
  document.getElementById("btnClear").addEventListener("click", clearCanvas);
  document.getElementById("btnRecognizePad").addEventListener("click", () => {
    recognizeCanvasDraft().catch((err) => window.alert(String(err)));
  });
  document.getElementById("btnSave").addEventListener("click", () => {
    savePrescription().catch((err) => window.alert(String(err)));
  });
  document.getElementById("btnExportPng").addEventListener("click", () => {
    if (!strokes.length) {
      window.alert("Canvas is empty.");
      return;
    }
    exportPng(`disha_rx_${Date.now()}.png`);
  });
  document.getElementById("btnExportJson").addEventListener("click", () => {
    if (!strokes.length) {
      window.alert("Canvas is empty.");
      return;
    }
    exportJson(`disha_rx_${Date.now()}_strokes.json`);
  });
  document.getElementById("btnRefreshSaved").addEventListener("click", renderSavedList);

  window.addEventListener("resize", () => {
    // Debounce lightly
    clearTimeout(window.__rxResizeTimer);
    window.__rxResizeTimer = setTimeout(resizeCanvas, 120);
  });

  // Init
  fillDoctors();
  fillTestPatient(0);
  penWidthLabel.textContent = penWidth.value;
  resizeCanvas();
  renderSavedList();
})();
