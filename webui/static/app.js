(() => {
  const fileInput = document.getElementById("fileInput");
  const drop = document.getElementById("drop");
  const dropHint = document.getElementById("dropHint");
  const preview = document.getElementById("preview");
  const generateBtn = document.getElementById("generateBtn");
  const consoleEl = document.getElementById("console");
  const stageLabel = document.getElementById("stageLabel");
  const pctLabel = document.getElementById("pctLabel");
  const barFill = document.getElementById("barFill");
  const viewer = document.getElementById("viewer");
  const downloadLink = document.getElementById("downloadLink");
  const healthEl = document.getElementById("health");
  const vramEl = document.getElementById("vram");
  const presetEl = document.getElementById("preset");
  const lowVramEl = document.getElementById("lowVram");
  const resolutionEl = document.getElementById("resolution");
  const stepsEl = document.getElementById("steps");
  const maxTokensEl = document.getElementById("maxTokens");
  const textureSizeEl = document.getElementById("textureSize");
  const decimationEl = document.getElementById("decimation");

  let selectedFile = null;
  let running = false;
  let previewUrl = null;
  let vramTimer = null;
  let applyingPreset = false;

  const IMAGE_EXT = /\.(png|jpe?g|webp|bmp|gif|tif{1,2})$/i;

  const PRESETS = {
    preview: {
      low_vram: true,
      resolution: 1024,
      steps: 8,
      max_tokens: 16384,
      texture_size: 1024,
      decimation: 200000,
    },
    balanced: {
      low_vram: true,
      resolution: 1024,
      steps: 12,
      max_tokens: 32768,
      texture_size: 2048,
      decimation: 500000,
    },
    max: {
      low_vram: false,
      resolution: 1536,
      steps: 12,
      max_tokens: 49152,
      texture_size: 4096,
      decimation: 1000000,
    },
  };

  function log(line) {
    consoleEl.textContent += (consoleEl.textContent ? "\n" : "") + line;
    consoleEl.scrollTop = consoleEl.scrollHeight;
  }

  function setProgress(pct, stage) {
    const p = Math.max(0, Math.min(100, Number(pct) || 0));
    barFill.style.width = `${p}%`;
    pctLabel.textContent = `${Math.round(p)}%`;
    if (stage) stageLabel.textContent = stage;
  }

  function formatVram(snap) {
    if (!snap) return "VRAM: —";
    return `VRAM ${snap.used_gb.toFixed(1)} / ${snap.total_gb.toFixed(1)} GB · alloc ${snap.allocated_gb.toFixed(1)}G`;
  }

  function updateVramDisplay(snap) {
    vramEl.textContent = formatVram(snap);
    if (snap && snap.used_gb / snap.total_gb > 0.92) {
      vramEl.classList.add("warn");
    } else {
      vramEl.classList.remove("warn");
    }
  }

  function applyPreset(name) {
    const p = PRESETS[name] || PRESETS.balanced;
    applyingPreset = true;
    lowVramEl.checked = !!p.low_vram;
    resolutionEl.value = String(p.resolution);
    stepsEl.value = String(p.steps);
    maxTokensEl.value = String(p.max_tokens);
    textureSizeEl.value = String(p.texture_size);
    decimationEl.value = String(p.decimation);
    applyingPreset = false;
  }

  function isImageFile(file) {
    if (!file) return false;
    if (file.type && file.type.startsWith("image/")) return true;
    return IMAGE_EXT.test(file.name || "");
  }

  function setFile(file) {
    if (!isImageFile(file)) {
      log(`Ignored non-image file: ${file?.name || "(unknown)"}`);
      return;
    }
    selectedFile = file;
    if (previewUrl) URL.revokeObjectURL(previewUrl);
    previewUrl = URL.createObjectURL(file);
    preview.src = previewUrl;
    preview.classList.remove("hidden");
    dropHint.classList.add("hidden");
    drop.classList.add("has-file");
    generateBtn.disabled = running;
    log(`Loaded: ${file.name} (${Math.round(file.size / 1024)} KB)`);
  }

  function openPicker() {
    fileInput.value = "";
    fileInput.click();
  }

  drop.addEventListener("click", (e) => {
    if (e.target === fileInput) return;
    e.preventDefault();
    openPicker();
  });

  drop.addEventListener("keydown", (e) => {
    if (e.key === "Enter" || e.key === " ") {
      e.preventDefault();
      openPicker();
    }
  });

  fileInput.addEventListener("change", () => {
    const file = fileInput.files && fileInput.files[0];
    if (file) setFile(file);
  });

  function onDrag(e) {
    e.preventDefault();
    e.stopPropagation();
  }

  ["dragenter", "dragover"].forEach((ev) => {
    drop.addEventListener(ev, (e) => {
      onDrag(e);
      drop.classList.add("dragover");
    });
  });

  drop.addEventListener("dragleave", (e) => {
    onDrag(e);
    if (!drop.contains(e.relatedTarget)) drop.classList.remove("dragover");
  });

  drop.addEventListener("drop", (e) => {
    onDrag(e);
    drop.classList.remove("dragover");
    const file = e.dataTransfer && e.dataTransfer.files && e.dataTransfer.files[0];
    if (file) setFile(file);
  });

  window.addEventListener("dragover", (e) => e.preventDefault());
  window.addEventListener("drop", (e) => e.preventDefault());

  presetEl.addEventListener("change", () => {
    applyPreset(presetEl.value);
    log(`Preset: ${presetEl.value}`);
  });

  lowVramEl.addEventListener("change", () => {
    if (applyingPreset) return;
    if (lowVramEl.checked && resolutionEl.value === "1536") {
      resolutionEl.value = "1024";
      log("Low VRAM on → snapped resolution to 1024 (use Max preset for 1536).");
    }
  });

  async function refreshHealth() {
    try {
      const res = await fetch("/api/health");
      const data = await res.json();
      const gpu = data.gpu
        ? `${data.gpu.name} · ${data.gpu.vram_gb} GB`
        : "No CUDA GPU detected";
      healthEl.textContent = `${gpu} · ATTN=${data.attn_backend || "?"}`;
      if (data.vram) updateVramDisplay(data.vram);
      if (data.default_preset && PRESETS[data.default_preset]) {
        presetEl.value = data.default_preset;
        applyPreset(data.default_preset);
      }
    } catch {
      healthEl.textContent = "Server health check failed";
    }
  }

  async function pollVram() {
    try {
      const res = await fetch("/api/vram");
      if (!res.ok) return;
      updateVramDisplay(await res.json());
    } catch {
      /* ignore */
    }
  }

  function startVramPolling() {
    stopVramPolling();
    vramTimer = setInterval(pollVram, 2000);
  }

  function stopVramPolling() {
    if (vramTimer) {
      clearInterval(vramTimer);
      vramTimer = null;
    }
  }

  async function startGenerate() {
    if (!selectedFile || running) return;
    running = true;
    generateBtn.disabled = true;
    consoleEl.textContent = "";
    downloadLink.classList.add("hidden");
    viewer.removeAttribute("src");
    setProgress(0, "queued");
    log("Uploading image…");
    startVramPolling();

    const fd = new FormData();
    fd.append("image", selectedFile, selectedFile.name);
    fd.append("seed", document.getElementById("seed").value || "42");
    fd.append("preset", presetEl.value || "balanced");
    fd.append("low_vram", lowVramEl.checked ? "true" : "false");
    fd.append("resolution", resolutionEl.value || "1024");
    fd.append("steps", stepsEl.value || "0");
    fd.append("max_tokens", maxTokensEl.value || "0");
    fd.append("texture_size", textureSizeEl.value || "0");
    fd.append("decimation", decimationEl.value || "0");
    fd.append("fov", document.getElementById("fov").value || "-1");

    try {
      const res = await fetch("/api/generate", { method: "POST", body: fd });
      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        throw new Error(err.detail || `HTTP ${res.status}`);
      }
      const body = await res.json();
      log(`Job ${body.job_id.slice(0, 8)}… started`);
      if (body.settings) {
        log(
          `Using: low_vram=${body.settings.low_vram} res=${body.settings.resolution} ` +
            `steps=${body.settings.steps} tokens=${body.settings.max_tokens} ` +
            `tex=${body.settings.texture_size} decim=${body.settings.decimation}`
        );
      }
      await watchJob(body.job_id);
    } catch (e) {
      log(`ERROR: ${e.message || e}`);
      setProgress(0, "error");
    } finally {
      running = false;
      generateBtn.disabled = !selectedFile;
      stopVramPolling();
      pollVram();
    }
  }

  function watchJob(jobId) {
    return new Promise((resolve) => {
      const es = new EventSource(`/api/jobs/${jobId}/events`);
      es.onmessage = (ev) => {
        let data;
        try {
          data = JSON.parse(ev.data);
        } catch {
          return;
        }
        if (data.logs && data.logs.length) data.logs.forEach((line) => log(line));
        if (data.vram) updateVramDisplay(data.vram);
        setProgress(data.progress, data.stage);
        if (data.status === "done") {
          if (data.glb_url) {
            viewer.src = data.glb_url;
            downloadLink.href = data.glb_url;
            downloadLink.classList.remove("hidden");
            log("Preview ready.");
          }
          es.close();
          resolve();
        } else if (data.status === "error") {
          log(data.error ? `Failed: ${data.error}` : "Failed.");
          es.close();
          resolve();
        }
      };
      es.onerror = () => {
        log("Event stream disconnected; polling status…");
        es.close();
        pollJob(jobId).then(resolve);
      };
    });
  }

  async function pollJob(jobId) {
    for (;;) {
      const res = await fetch(`/api/jobs/${jobId}`);
      if (!res.ok) break;
      const data = await res.json();
      setProgress(data.progress, data.stage);
      if (data.status === "done") {
        if (data.glb_url) {
          viewer.src = data.glb_url;
          downloadLink.href = data.glb_url;
          downloadLink.classList.remove("hidden");
        }
        return;
      }
      if (data.status === "error") {
        log(data.error || "Failed.");
        return;
      }
      await new Promise((r) => setTimeout(r, 1000));
    }
  }

  generateBtn.addEventListener("click", startGenerate);
  applyPreset("balanced");
  refreshHealth();
  setInterval(pollVram, 5000);
})();
