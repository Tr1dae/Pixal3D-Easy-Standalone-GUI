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
  const wipBadge = document.getElementById("wipBadge");
  const shadingBtn = document.getElementById("shadingBtn");
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
  let lastPreviewSrc = null;
  let vramTimer = null;
  let applyingPreset = false;

  const PREVIEW_REVEAL_MS = 5000;
  const PREVIEW_REVEAL_JITTER = 0.22; // fraction of Y-range mixed into up-order
  // Dwell covers the full per-item reveal, then a short hold before the next WIP
  const PREVIEW_DWELL_MS = PREVIEW_REVEAL_MS + 800;
  let previewQueue = [];
  let seenPreviewKeys = new Set();
  let previewShowing = false;
  let previewDwellTimer = null;
  let pendingFinal = null;
  let revealLoadHandler = null;
  let revealRaf = null;
  let shadingMode = "pbr"; // pbr | flat | wire
  const SHADING_CYCLE = ["pbr", "flat", "wire"];
  const SHADING_LABEL = { pbr: "PBR", flat: "FLAT", wire: "WIRE" };
  let savedEnvImage = viewer.getAttribute("environment-image");
  let savedShadow = viewer.getAttribute("shadow-intensity");
  let savedExposure = viewer.getAttribute("exposure");

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

  function setWipBadge(label, isFinal) {
    if (!wipBadge) return;
    if (!label) {
      wipBadge.classList.add("hidden");
      wipBadge.classList.remove("final");
      wipBadge.textContent = "";
      return;
    }
    wipBadge.textContent = isFinal ? `Final` : `WIP: ${label}`;
    wipBadge.classList.toggle("final", !!isFinal);
    wipBadge.classList.remove("hidden");
  }

  function getModelScene(mv) {
    for (const sym of Object.getOwnPropertySymbols(mv)) {
      const val = mv[sym];
      if (val && typeof val === "object" && val.isScene) return val;
    }
    return null;
  }

  function queueViewerRender() {
    const scene = getModelScene(viewer);
    if (scene && typeof scene.queueRender === "function") scene.queueRender();
  }

  function hash01(n) {
    const x = Math.sin(n * 127.1 + 311.7) * 43758.5453;
    return x - Math.floor(x);
  }

  function stopItemReveal() {
    if (revealRaf) {
      cancelAnimationFrame(revealRaf);
      revealRaf = null;
    }
    if (revealLoadHandler) {
      viewer.removeEventListener("load", revealLoadHandler);
      revealLoadHandler = null;
    }
  }

  function eachMeshMaterial(root, fn) {
    if (!root) return;
    root.traverse((obj) => {
      if (!obj.isMesh || !obj.material) return;
      const mats = Array.isArray(obj.material) ? obj.material : [obj.material];
      for (const m of mats) fn(obj, m);
    });
  }

  function inferVertsPerBox(vertCount, triCount) {
    if (vertCount >= 8 && Math.abs(triCount / 12 - vertCount / 8) < 0.51) return 8;
    if (vertCount >= 24 && Math.abs(triCount / 12 - vertCount / 24) < 0.51) return 24;
    if (vertCount >= 36 && Math.abs(triCount / 12 - vertCount / 36) < 0.51) return 36;
    return 0;
  }

  function buildRevealOrders(geometry) {
    const pos = geometry.getAttribute("position");
    if (!pos) return null;
    const n = pos.count;
    const orders = new Float32Array(n);
    const index = geometry.getIndex();
    const triCount = index ? index.count / 3 : n / 3;
    const vpb = inferVertsPerBox(n, triCount);

    let minY = Infinity;
    let maxY = -Infinity;
    for (let i = 0; i < n; i++) {
      const y = pos.getY(i);
      if (y < minY) minY = y;
      if (y > maxY) maxY = y;
    }
    const ySpan = Math.max(maxY - minY, 1e-6);
    const jitter = ySpan * PREVIEW_REVEAL_JITTER;

    let minO = Infinity;
    let maxO = -Infinity;

    if (vpb > 0) {
      const boxCount = Math.round(n / vpb);
      for (let b = 0; b < boxCount; b++) {
        let ySum = 0;
        const base = b * vpb;
        const count = Math.min(vpb, n - base);
        for (let k = 0; k < count; k++) ySum += pos.getY(base + k);
        const y = ySum / Math.max(count, 1);
        const order = y + (hash01(b + 17) - 0.5) * jitter;
        for (let k = 0; k < count; k++) orders[base + k] = order;
        if (order < minO) minO = order;
        if (order > maxO) maxO = order;
      }
    } else {
      for (let i = 0; i < n; i++) {
        const order = pos.getY(i) + (hash01(i + 17) - 0.5) * jitter;
        orders[i] = order;
        if (order < minO) minO = order;
        if (order > maxO) maxO = order;
      }
    }

    const Attr = pos.constructor;
    geometry.setAttribute("aReveal", new Attr(orders, 1));
    return { min: minO, max: maxO };
  }

  function patchMaterialReveal(material) {
    if (material.userData._revealPatched) return;
    material.userData._revealPatched = true;
    material.userData.revealUniform = { value: -1e9 };
    const prevCompile = material.onBeforeCompile;
    const prevKey = material.customProgramCacheKey
      ? material.customProgramCacheKey.bind(material)
      : null;
    material.customProgramCacheKey = () =>
      `wip-reveal|${prevKey ? prevKey() : ""}`;
    material.onBeforeCompile = (shader, ...rest) => {
      if (typeof prevCompile === "function") prevCompile(shader, ...rest);
      shader.uniforms.uRevealThresh = material.userData.revealUniform;
      shader.vertexShader = shader.vertexShader
        .replace(
          "#include <common>",
          `#include <common>
attribute float aReveal;
varying float vRevealOrder;
uniform float uRevealThresh;`
        )
        .replace(
          "#include <begin_vertex>",
          `#include <begin_vertex>
vRevealOrder = aReveal;
float revealVis = step(aReveal, uRevealThresh);
transformed *= revealVis;`
        );
      shader.fragmentShader = shader.fragmentShader
        .replace(
          "#include <common>",
          `#include <common>
varying float vRevealOrder;
uniform float uRevealThresh;`
        )
        .replace(
          "#include <clipping_planes_fragment>",
          `#include <clipping_planes_fragment>
if (vRevealOrder > uRevealThresh) discard;`
        );
      material.userData.shader = shader;
    };
    material.needsUpdate = true;
  }

  function startItemReveal() {
    stopItemReveal();
    revealLoadHandler = () => {
      revealLoadHandler = null;
      // Wait a frame so model-viewer finishes building the Three scene
      requestAnimationFrame(() => {
        const scene = getModelScene(viewer);
        if (!scene) return;

        let minO = Infinity;
        let maxO = -Infinity;
        const mats = [];

        eachMeshMaterial(scene, (mesh, material) => {
          if (!mesh.geometry) return;
          const range = buildRevealOrders(mesh.geometry);
          if (!range) return;
          if (range.min < minO) minO = range.min;
          if (range.max > maxO) maxO = range.max;
          patchMaterialReveal(material);
          mats.push(material);
        });

        if (!mats.length || !Number.isFinite(minO) || !Number.isFinite(maxO)) return;

        const t0 = performance.now();
        const start = minO - (maxO - minO) * 0.02;
        const end = maxO + (maxO - minO) * 0.02;

        const tick = (now) => {
          const u = Math.min(1, (now - t0) / PREVIEW_REVEAL_MS);
          // ease-out so early pops feel denser near the base
          const e = 1 - (1 - u) * (1 - u);
          const thresh = start + (end - start) * e;
          for (const m of mats) {
            if (m.userData.revealUniform) m.userData.revealUniform.value = thresh;
          }
          queueViewerRender();
          if (u < 1) {
            revealRaf = requestAnimationFrame(tick);
          } else {
            revealRaf = null;
          }
        };
        // Start fully hidden
        for (const m of mats) {
          if (m.userData.revealUniform) m.userData.revealUniform.value = start;
        }
        queueViewerRender();
        revealRaf = requestAnimationFrame(tick);
        applyShadingMode(false);
      });
    };
    viewer.addEventListener("load", revealLoadHandler, { once: true });
  }

  function backupMaterial(m) {
    if (m.userData._shadingBackup) return;
    m.userData._shadingBackup = {
      wireframe: !!m.wireframe,
      flatShading: !!m.flatShading,
      metalness: m.metalness,
      roughness: m.roughness,
      envMapIntensity: m.envMapIntensity,
    };
  }

  function applyShadingMode(requestRender = true) {
    if (shadingBtn) {
      shadingBtn.textContent = SHADING_LABEL[shadingMode] || "PBR";
      shadingBtn.dataset.mode = shadingMode;
    }

    if (shadingMode === "pbr") {
      if (savedEnvImage != null) viewer.setAttribute("environment-image", savedEnvImage);
      if (savedShadow != null) viewer.setAttribute("shadow-intensity", savedShadow);
      if (savedExposure != null) viewer.setAttribute("exposure", savedExposure);
    } else {
      viewer.removeAttribute("environment-image");
      viewer.setAttribute("shadow-intensity", "0");
      viewer.setAttribute("exposure", shadingMode === "flat" ? "0.85" : "1");
    }

    const scene = getModelScene(viewer);
    eachMeshMaterial(scene, (_mesh, m) => {
      backupMaterial(m);
      const b = m.userData._shadingBackup;
      if (shadingMode === "wire") {
        m.wireframe = true;
        m.flatShading = true;
        if ("metalness" in m) m.metalness = 0;
        if ("roughness" in m) m.roughness = 1;
        if ("envMapIntensity" in m) m.envMapIntensity = 0;
      } else if (shadingMode === "flat") {
        m.wireframe = false;
        m.flatShading = true;
        if ("metalness" in m) m.metalness = 0;
        if ("roughness" in m) m.roughness = 1;
        if ("envMapIntensity" in m) m.envMapIntensity = 0;
      } else {
        m.wireframe = b.wireframe;
        m.flatShading = b.flatShading;
        if ("metalness" in m && b.metalness != null) m.metalness = b.metalness;
        if ("roughness" in m && b.roughness != null) m.roughness = b.roughness;
        if ("envMapIntensity" in m && b.envMapIntensity != null) {
          m.envMapIntensity = b.envMapIntensity;
        }
      }
      m.needsUpdate = true;
    });
    if (requestRender) queueViewerRender();
  }

  if (shadingBtn) {
    shadingBtn.addEventListener("click", () => {
      const i = SHADING_CYCLE.indexOf(shadingMode);
      shadingMode = SHADING_CYCLE[(i + 1) % SHADING_CYCLE.length];
      applyShadingMode(true);
    });
  }

  function applyViewerSrc(url, label, { final = false } = {}) {
    if (!url) return;
    if (url === lastPreviewSrc && !final) return;
    lastPreviewSrc = url;
    stopItemReveal();
    if (!final) startItemReveal();
    viewer.src = url;
    setWipBadge(label || (final ? "Final" : "preview"), final);
    // Re-apply shading after final loads too
    if (final) {
      const onFinalLoad = () => applyShadingMode(true);
      viewer.addEventListener("load", onFinalLoad, { once: true });
    }
  }

  function resetPreviewQueue() {
    if (previewDwellTimer) {
      clearTimeout(previewDwellTimer);
      previewDwellTimer = null;
    }
    previewQueue = [];
    seenPreviewKeys = new Set();
    previewShowing = false;
    pendingFinal = null;
    lastPreviewSrc = null;
    stopItemReveal();
  }

  function previewKey(item) {
    if (item.seq != null) return `seq:${item.seq}`;
    return `url:${item.url}`;
  }

  function enqueuePreviewHistory(history) {
    if (!history || !history.length) return;
    for (const item of history) {
      if (!item || !item.url) continue;
      const key = previewKey(item);
      if (seenPreviewKeys.has(key)) continue;
      seenPreviewKeys.add(key);
      previewQueue.push({
        url: item.url,
        label: item.label || "preview",
        final: false,
      });
    }
    pumpPreviewQueue();
  }

  function showFinalNow(url, label) {
    if (previewDwellTimer) {
      clearTimeout(previewDwellTimer);
      previewDwellTimer = null;
    }
    previewQueue = [];
    previewShowing = false;
    pendingFinal = null;
    applyViewerSrc(url, label || "Final", { final: true });
  }

  function pumpPreviewQueue() {
    if (previewShowing) return;
    if (pendingFinal) {
      const fin = pendingFinal;
      pendingFinal = null;
      showFinalNow(fin.url, fin.label);
      return;
    }
    if (!previewQueue.length) return;
    const next = previewQueue.shift();
    previewShowing = true;
    applyViewerSrc(next.url, next.label, { final: !!next.final });
    previewDwellTimer = setTimeout(() => {
      previewDwellTimer = null;
      previewShowing = false;
      pumpPreviewQueue();
    }, PREVIEW_DWELL_MS);
  }

  function requestFinalPreview(url, label) {
    if (!url) return;
    // Preempt: drop remaining WIP dwell and show final immediately
    showFinalNow(url, label || "Final");
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
    resetPreviewQueue();
    setWipBadge(null);
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
        if (data.preview_history && data.preview_history.length) {
          enqueuePreviewHistory(data.preview_history);
        } else if (data.preview_url && data.status !== "done") {
          enqueuePreviewHistory([
            { url: data.preview_url, label: data.preview_label, seq: data.preview_url },
          ]);
        }
        if (data.status === "done") {
          if (data.glb_url) {
            requestFinalPreview(data.glb_url, "Final");
            downloadLink.href = data.glb_url;
            downloadLink.classList.remove("hidden");
            log("Final GLB ready.");
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
      if (data.preview_history && data.preview_history.length) {
        enqueuePreviewHistory(data.preview_history);
      } else if (data.preview_url && data.status !== "done") {
        enqueuePreviewHistory([
          { url: data.preview_url, label: data.preview_label, seq: data.preview_url },
        ]);
      }
      if (data.status === "done") {
        if (data.glb_url) {
          requestFinalPreview(data.glb_url, "Final");
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
  setInterval(pollVram, 10000);
})();
