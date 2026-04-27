import * as THREE from './vendor/three/build/three.module.js';
import { OrbitControls } from './vendor/three/examples/jsm/controls/OrbitControls.js';
import { EffectComposer } from './vendor/three/examples/jsm/postprocessing/EffectComposer.js';
import { RenderPass } from './vendor/three/examples/jsm/postprocessing/RenderPass.js';
import { UnrealBloomPass } from './vendor/three/examples/jsm/postprocessing/UnrealBloomPass.js';

const COLORS = {
  existing: 0x1f77b4,
  newNode: 0x2ca02c,
  removed: 0xd62728,
  output: 0xf1c40f
};

const MODEL_STYLE = {
  Static: {
    emissive: 0.06,
    lineOpacity: 0.2,
    xSpacing: 5.4,
    yzSpacing: 0.7,
    exposure: 1.05,
    spin: 0.0006,
    bloomStrength: 0.55,
    bloomRadius: 0.25,
    bloomThreshold: 0.3
  },
  SANN: {
    emissive: 0.18,
    lineOpacity: 0.3,
    xSpacing: 6.0,
    yzSpacing: 0.75,
    exposure: 1.12,
    spin: 0.001,
    bloomStrength: 0.95,
    bloomRadius: 0.35,
    bloomThreshold: 0.26
  },
  'CA-SANN': {
    emissive: 0.14,
    lineOpacity: 0.24,
    xSpacing: 6.4,
    yzSpacing: 0.78,
    exposure: 1.08,
    spin: 0.0013,
    bloomStrength: 0.22,
    bloomRadius: 0.12,
    bloomThreshold: 0.6
  }
};

const UI = {
  modelSelect: document.getElementById('modelSelect'),
  epochSlider: document.getElementById('epochSlider'),
  epochLabel: document.getElementById('epochLabel'),
  playPause: document.getElementById('playPause'),
  canvasContainer: document.getElementById('canvasContainer'),
  flash: document.getElementById('flash'),
  hint: document.getElementById('hint'),
  sModel: document.getElementById('sModel'),
  sEpoch: document.getElementById('sEpoch'),
  sAcc: document.getElementById('sAcc'),
  sSize: document.getElementById('sSize'),
  sEff: document.getElementById('sEff'),
  sGrowth: document.getElementById('sGrowth'),
  sDecision: document.getElementById('sDecision'),
  metaDataset: document.getElementById('metaDataset'),
  metaSeed: document.getElementById('metaSeed'),
  metaEpochs: document.getElementById('metaEpochs'),
  metaSource: document.getElementById('metaSource'),
  metaNotes: document.getElementById('metaNotes')
};

function clamp(v, a, b) {
  return Math.max(a, Math.min(b, v));
}

function deltaFromCumulative(arr) {
  const out = [];
  let prev = 0;
  for (let i = 0; i < arr.length; i++) {
    const cur = Number(arr[i] ?? 0);
    out.push(cur - prev);
    prev = cur;
  }
  return out;
}

function seededRandom(seed) {
  let t = seed >>> 0;
  return () => {
    t += 0x6d2b79f5;
    let r = Math.imul(t ^ (t >>> 15), 1 | t);
    r ^= r + Math.imul(r ^ (r >>> 7), 61 | r);
    return ((r ^ (r >>> 14)) >>> 0) / 4294967296;
  };
}

function gridYZ(index, nCols, spacing) {
  const r = Math.floor(index / nCols);
  const c = index % nCols;
  const y = (c - (nCols - 1) / 2) * spacing;
  const z = -r * spacing;
  return { y, z };
}

function isWebGLAvailable() {
  try {
    const canvas = document.createElement('canvas');
    return !!(
      window.WebGLRenderingContext &&
      (canvas.getContext('webgl') || canvas.getContext('experimental-webgl'))
    );
  } catch {
    return false;
  }
}

function computeLayerLayout(maxWidth, spacing) {
  const nCols = Math.max(1, Math.ceil(Math.sqrt(maxWidth)));
  return { nCols, spacing };
}

class NetworkRenderer {
  constructor(container) {
    this.container = container;
    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(0x05070d);
    this.scene.fog = new THREE.Fog(0x090d12, 35, 130);

    const w = container.clientWidth;
    const h = container.clientHeight;

    this.camera = new THREE.PerspectiveCamera(55, w / h, 0.1, 500);
    this.camera.position.set(26, 14, 22);

    this.renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    this.renderer.setSize(w, h);
    this.renderer.setPixelRatio(Math.min(2, window.devicePixelRatio || 1));
    this.renderer.outputColorSpace = THREE.SRGBColorSpace;
    this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
    this.renderer.toneMappingExposure = 1.1;
    this.renderer.setClearColor(0x05070d, 1);
    container.appendChild(this.renderer.domElement);

    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.08;
    this.controls.target.set(10, 0, -4);

    this.scene.add(new THREE.AmbientLight(0x9fb8d9, 0.38));

    const key = new THREE.DirectionalLight(0xffffff, 0.75);
    key.position.set(20, 25, 18);
    this.scene.add(key);

    const rim = new THREE.DirectionalLight(0x7aa7ff, 0.3);
    rim.position.set(-25, 10, -25);
    this.scene.add(rim);

    this.composer = new EffectComposer(this.renderer);
    this.renderPass = new RenderPass(this.scene, this.camera);
    this.composer.addPass(this.renderPass);
    this.bloomPass = new UnrealBloomPass(new THREE.Vector2(w, h), 1.0, 0.35, 0.25);
    this.composer.addPass(this.bloomPass);

    const starCount = 900;
    const starPositions = new Float32Array(starCount * 3);
    const rng = seededRandom(424242);

    for (let i = 0; i < starCount; i++) {
      const r = 85 * Math.pow(rng(), 0.55);
      const theta = rng() * Math.PI * 2;
      const phi = Math.acos(2 * rng() - 1);
      const x = r * Math.sin(phi) * Math.cos(theta);
      const y = r * Math.sin(phi) * Math.sin(theta);
      const z = r * Math.cos(phi);
      starPositions[i * 3] = x;
      starPositions[i * 3 + 1] = y;
      starPositions[i * 3 + 2] = z;
    }

    const starsGeo = new THREE.BufferGeometry();
    starsGeo.setAttribute('position', new THREE.BufferAttribute(starPositions, 3));
    const starsMat = new THREE.PointsMaterial({
      size: 0.06,
      color: 0xa9c4ff,
      transparent: true,
      opacity: 0.55,
      blending: THREE.AdditiveBlending,
      depthWrite: false
    });
    this.stars = new THREE.Points(starsGeo, starsMat);
    this.scene.add(this.stars);

    const grid = new THREE.GridHelper(120, 60, 0x1f2a3a, 0x131b27);
    grid.position.set(0, -10, 0);
    grid.material.opacity = 0.35;
    grid.material.transparent = true;
    this.scene.add(grid);

    this.maxSpheresPerLayer = 160;
    this.lineBudget = 1800;
    this.group = new THREE.Group();
    this.scene.add(this.group);

    this.sphereGeo = new THREE.SphereGeometry(0.26, 12, 12);
    this.materialsBase = {
      existing: new THREE.MeshStandardMaterial({
        color: COLORS.existing,
        roughness: 0.16,
        metalness: 0.45,
        emissive: COLORS.existing,
        emissiveIntensity: 0.16
      }),
      newNode: new THREE.MeshStandardMaterial({
        color: COLORS.newNode,
        roughness: 0.1,
        metalness: 0.5,
        emissive: COLORS.newNode,
        emissiveIntensity: 0.28
      }),
      removed: new THREE.MeshStandardMaterial({
        color: COLORS.removed,
        roughness: 0.22,
        metalness: 0.35,
        emissive: COLORS.removed,
        emissiveIntensity: 0.18,
        transparent: true,
        opacity: 0.78
      }),
      output: new THREE.MeshStandardMaterial({
        color: COLORS.output,
        roughness: 0.1,
        metalness: 0.55,
        emissive: COLORS.output,
        emissiveIntensity: 0.22
      })
    };

    this.materialsByModel = new Map();
    this.lines = null;
    this.lineMaterial = new THREE.LineBasicMaterial({
      color: 0x9fd2c2,
      transparent: true,
      opacity: 0.3,
      blending: THREE.AdditiveBlending,
      depthWrite: false
    });

    this.activeMeshes = [];
    this.spawnStartMs = 0;
    this.style = MODEL_STYLE['CA-SANN'];

    window.addEventListener('resize', () => this.resize());
  }

  resize() {
    const w = this.container.clientWidth;
    const h = this.container.clientHeight;
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
    this.renderer.setSize(w, h);
    this.composer.setSize(w, h);
  }

  clear() {
    for (const m of this.activeMeshes) {
      this.group.remove(m);
    }
    this.activeMeshes = [];

    if (this.lines) {
      this.group.remove(this.lines);
      this.lines.geometry.dispose();
      this.lines = null;
    }
  }

  _materialsForModel(modelName) {
    if (this.materialsByModel.has(modelName)) {
      return this.materialsByModel.get(modelName);
    }

    const style = MODEL_STYLE[modelName] ?? MODEL_STYLE['CA-SANN'];
    const make = (base) => {
      const m = base.clone();
      m.emissiveIntensity = (base.emissiveIntensity ?? 0.12) + style.emissive;
      return m;
    };

    const mats = {
      existing: make(this.materialsBase.existing),
      newNode: make(this.materialsBase.newNode),
      removed: make(this.materialsBase.removed),
      output: make(this.materialsBase.output)
    };

    this.materialsByModel.set(modelName, mats);
    return mats;
  }

  buildEpoch({ modelName, hiddenWidths, outputWidth, prevHiddenWidths, growthDecision, epochKey }) {
    this.clear();

    this.style = MODEL_STYLE[modelName] ?? MODEL_STYLE['CA-SANN'];
    this.renderer.toneMappingExposure = this.style.exposure;
    this.bloomPass.strength = this.style.bloomStrength;
    this.bloomPass.radius = this.style.bloomRadius;
    this.bloomPass.threshold = this.style.bloomThreshold;

    if (modelName === 'Static') {
      this.controls.target.set(9.2, 0.0, -3.6);
    } else if (modelName === 'SANN') {
      this.controls.target.set(10.0, 0.0, -4.0);
    } else {
      this.controls.target.set(10.8, 0.0, -4.2);
    }

    const xSpacing = this.style.xSpacing;
    const yzSpacing = this.style.yzSpacing;
    const layerCount = hiddenWidths.length + 1;
    const materials = this._materialsForModel(modelName);
    const maxPerLayer = [];

    for (let i = 0; i < hiddenWidths.length; i++) {
      maxPerLayer.push(Math.max(hiddenWidths[i], prevHiddenWidths?.[i] ?? hiddenWidths[i]));
    }
    maxPerLayer.push(outputWidth);

    const layouts = maxPerLayer.map((w) => computeLayerLayout(w, yzSpacing));
    const nodePositionsByLayer = [];
    this.spawnStartMs = performance.now();

    function sampleIndices(width, limit) {
      if (width <= limit) {
        return Array.from({ length: width }, (_, i) => i);
      }
      const out = new Set();
      for (let i = 0; i < limit; i++) {
        out.add(Math.round((i * (width - 1)) / (limit - 1)));
      }
      return Array.from(out).sort((a, b) => a - b);
    }

    for (let layer = 0; layer < layerCount; layer++) {
      const isOutput = layer === layerCount - 1;
      const width = isOutput ? outputWidth : hiddenWidths[layer];
      const prevWidth = isOutput ? outputWidth : (prevHiddenWidths?.[layer] ?? width);
      const { nCols, spacing } = layouts[layer];
      const limit = isOutput ? Math.min(60, this.maxSpheresPerLayer) : this.maxSpheresPerLayer;
      const indices = sampleIndices(width, limit);
      const layerNodes = [];

      for (const idx of indices) {
        const { y, z } = gridYZ(idx, nCols, spacing);
        const x = layer * xSpacing;

        let mat = materials.existing;
        let isNew = false;

        if (isOutput) {
          mat = materials.output;
        } else {
          const grewBy = Math.max(0, width - prevWidth);
          if (grewBy > 0 && idx >= width - grewBy) {
            mat = materials.newNode;
            isNew = true;
          }
        }

        const mesh = new THREE.Mesh(this.sphereGeo, mat);
        mesh.position.set(x, y, z);
        mesh.scale.setScalar(isNew ? 0 : 1);
        mesh.userData = { layer, idx, isNew, isOutput };
        this.group.add(mesh);
        this.activeMeshes.push(mesh);
        layerNodes.push(mesh.position.clone());
      }

      if (!isOutput) {
        const shrankBy = Math.max(0, prevWidth - width);
        if (shrankBy > 0) {
          const removedCount = Math.min(18, shrankBy);
          const removedIndices = sampleIndices(shrankBy, Math.max(2, removedCount));

          for (const removedIdx of removedIndices) {
            const ghostIdx = width + removedIdx;
            const { y, z } = gridYZ(ghostIdx, nCols, spacing);
            const mesh = new THREE.Mesh(this.sphereGeo, materials.removed);
            mesh.position.set(layer * xSpacing, y, z - 0.6);
            mesh.scale.setScalar(0.95);
            mesh.userData = { layer, idx: ghostIdx, isRemoved: true };
            this.group.add(mesh);
            this.activeMeshes.push(mesh);
          }
        }
      }

      nodePositionsByLayer.push(layerNodes);
    }

    const rand = seededRandom(1337 + epochKey * 17);
    const segments = [];
    let budget = this.lineBudget;

    for (let layer = 0; layer < nodePositionsByLayer.length - 1; layer++) {
      const src = nodePositionsByLayer[layer];
      const dst = nodePositionsByLayer[layer + 1];
      if (!src.length || !dst.length) {
        continue;
      }

      const perSrc = Math.max(1, Math.min(5, Math.floor(budget / Math.max(1, src.length))));
      for (let i = 0; i < src.length; i++) {
        if (budget <= 0) {
          break;
        }
        for (let k = 0; k < perSrc; k++) {
          if (budget <= 0) {
            break;
          }
          const j = Math.floor(rand() * dst.length);
          segments.push(src[i].x, src[i].y, src[i].z);
          segments.push(dst[j].x, dst[j].y, dst[j].z);
          budget--;
        }
      }
    }

    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.Float32BufferAttribute(segments, 3));
    this.lineMaterial.opacity = this.style.lineOpacity;

    if (modelName === 'Static') {
      this.lineMaterial.color.setHex(0x7fd0ff);
    } else if (modelName === 'SANN') {
      this.lineMaterial.color.setHex(0xa6ffd8);
    } else {
      this.lineMaterial.color.setHex(0x9fd2ff);
    }

    this.lines = new THREE.LineSegments(geo, this.lineMaterial);
    this.group.add(this.lines);

    if (growthDecision === 'accepted') {
      flash('accepted');
    } else if (growthDecision === 'rejected') {
      flash('rejected');
    }
  }

  update() {
    const now = performance.now();
    const t = clamp((now - this.spawnStartMs) / 420, 0, 1);
    const ease = t * t * (3 - 2 * t);

    for (const mesh of this.activeMeshes) {
      if (mesh.userData?.isNew) {
        mesh.scale.setScalar(ease);
      }
    }

    this.stars.rotation.y += this.style.spin;
    this.stars.rotation.x += this.style.spin * 0.33;
    this.controls.update();
    this.composer.render();
  }
}

let DATA = null;
let currentModel = 'CA-SANN';
let epochIndex = 0;
let playing = false;
let timer = null;

if (!isWebGLAvailable()) {
  if (UI.hint) {
    UI.hint.textContent = 'WebGL is not available in this browser or environment.';
  }
  throw new Error('WebGL is not available.');
}

const renderer3d = new NetworkRenderer(UI.canvasContainer);

function setFlash(color) {
  UI.flash.style.background = color;
  UI.flash.classList.add('on');
  window.setTimeout(() => UI.flash.classList.remove('on'), 260);
}

function flash(kind) {
  if (kind === 'accepted') {
    setFlash('rgba(44,160,44,0.55)');
  }
  if (kind === 'rejected') {
    setFlash('rgba(214,39,40,0.55)');
  }
}

function fmtNum(x) {
  if (x === null || x === undefined || Number.isNaN(Number(x))) {
    return '-';
  }
  return Number(x).toLocaleString(undefined, { maximumFractionDigits: 6 });
}

function fmtSci(x) {
  if (x === null || x === undefined || Number.isNaN(Number(x))) {
    return '-';
  }
  const v = Number(x);
  if (v === 0) {
    return '0';
  }
  return v.toExponential(6);
}

function titleCase(value) {
  if (!value) {
    return '-';
  }
  const text = String(value).replace(/[_-]+/g, ' ');
  return text.charAt(0).toUpperCase() + text.slice(1);
}

function getModelData(modelName) {
  return DATA?.models?.[modelName] ?? null;
}

function getEpochKey(modelData, idx) {
  return Number(modelData.epochs[idx]);
}

function getHiddenWidths(modelData, epochKey) {
  const hw = modelData.structure.hidden_widths_by_epoch[String(epochKey)];
  return hw ? hw.map(Number) : [64, 64];
}

function decisionForEpoch(modelData, idx) {
  const growDelta = deltaFromCumulative(modelData.events.growth_events_cumulative);
  const rejectedDelta = deltaFromCumulative(modelData.events.rejected_growth_events_cumulative);
  const grew = (growDelta[idx] ?? 0) > 0;
  const rejected = (rejectedDelta[idx] ?? 0) > 0;

  if (grew && rejected) {
    return 'rejected';
  }
  if (grew) {
    return 'accepted';
  }
  return 'none';
}

function growthHappened(modelData, idx) {
  const growDelta = deltaFromCumulative(modelData.events.growth_events_cumulative);
  return (growDelta[idx] ?? 0) > 0;
}

function updateSidePanel(modelName, modelData, idx) {
  const e = modelData.epochs[idx];
  UI.sModel.textContent = modelName;
  UI.sEpoch.textContent = String(e);
  UI.sAcc.textContent = fmtNum(modelData.metrics.accuracy[idx]);
  UI.sSize.textContent = fmtNum(modelData.metrics.model_size[idx]);
  UI.sEff.textContent = fmtSci(modelData.metrics.efficiency[idx]);
  UI.sGrowth.textContent = growthHappened(modelData, idx) ? 'Yes' : 'No';

  const decision = decisionForEpoch(modelData, idx);
  UI.sDecision.textContent =
    decision === 'accepted' ? 'Accepted' :
    decision === 'rejected' ? 'Rejected' :
    '-';
}

function updateMetaPanel() {
  const model = getModelData(currentModel);
  const meta = DATA?.meta ?? {};
  UI.metaDataset.textContent = titleCase(meta.dataset);
  UI.metaSeed.textContent = fmtNum(meta.seed);
  UI.metaEpochs.textContent = model ? String(model.epochs.length) : '-';
  UI.metaSource.textContent = meta.source ?? '-';
  UI.metaNotes.textContent = meta.notes ?? '';
}

function renderEpoch() {
  const modelData = getModelData(currentModel);
  if (!modelData) {
    return;
  }

  epochIndex = clamp(epochIndex, 0, modelData.epochs.length - 1);
  const epochKey = getEpochKey(modelData, epochIndex);
  UI.epochSlider.value = String(epochIndex);
  UI.epochLabel.textContent = String(epochKey);

  const hiddenWidths = getHiddenWidths(modelData, epochKey);
  const prevEpochKey = getEpochKey(modelData, Math.max(0, epochIndex - 1));
  const prevHiddenWidths = epochIndex > 0 ? getHiddenWidths(modelData, prevEpochKey) : hiddenWidths;
  const decision = decisionForEpoch(modelData, epochIndex);

  renderer3d.buildEpoch({
    modelName: currentModel,
    hiddenWidths,
    outputWidth: Number(modelData.structure.output_width ?? 10),
    prevHiddenWidths,
    growthDecision: decision,
    epochKey
  });

  updateSidePanel(currentModel, modelData, epochIndex);
  updateMetaPanel();
}

function setModel(modelName) {
  currentModel = modelName;
  const modelData = getModelData(currentModel);
  if (!modelData) {
    return;
  }

  UI.epochSlider.min = '0';
  UI.epochSlider.max = String(modelData.epochs.length - 1);
  epochIndex = 0;
  renderEpoch();
}

function play() {
  if (playing) {
    return;
  }

  playing = true;
  UI.playPause.textContent = 'Pause';
  const modelData = getModelData(currentModel);
  const max = modelData.epochs.length - 1;

  timer = window.setInterval(() => {
    epochIndex = epochIndex >= max ? 0 : epochIndex + 1;
    renderEpoch();
  }, 900);
}

function pause() {
  playing = false;
  UI.playPause.textContent = 'Play';
  if (timer) {
    window.clearInterval(timer);
    timer = null;
  }
}

async function main() {
  let res;
  try {
    res = await fetch('./data.json');
  } catch (error) {
    if (window.location.protocol === 'file:') {
      throw new Error('Open this page through a local static server so data.json can be loaded.');
    }
    throw error;
  }

  if (!res.ok) {
    throw new Error(`Failed to load data.json: ${res.status}`);
  }

  DATA = await res.json();

  if (DATA.models['CA-SANN']) {
    UI.modelSelect.value = 'CA-SANN';
    currentModel = 'CA-SANN';
  } else {
    currentModel = Object.keys(DATA.models)[0];
    UI.modelSelect.value = currentModel;
  }

  UI.modelSelect.addEventListener('change', () => {
    pause();
    setModel(UI.modelSelect.value);
  });

  UI.epochSlider.addEventListener('input', () => {
    pause();
    epochIndex = Number(UI.epochSlider.value);
    renderEpoch();
  });

  UI.playPause.addEventListener('click', () => {
    if (playing) {
      pause();
    } else {
      play();
    }
  });

  updateMetaPanel();
  setModel(currentModel);

  function animate() {
    renderer3d.update();
    requestAnimationFrame(animate);
  }

  animate();
}

main().catch((err) => {
  console.error(err);
  if (UI.hint) {
    UI.hint.textContent = `Unable to load visualization data: ${err.message}`;
  }
});
