import * as THREE from 'three';
import { RoundedBoxGeometry } from 'three/addons/geometries/RoundedBoxGeometry.js';
import { RoomEnvironment } from 'three/addons/environments/RoomEnvironment.js';
import { resolveRobotAsset } from './robotAsset.js';

/**
 * Scene3D: real-time WebGL scene (three.js) with a robot that reacts to
 * the InteractionController snapshot and the primary hand's position.
 *
 * Inputs per frame (no gesture logic in here):
 *   snapshot  controller.snapshot(): state, action {name, progress}, hold {name, params},
 *             pose {yaw, x, z} and movement {state, speed, gaitPhase, rateDegS} (the
 *             controller owns position, yaw, speed and gait phase; the scene renders them)
 *   gaze      AttentionController output {mode, yaw, pitch}: the HEAD pivot (and the eyes)
 *             follow it; the locomotion root is never turned to "look"
 * Emotions (LAUGH / SAD actions) are drawn with the existing rig plus three small
 * emissive arcs on the visor (happy eyes, smile, frown): the procedural robot has no
 * mouth or eyelids, so these are added shapes, not facial animation of a model.
 *   hand      { palm: {x,y}, indexTip: {x,y} } in normalised *display* coords, or null
 *   map       cover mapping (display coords -> container px)
 *
 * Robot: procedural by default (documented in docs/ARCHITECTURE.md). If
 * config.scene.robotModelUrl points to a GLB/glTF file, it is loaded
 * asynchronously (the tracking pipeline is never blocked); on failure the
 * procedural robot stays and the reason is reported.
 */

const COLORS = {
  orange: new THREE.Color('#ff7a1a'),
  cyan: new THREE.Color('#38e1ff'),
  white: new THREE.Color('#d8f6ff'),
  amber: new THREE.Color('#ffb020'),
  red: new THREE.Color('#ff2a3a'),
};
const SAD_BLUE = new THREE.Color('#5d7dff');
const UP = new THREE.Vector3(0, 1, 0);
const DOWN = new THREE.Vector3(0, -1, 0);
const damp = (k, dt) => 1 - Math.exp(-k * dt);
const easeInOut = (t) => (t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2);

function glowTexture() {
  const c = document.createElement('canvas');
  c.width = c.height = 128;
  const g = c.getContext('2d');
  const grad = g.createRadialGradient(64, 64, 0, 64, 64, 64);
  grad.addColorStop(0, 'rgba(255,255,255,1)');
  grad.addColorStop(0.3, 'rgba(255,255,255,0.35)');
  grad.addColorStop(1, 'rgba(255,255,255,0)');
  g.fillStyle = grad;
  g.fillRect(0, 0, 128, 128);
  const tex = new THREE.CanvasTexture(c);
  tex.colorSpace = THREE.SRGBColorSpace;
  return tex;
}

/**
 * Yemen national flag as a real 3D surface on the torso: three horizontal strips, red over
 * white over black (official colours #CE1126 / #FFFFFF / #000000), 2:3 height:width, in a
 * thin light frame so the black stripe stays visible on the dark armour. Children of the
 * torso: they inherit every body transform (walk, spin, lift, emotions) and never float.
 * @returns {THREE.Group} named 'yemenFlag', front face at z = front + 0.008
 */
export const YEMEN_FLAG = Object.freeze({ colors: ['#ce1126', '#ffffff', '#000000'], height: 0.14, aspect: 1.5, border: 0.008 });
// y: the whole framed flag stays inside the torso's flat front face (y 0.15 .. 0.31, below the chest plate)
export function addYemenFlag(torso, { y = 0.228, front = 0.28 } = {}) {
  const h = YEMEN_FLAG.height;
  const w = h * YEMEN_FLAG.aspect;
  const flag = new THREE.Group();
  flag.name = 'yemenFlag';
  flag.position.set(0, y, front);
  const frame = new THREE.Mesh(new THREE.BoxGeometry(w + 2 * YEMEN_FLAG.border, h + 2 * YEMEN_FLAG.border, 0.004), new THREE.MeshStandardMaterial({ color: '#9aa3ad', metalness: 0.7, roughness: 0.35 }));
  frame.name = 'yemenFlagFrame';
  frame.position.z = 0.002;
  flag.add(frame);
  YEMEN_FLAG.colors.forEach((color, i) => {
    const c = new THREE.Color(color);
    const strip = new THREE.Mesh(
      new THREE.BoxGeometry(w, h / 3, 0.006),
      // a little emissive so the colours read under the scene's dark lighting; matte cloth-like
      new THREE.MeshStandardMaterial({ color: c, emissive: c.clone().multiplyScalar(0.18), roughness: 0.75, metalness: 0.0 }),
    );
    strip.name = `yemenFlag_${['red', 'white', 'black'][i]}`;
    strip.position.set(0, h / 3 - (i * h) / 3, 0.006); // top, middle, bottom
    flag.add(strip);
  });
  torso.add(flag);
  return flag;
}

/** Builds the procedural robot. Units: metres, feet at y=0, facing +z. */
export function buildProceduralRobot(glowTex) {
  const metal = new THREE.MeshStandardMaterial({ color: '#2b2f36', metalness: 0.85, roughness: 0.32 });
  const metalLight = new THREE.MeshStandardMaterial({ color: '#5b626c', metalness: 0.9, roughness: 0.25 });
  const joint = new THREE.MeshStandardMaterial({ color: '#16181c', metalness: 0.6, roughness: 0.5 });
  const visor = new THREE.MeshStandardMaterial({ color: '#030405', metalness: 1, roughness: 0.08 });
  const accent = new THREE.MeshStandardMaterial({ color: '#2a1405', emissive: COLORS.orange, emissiveIntensity: 1.4, metalness: 0.4, roughness: 0.4 });
  const eyeMat = new THREE.MeshStandardMaterial({ color: '#000000', emissive: COLORS.cyan.clone(), emissiveIntensity: 2.5 });
  const coreMat = new THREE.MeshStandardMaterial({ color: '#000000', emissive: COLORS.orange.clone(), emissiveIntensity: 2.0 });
  const ringMat = new THREE.MeshBasicMaterial({ color: COLORS.cyan.clone(), transparent: true, opacity: 0.8, blending: THREE.AdditiveBlending, depthWrite: false });

  const mesh = (geo, mat, pos = [0, 0, 0], parent = null) => {
    const m = new THREE.Mesh(geo, mat);
    m.position.set(...pos);
    parent?.add(m);
    return m;
  };
  const sprite = (color, size, pos, parent) => {
    const s = new THREE.Sprite(new THREE.SpriteMaterial({ map: glowTex, color, transparent: true, blending: THREE.AdditiveBlending, depthWrite: false }));
    s.scale.set(size, size, 1);
    s.position.set(...pos);
    parent.add(s);
    return s;
  };

  const root = new THREE.Group();
  root.name = 'robot';
  const body = new THREE.Group(); // bobbing body
  root.add(body);

  // hover base
  const ringA = mesh(new THREE.TorusGeometry(0.62, 0.012, 8, 96), ringMat, [0, 0.02, 0], root);
  ringA.rotation.x = Math.PI / 2;
  const ringBMat = ringMat.clone();
  ringBMat.color = COLORS.orange.clone();
  ringBMat.opacity = 0.45;
  const ringB = mesh(new THREE.TorusGeometry(0.9, 0.006, 8, 128, Math.PI * 1.4), ringBMat, [0, 0.02, 0], root);
  ringB.rotation.x = Math.PI / 2;
  const floorGlow = new THREE.Mesh(
    new THREE.PlaneGeometry(2.2, 2.2),
    new THREE.MeshBasicMaterial({ map: glowTex, color: COLORS.cyan.clone().multiplyScalar(0.35), transparent: true, blending: THREE.AdditiveBlending, depthWrite: false }),
  );
  floorGlow.rotation.x = -Math.PI / 2;
  floorGlow.position.y = 0.01;
  root.add(floorGlow);

  // thruster + pelvis
  mesh(new THREE.CylinderGeometry(0.2, 0.11, 0.36, 24), metal, [0, 0.5, 0], body);
  const thruster = mesh(new THREE.CylinderGeometry(0.1, 0.1, 0.02, 24), coreMat, [0, 0.31, 0], body);
  sprite(COLORS.orange, 0.6, [0, 0.26, 0], body);
  mesh(new RoundedBoxGeometry(0.62, 0.24, 0.42, 3, 0.08), joint, [0, 0.75, 0], body);

  // torso
  const torso = new THREE.Group();
  torso.position.set(0, 0.86, 0);
  body.add(torso);
  mesh(new RoundedBoxGeometry(0.98, 0.86, 0.56, 4, 0.13), metal, [0, 0.45, 0], torso);
  mesh(new RoundedBoxGeometry(0.7, 0.42, 0.06, 3, 0.03), metalLight, [0, 0.52, 0.27], torso);
  const core = mesh(new THREE.CylinderGeometry(0.1, 0.1, 0.05, 32), coreMat, [0, 0.52, 0.305], torso);
  core.rotation.x = Math.PI / 2;
  const coreRing = mesh(new THREE.TorusGeometry(0.15, 0.012, 8, 48), ringMat.clone(), [0, 0.52, 0.305], torso);
  const coreGlow = sprite(COLORS.orange, 0.7, [0, 0.52, 0.33], torso);
  for (const sx of [-1, 1]) {
    mesh(new THREE.BoxGeometry(0.03, 0.3, 0.02), accent, [sx * 0.42, 0.5, 0.275], torso);
  }
  const flag = addYemenFlag(torso); // belly, below the chest plate, on the flat front face

  // head
  mesh(new THREE.CylinderGeometry(0.09, 0.11, 0.16, 16), joint, [0, 0.95, 0], torso);
  const head = new THREE.Group();
  head.position.set(0, 1.02, 0);
  torso.add(head);
  mesh(new RoundedBoxGeometry(0.74, 0.56, 0.6, 4, 0.15), metal, [0, 0.3, 0], head);
  mesh(new RoundedBoxGeometry(0.62, 0.25, 0.1, 3, 0.05), visor, [0, 0.31, 0.27], head);
  const eyes = [];
  const eyeGlows = [];
  for (const sx of [-1, 1]) {
    const e = mesh(new RoundedBoxGeometry(0.13, 0.055, 0.02, 2, 0.02), eyeMat, [sx * 0.13, 0.31, 0.325], head);
    eyes.push(e);
    eyeGlows.push(sprite(COLORS.cyan, 0.32, [sx * 0.13, 0.31, 0.34], head));
    const ear = mesh(new THREE.CylinderGeometry(0.1, 0.1, 0.08, 24), joint, [sx * 0.39, 0.3, 0], head);
    ear.rotation.z = Math.PI / 2;
    const earRing = mesh(new THREE.TorusGeometry(0.075, 0.01, 8, 32), accent, [sx * 0.435, 0.3, 0], head);
    earRing.rotation.y = Math.PI / 2;
  }
  // emotion shapes (hidden until an emotion fades them in): happy "^" eyes, smile, frown
  const arcMat = eyeMat.clone();
  const happyEyes = [];
  for (const sx of [-1, 1]) {
    const arc = mesh(new THREE.TorusGeometry(0.055, 0.013, 8, 20, Math.PI), arcMat, [sx * 0.13, 0.29, 0.33], head);
    arc.scale.setScalar(0.001);
    happyEyes.push(arc);
  }
  const smile = mesh(new THREE.TorusGeometry(0.085, 0.011, 8, 24, Math.PI), arcMat, [0, 0.245, 0.33], head);
  smile.rotation.z = Math.PI; // lower half: a smile
  smile.scale.setScalar(0.001);
  const frown = mesh(new THREE.TorusGeometry(0.06, 0.009, 8, 24, Math.PI), arcMat, [0, 0.2, 0.33], head);
  frown.scale.setScalar(0.001);
  mesh(new THREE.CylinderGeometry(0.012, 0.012, 0.22, 8), metalLight, [0.18, 0.66, -0.08], head);
  const antennaTip = mesh(new THREE.SphereGeometry(0.035, 16, 16), coreMat, [0.18, 0.78, -0.08], head);
  sprite(COLORS.orange, 0.25, [0.18, 0.78, -0.08], head);

  // arms: shoulder pivot -> upper arm -> elbow pivot -> forearm -> hand
  const arms = {};
  for (const [side, sx] of [['right', -1], ['left', 1]]) {
    const shoulder = new THREE.Group();
    shoulder.position.set(sx * 0.62, 0.74, 0);
    torso.add(shoulder);
    mesh(new THREE.SphereGeometry(0.15, 24, 16), metalLight, [0, 0, 0], shoulder);
    mesh(new THREE.CapsuleGeometry(0.08, 0.28, 6, 16), metal, [0, -0.25, 0], shoulder);
    const elbow = new THREE.Group();
    elbow.position.set(0, -0.47, 0);
    shoulder.add(elbow);
    mesh(new THREE.SphereGeometry(0.09, 16, 12), joint, [0, 0, 0], elbow);
    mesh(new THREE.CapsuleGeometry(0.075, 0.24, 6, 16), metal, [0, -0.2, 0], elbow);
    const band = mesh(new THREE.TorusGeometry(0.082, 0.012, 8, 24), accent, [0, -0.12, 0], elbow);
    band.rotation.x = Math.PI / 2;
    const hand = new THREE.Group();
    hand.position.set(0, -0.42, 0);
    elbow.add(hand);
    mesh(new RoundedBoxGeometry(0.15, 0.16, 0.1, 2, 0.04), joint, [0, -0.04, 0], hand);
    arms[side] = { shoulder, elbow, hand, side: sx };
  }

  // aim beam (world-space, added by the scene)
  const beam = new THREE.Mesh(
    new THREE.CylinderGeometry(0.008, 0.008, 1, 8, 1, true),
    new THREE.MeshBasicMaterial({ color: COLORS.orange, transparent: true, opacity: 0, blending: THREE.AdditiveBlending, depthWrite: false }),
  );
  beam.visible = false;

  return {
    root, body, torso, head, arms, eyes, eyeGlows, beam, thruster,
    ringA, ringB, coreRing, coreGlow, floorGlow, antennaTip,
    happyEyes, smile, frown, flag,
    mats: { eyeMat, coreMat, ringMat, ringBMat, arcMat },
  };
}

/**
 * The status plate above the robot: a small canvas texture ("AGENT #: AURA-01 / STATUS: …",
 * plus the tracking line) that tilts with the robot's turn, rises with the lift, and
 * resolves new text through a short scrambled-character "glitch" (as in the reference video).
 * The canvas is redrawn only when the text changes or during a transition.
 */
function makeStatusPlate() {
  const canvas = document.createElement('canvas');
  canvas.width = 512;
  canvas.height = 160;
  const g = canvas.getContext('2d');
  const tex = new THREE.CanvasTexture(canvas);
  tex.colorSpace = THREE.SRGBColorSpace;
  const mesh = new THREE.Mesh(new THREE.PlaneGeometry(1.7, 0.53), new THREE.MeshBasicMaterial({ map: tex, transparent: true, depthWrite: false, side: THREE.DoubleSide }));
  mesh.name = 'statusPlate';
  mesh.renderOrder = 5;
  const GLYPHS = '#%&$@*+=<>/\\|_-01ABCDEFXYZ';
  const st = { text: '', from: '', since: -1e9, line: '', drawn: null, yaw: 0 };
  const draw = (text, line) => {
    g.clearRect(0, 0, 512, 160);
    g.fillStyle = 'rgba(8,10,14,0.62)';
    g.strokeStyle = 'rgba(235,240,245,0.85)';
    g.lineWidth = 3;
    g.beginPath();
    g.roundRect(6, 6, 500, 148, 10);
    g.fill();
    g.stroke();
    g.font = 'bold 30px ui-monospace, Consolas, monospace';
    g.fillStyle = '#ff8a2a';
    g.fillText('AGENT #: AURA-01', 22, 46);
    g.fillStyle = '#f2f5f8';
    g.fillText(`STATUS: ${text}`, 22, 92);
    g.font = '22px ui-monospace, Consolas, monospace';
    g.fillStyle = line.startsWith('[ok]') ? '#7dffb0' : '#ffb020';
    g.fillText(line, 22, 134);
    tex.needsUpdate = true;
  };
  return {
    mesh,
    update(status, now, pose) {
      const text = String(status.label ?? '').slice(0, 26);
      const line = String(status.line ?? '').slice(0, 34);
      if (text !== st.text) {
        st.from = st.text;
        st.text = text;
        st.since = now;
      }
      const k = Math.min(1, (now - st.since) / 450);
      let shown = text;
      if (k < 1) {
        // scramble: characters resolve left to right; unresolved ones flicker
        const n = Math.max(text.length, st.from.length);
        shown = '';
        for (let i = 0; i < n; i++) {
          if (i / n < k) shown += text[i] ?? '';
          else shown += GLYPHS[Math.floor(Math.random() * GLYPHS.length)];
        }
      }
      const key = `${shown}|${line}`;
      if (key !== st.drawn) {
        draw(shown, line);
        st.drawn = key;
      }
      // follow the robot: above its head, tilted part of the way with its turn
      const wrapped = Math.atan2(Math.sin(pose.yaw), Math.cos(pose.yaw));
      st.yaw += (THREE.MathUtils.clamp(wrapped, -1.1, 1.1) * 0.7 - st.yaw) * (1 - Math.exp(-8 * pose.dt));
      // beside the head, on the open side of the stage (above it, the plate would clip the top bar)
      mesh.position.set(pose.x - 1.45, 2.2 + pose.lift, pose.z + 0.2);
      mesh.rotation.set(-0.06, st.yaw, 0);
    },
  };
}

export class Scene3D {
  /**
   * @param {HTMLCanvasElement} canvas
   * @param {object} cfg scene configuration
   */
  constructor(canvas, cfg) {
    this.cfg = cfg;
    this.canvas = canvas;
    this.renderer = new THREE.WebGLRenderer({ canvas, alpha: true, antialias: true, powerPreference: 'high-performance' });
    this.renderer.setClearColor(0x000000, 0);
    this.renderer.outputColorSpace = THREE.SRGBColorSpace;
    this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
    this.renderer.toneMappingExposure = 1.05;

    this.scene = new THREE.Scene();
    this.pmrem = new THREE.PMREMGenerator(this.renderer);
    this.envTex = this.pmrem.fromScene(new RoomEnvironment(), 0.04).texture;
    this.scene.environment = this.envTex;
    this.scene.environmentIntensity = 0.55;

    this.camera = new THREE.PerspectiveCamera(30, 1, 0.1, 100);
    this.camera.position.set(0, 1.45, 7.2);
    this.cameraTarget = new THREE.Vector3(0, 1.15, 0);

    this.scene.add(new THREE.HemisphereLight('#9fb4c8', '#120c08', 0.5));
    const key = new THREE.DirectionalLight('#ffe2c4', 1.6);
    key.position.set(-3, 5, 4);
    this.scene.add(key);
    this.rim = new THREE.PointLight(COLORS.cyan, 18, 8, 2);
    this.rim.position.set(1.6, 2.4, -1.6);
    this.fill = new THREE.PointLight(COLORS.orange, 6, 6, 2);
    this.fill.position.set(-1.5, 0.6, 1.8);
    this.scene.add(this.rim, this.fill);

    this.glowTex = glowTexture();
    this.robot = buildProceduralRobot(this.glowTex);
    this.anchor = new THREE.Group(); // positioned on screen by layout()
    this.anchor.add(this.robot.root);
    this.anchor.add(this.rim, this.fill);
    this.scene.add(this.anchor);
    this.scene.add(this.robot.beam);
    // spin vortex: three additive rings around the robot, visible only when rotating fast
    this.vortex = new THREE.Group();
    this.vortexRings = [0.35, 1.15, 1.95].map((y, i) => {
      const m = new THREE.Mesh(
        new THREE.TorusGeometry(0.95 - i * 0.12, 0.01, 6, 64, Math.PI * 1.25),
        new THREE.MeshBasicMaterial({ color: COLORS.cyan.clone(), transparent: true, opacity: 0, blending: THREE.AdditiveBlending, depthWrite: false }),
      );
      m.rotation.x = Math.PI / 2;
      m.position.y = y;
      this.vortex.add(m);
      return m;
    });
    this.robot.root.add(this.vortex);
    this._emo = { laugh: 0, sad: 0 };
    this._react = { happy: 0, sad: 0, surprised: 0 }; // reactions to the user's expression
    this._lean = 0;
    this._puppet = { yaw: 0, lift: 0, x: 0 }; // marionette coupling to the tracked hand (visual only)
    this._squat = 0; // squat depth envelope (SQUAT action)
    this._run = 0; // running blend
    this._mimic = { yaw: 0, pitch: 0 }; // head imitation of the user (radians)
    this.mimicCfg = { enabled: true, yawGain: 1.6, pitchGain: 1.4, maxDeg: 40 };
    // status plate (video: "AGENT #: … / STATUS: …" above the robot, glitch-text transitions)
    this.plate = makeStatusPlate();
    this.anchor.add(this.plate.mesh);

    this.asset = { kind: 'procedural', reason: 'procedural robot (no model configured)', status: 'ready' };
    this.mixer = null;
    this.clips = [];

    // animation state
    this.lookTarget = new THREE.Vector3(0, 1.6, 3);
    this.hasLookTarget = false;
    this._eye = { color: COLORS.cyan.clone(), intensity: 2 };
    this._q = new THREE.Quaternion();
    this._v = new THREE.Vector3();
    this._lastAction = null;
    this._gait = 0; // 0..1 walk-gait blend (visual only)
    this._walkClip = null;
    this.width = 1;
    this.height = 1;
  }

  /** Loads the optional GLB model without blocking. */
  async loadRobot(loadGLTF) {
    const url = this.cfg.robotModelUrl;
    if (!url) return this.asset;
    this.asset = { kind: 'procedural', reason: `loading ${url}…`, status: 'loading' };
    const loader = loadGLTF ?? (async (u) => {
      const { GLTFLoader } = await import('three/addons/loaders/GLTFLoader.js');
      return new GLTFLoader().loadAsync(u);
    });
    const res = await resolveRobotAsset(url, { loadGLTF: loader });
    if (res.kind === 'gltf') {
      const model = res.gltf.scene;
      const box = new THREE.Box3().setFromObject(model);
      const size = box.getSize(new THREE.Vector3());
      const s = (2.4 / (size.y || 1)) * this.cfg.robotScale;
      model.scale.setScalar(s);
      model.position.y -= box.min.y * s;
      // keep rings, glow and beam; hide the procedural body
      this.robot.body.visible = false;
      this.robot.root.add(model);
      this.robot.gltf = model;
      this.clips = res.gltf.animations ?? [];
      if (this.clips.length) {
        this.mixer = new THREE.AnimationMixer(model);
        const idle = this.clips.find((c) => /idle/i.test(c.name)) ?? null;
        if (idle) this.mixer.clipAction(idle).play();
        const walk = this.clips.find((c) => /walk/i.test(c.name)) ?? null;
        if (walk) {
          this._walkClip = this.mixer.clipAction(walk);
          this._walkClip.setEffectiveWeight(0).play();
        }
      }
      this.asset = { kind: 'gltf', reason: `${url} (${this.clips.length} clips)`, status: 'ready' };
    } else {
      this.asset = { kind: 'procedural', reason: res.reason, status: url ? 'fallback' : 'ready' };
    }
    return this.asset;
  }

  resize(width, height) {
    this.width = width;
    this.height = height;
    this.renderer.setPixelRatio(Math.min(globalThis.devicePixelRatio || 1, this.cfg.pixelRatioCap));
    this.renderer.setSize(width, height, false);
    this.camera.aspect = width / height;
    this.camera.updateProjectionMatrix();
    this.layout();
  }

  /** Place the robot in the right part of the frame (centre-bottom on narrow screens). */
  layout() {
    const narrow = this.width < 760;
    const dist = this.camera.position.z;
    const halfH = Math.tan(THREE.MathUtils.degToRad(this.camera.fov / 2)) * dist;
    const halfW = halfH * this.camera.aspect;
    const ndcX = narrow ? 0.3 : 0.5;
    const scale = (narrow ? 0.7 : 0.86) * this.cfg.robotScale;
    this.anchor.position.set(ndcX * halfW, narrow ? 0.4 : 0.22, 0);
    this.anchor.scale.setScalar(scale);
    this.camera.lookAt(this.cameraTarget);
  }

  /** Display-space normalised point -> world point on the plane z = depth. */
  displayToWorld(p, map, depth = 2.2, out = new THREE.Vector3()) {
    const s = map.displayToScreen(p);
    const ndc = new THREE.Vector3((s.x / this.width) * 2 - 1, -(s.y / this.height) * 2 + 1, 0.5);
    ndc.unproject(this.camera);
    const dir = ndc.sub(this.camera.position).normalize();
    const t = (depth - this.camera.position.z) / dir.z;
    return out.copy(this.camera.position).addScaledVector(dir, t);
  }

  /**
   * @param {number} dt seconds
   * @param {number} now ms
   * @param {{ snapshot: any, hand: null | { palm: {x:number,y:number}, indexTip: {x:number,y:number} }, map: any }} input
   */
  update(dt, now, { snapshot, hand, map, gaze = null, user = null, status = null }) {
    const r = this.robot;
    const t = now / 1000;
    const state = snapshot.state;
    const action = snapshot.action;
    const hold = snapshot.hold;

    // ---- targets from the hand
    if (hand && map) {
      const aimPoint = hold?.name === 'AIM' ? hand.indexTip : hand.palm;
      this.displayToWorld(aimPoint, map, 2.2, this._v);
      this.lookTarget.lerp(this._v, damp(10, dt));
      this.hasLookTarget = true;
    } else {
      this.hasLookTarget = false;
    }

    // ---- yaw: the controller owns orientation (pose.yaw: GRAB_ROTATE / RESET_POSE);
    // SPIN_* is a pure animation offset that returns to the same orientation.
    const actionKey = action ? `${action.name}@${action.startedAt}` : null;
    if (actionKey !== this._lastAction) {
      this._lastAction = actionKey;
      this._playClipFor(action?.name);
    }
    let spin = 0;
    if (action?.name === 'SPIN_LEFT') spin = -Math.PI * 2 * easeInOut(action.progress);
    if (action?.name === 'SPIN_RIGHT') spin = Math.PI * 2 * easeInOut(action.progress);
    r.root.rotation.y = (snapshot.pose?.yaw ?? 0) + spin;
    const lift = snapshot.pose?.lift ?? 0; // marionette lift (controller-owned, smoothed)
    // puppet coupling (reference video): with a hand on the strings and nothing else running,
    // the body turns with the hand's roll, rises / sinks with its height and sways sideways
    // with it. A bounded, smoothed VISUAL offset (fast: time constant ~80 ms): the controller's
    // pose and safety state are untouched, it is off while locked / in error / during any
    // command, and relaxes when the hand is lost.
    const free = (state === 'TRACKING') && !action && !hold;
    const pyaw = free && hand && Number.isFinite(hand.roll) ? THREE.MathUtils.clamp(hand.roll * 0.8, -1.2, 1.2) : 0;
    const plift = free && hand?.palm && Number.isFinite(hand.palm.y) ? THREE.MathUtils.clamp((0.55 - hand.palm.y) * 0.5, -0.1, 0.22) : 0;
    const px = free && hand?.palm && Number.isFinite(hand.palm.x) ? THREE.MathUtils.clamp((hand.palm.x - 0.6) * 1.4, -0.45, 0.45) : 0;
    this._puppet.yaw = THREE.MathUtils.lerp(this._puppet.yaw, pyaw, damp(12, dt));
    this._puppet.lift = THREE.MathUtils.lerp(this._puppet.lift, plift, damp(12, dt));
    this._puppet.x = THREE.MathUtils.lerp(this._puppet.x, px, damp(10, dt));
    r.root.rotation.y += this._puppet.yaw;
    // squat (SQUAT action): down and up once, the body sinks and leans forward
    const sq = action?.name === 'SQUAT' ? Math.sin(Math.PI * Math.min(1, action.progress)) : 0;
    this._squat = THREE.MathUtils.lerp(this._squat, sq, damp(18, dt));
    const squat = this._squat < 0.005 ? 0 : this._squat;
    r.root.position.set((snapshot.pose?.x ?? 0) + this._puppet.x, lift + this._puppet.lift - 0.45 * squat, snapshot.pose?.z ?? 0);
    const mv = snapshot.movement ?? {};
    if (status) this.plate.update(status, now, { yaw: snapshot.pose?.yaw ?? 0, x: snapshot.pose?.x ?? 0, z: snapshot.pose?.z ?? 0, lift, dt });
    // the gait is shown only while really walking (never while rotating), with an amplitude
    // that follows the speed and a phase that comes from the distance actually walked
    const walking = (mv.state === 'WALKING' || mv.state === 'RUNNING') && (mv.speed ?? 0) > 0.02;
    this._gait = THREE.MathUtils.lerp(this._gait, walking ? Math.min(1, 0.45 + (mv.speed ?? 0)) : 0, damp(12, dt));
    this._run = THREE.MathUtils.lerp(this._run, mv.state === 'RUNNING' && walking ? 1 : 0, damp(10, dt));
    const gait = this._gait < 0.01 ? 0 : this._gait;
    const phase = mv.gaitPhase ?? 0;
    this._walkClip?.setEffectiveWeight(gait);

    // ---- emotions (envelopes from the LAUGH / SAD action progress)
    const env = (name) => (action?.name === name ? Math.min(1, action.progress * 6, (1 - action.progress) * 5) : 0);
    this._emo.laugh = THREE.MathUtils.lerp(this._emo.laugh, env('LAUGH'), damp(12, dt));
    this._emo.sad = THREE.MathUtils.lerp(this._emo.sad, env('SAD'), damp(8, dt));
    const laugh = this._emo.laugh < 0.01 ? 0 : this._emo.laugh;
    const sad = this._emo.sad < 0.01 ? 0 : this._emo.sad;
    // reactions to the user's facial expression: visual only, weaker than the gesture
    // emotions, suppressed while locked / in error / during a LAUGH or SAD action
    const ex = user?.expression;
    const canReact = state !== 'LOCKED' && state !== 'ERROR' && !laugh && !sad;
    for (const [k, label] of [['happy', 'HAPPY'], ['sad', 'SAD'], ['surprised', 'SURPRISED']]) {
      const target = canReact && ex?.label === label ? Math.min(1, ex.confidence ?? 0) : 0;
      this._react[k] = THREE.MathUtils.lerp(this._react[k], target, damp(label === 'SURPRISED' ? 14 : 10, dt));
    }
    // mirroring: a full smile / sad face on the robot when you smile / look sad
    const rh = this._react.happy < 0.01 ? 0 : this._react.happy;
    const rs = this._react.sad < 0.01 ? 0 : this._react.sad;
    const rx = this._react.surprised < 0.01 ? 0 : this._react.surprised;
    const joy = Math.max(laugh, rh);
    const gloom = Math.max(sad, rs);

    // ---- body bob / lean (stride bob + forward lean while walking; laugh bounce; sad slump)
    const locked = state === 'LOCKED';
    const lifting = hold?.name === 'GRAB_ROTATE';
    const bob = Math.sin(t * 1.7) * (locked ? 0.015 : 0.045) * (1 - gait) * (1 - sad) + Math.abs(Math.sin(phase)) * 0.06 * gait
      + Math.abs(Math.sin(t * 15)) * 0.05 * laugh + Math.abs(Math.sin(t * 9)) * 0.025 * rh + Math.sin(t * 1.3) * 0.012 * sad - 0.07 * sad
      + 0.05 * rx + (lifting ? Math.sin(t * 3.2) * 0.02 : 0) + Math.abs(Math.sin(phase)) * 0.06 * this._run;
    r.body.position.y = bob + (locked ? -0.04 : 0);
    // spin lean: the body leans out against the turn as the yaw rate grows (centrifugal)
    const spinK = mv.source === 'SPIN' ? Math.min(1, Math.abs(mv.rateDegS ?? 0) / 500) * Math.sign(mv.rateDegS ?? 0) : 0;
    this._lean = THREE.MathUtils.lerp(this._lean, spinK, damp(5, dt));
    r.torso.rotation.x = THREE.MathUtils.lerp(r.torso.rotation.x, locked ? 0.12 : 0.2 * gait + 0.18 * this._run - 0.12 * laugh + 0.2 * sad + 0.1 * rs - 0.12 * rx + (lifting ? 0.1 : 0) + 0.4 * squat, damp(squat ? 18 : 8, dt));
    r.torso.rotation.z = Math.sin(t * 11) * 0.05 * laugh - 0.22 * this._lean;

    // ---- spin vortex: opacity and ring speed follow the real yaw rate
    const rate = Math.abs(mv.rateDegS ?? 0);
    const vk = mv.source === 'SPIN' ? Math.min(1, Math.max(0, (rate - 90) / 400)) : 0;
    this.vortexRings.forEach((m, i) => {
      m.material.opacity = THREE.MathUtils.lerp(m.material.opacity, 0.32 * vk * (1 - i * 0.2), damp(10, dt));
      m.visible = m.material.opacity > 0.01;
      m.rotation.z -= dt * Math.sign(mv.rateDegS ?? 0) * (4 + i * 2) * vk;
    });

    // ---- head (in torso space): face-oriented gaze from AttentionController; the hand
    // only while aiming (mode HAND); an idle scan when no face is seen (NEUTRAL)
    let yaw = Math.sin(t * 0.35) * 0.35;
    let pitch = 0.05;
    const mode = gaze?.mode ?? (this.hasLookTarget ? 'HAND' : 'NEUTRAL');
    if (mode === 'HAND' && this.hasLookTarget && !locked && state !== 'ERROR' && !r.gltf) {
      const local = r.head.parent.worldToLocal(this.lookTarget.clone()).sub(r.head.position);
      local.y -= 0.3;
      yaw = THREE.MathUtils.clamp(Math.atan2(local.x, local.z), -0.9, 0.9);
      pitch = THREE.MathUtils.clamp(-Math.atan2(local.y, Math.hypot(local.x, local.z)), -0.5, 0.45);
    } else if (gaze && mode === 'USER') {
      // gaze angles are relative to the default facing: compensate the body's own yaw,
      // except while spinning (the head then rides with the body)
      const bodyYaw = mv.state === 'ROTATING' ? 0 : snapshot.pose?.yaw ?? 0;
      const rel = Math.atan2(Math.sin(gaze.yaw - bodyYaw), Math.cos(gaze.yaw - bodyYaw));
      yaw = THREE.MathUtils.clamp(rel, -0.85, 0.85);
      pitch = THREE.MathUtils.clamp(gaze.pitch, -0.45, 0.45);
    } else if (gaze) {
      yaw = gaze.yaw + Math.sin(t * 0.35) * 0.25;
      pitch = gaze.pitch + 0.05;
    }
    // gaze-aware eyes/head: when the user looks to one side, the robot glances the same way
    if (user?.gaze && mode === 'USER') yaw += THREE.MathUtils.clamp(user.gaze.x, -1, 1) * 0.18;
    // head imitation: you turn left / right / look up -> the robot does the same, as a mirror
    // image on screen (+cue yaw = toward screen right = +head yaw), on top of looking at you
    const mc = this.mimicCfg;
    const cue = mc.enabled && user?.headCue && mode === 'USER' && !locked && state !== 'ERROR' ? user.headCue : null;
    const lim = (mc.maxDeg * Math.PI) / 180;
    const UNIT = (25 * Math.PI) / 180; // one cue unit = 25 deg before gain
    this._mimic.yaw = THREE.MathUtils.lerp(this._mimic.yaw, cue ? THREE.MathUtils.clamp(cue.yaw * mc.yawGain * UNIT, -lim, lim) : 0, damp(14, dt));
    this._mimic.pitch = THREE.MathUtils.lerp(this._mimic.pitch, cue ? THREE.MathUtils.clamp(cue.pitch * mc.pitchGain * UNIT, -lim, lim) : 0, damp(14, dt));
    yaw += this._mimic.yaw;
    pitch += this._mimic.pitch;
    // emotions on top of the gaze: laughing throws the head back and shakes it; sad hangs it;
    // reactions: a concerned tilt for a sad face, a startled lift for surprise
    yaw = yaw * (1 - 0.6 * sad) + Math.sin(t * 9) * 0.12 * laugh;
    pitch = pitch * (1 - sad) + 0.5 * sad - 0.28 * laugh + Math.sin(t * 15) * 0.06 * laugh + 0.15 * rs - 0.18 * rx;
    r.head.rotation.z = THREE.MathUtils.lerp(r.head.rotation.z, 0.16 * rs, damp(6, dt));
    if (locked) {
      yaw = 0;
      pitch = 0.32;
    }
    r.head.rotation.y = THREE.MathUtils.lerp(r.head.rotation.y, THREE.MathUtils.clamp(yaw, -1.2, 1.2), damp(12, dt));
    r.head.rotation.x = THREE.MathUtils.lerp(r.head.rotation.x, THREE.MathUtils.clamp(pitch, -0.7, 0.7), damp(12, dt));

    // ---- arms
    this._poseArms(dt, t, state, action, hold, gait, phase, laugh, sad, this._lean, squat, this._run);

    // ---- beam (AIM)
    const aiming = hold?.name === 'AIM' && this.hasLookTarget;
    const beamMat = r.beam.material;
    beamMat.opacity = THREE.MathUtils.lerp(beamMat.opacity, aiming ? 0.75 : 0, damp(10, dt));
    r.beam.visible = beamMat.opacity > 0.02;
    if (r.beam.visible) {
      const from = r.arms.right.hand.getWorldPosition(new THREE.Vector3());
      const dir = this.lookTarget.clone().sub(from);
      const len = dir.length();
      r.beam.position.copy(from).addScaledVector(dir, 0.5);
      r.beam.quaternion.setFromUnitVectors(UP, dir.normalize());
      r.beam.scale.set(1, len, 1);
    }

    // ---- lights & emissives by state
    let eye = this._eyeFor(state, t);
    if (joy > 0.05) eye = { color: COLORS.amber, intensity: 3.0 + 0.6 * joy };
    else if (gloom > 0.05) eye = { color: SAD_BLUE, intensity: 1.1 + 0.6 * (1 - gloom) };
    else if (rx > 0.05) eye = { color: COLORS.white, intensity: 3.6 };
    this._eye.color.lerp(eye.color, damp(8, dt));
    this._eye.intensity = THREE.MathUtils.lerp(this._eye.intensity, eye.intensity, damp(8, dt));
    r.mats.eyeMat.emissive.copy(this._eye.color);
    r.mats.eyeMat.emissiveIntensity = this._eye.intensity;
    r.mats.arcMat.emissive.copy(this._eye.color);
    r.mats.arcMat.emissiveIntensity = this._eye.intensity;
    for (const g of r.eyeGlows) {
      g.material.color.copy(this._eye.color);
      g.material.opacity = Math.min(1, this._eye.intensity / 3);
    }
    const lidY = (locked ? 0.35 : state === 'ERROR' ? 0.6 : 1) * (1 - 0.85 * joy) * (1 - 0.45 * gloom) * (1 + 0.45 * rx);
    r.eyes.forEach((e, i) => {
      e.scale.y = THREE.MathUtils.lerp(e.scale.y, Math.max(0.05, lidY), damp(10, dt));
      // sad: outer ends droop; gaze: the eyes shift a little toward the target
      const side = i === 0 ? -1 : 1;
      e.rotation.z = THREE.MathUtils.lerp(e.rotation.z, -side * 0.38 * gloom, damp(8, dt));
      const gx = (gaze && gaze.mode !== 'NEUTRAL' ? THREE.MathUtils.clamp(gaze.yaw, -0.8, 0.8) * 0.03 : 0)
        + (user?.gaze && gaze?.mode === 'USER' ? THREE.MathUtils.clamp(user.gaze.x, -1, 1) * 0.025 : 0);
      const gy = gaze && gaze.mode !== 'NEUTRAL' ? -THREE.MathUtils.clamp(gaze.pitch, -0.45, 0.45) * 0.03 : 0;
      e.position.x = THREE.MathUtils.lerp(e.position.x, side * 0.13 + gx, damp(8, dt));
      e.position.y = THREE.MathUtils.lerp(e.position.y, 0.31 + gy - 0.02 * sad, damp(8, dt));
    });
    const show = (m, k) => m.scale.setScalar(Math.max(0.001, k));
    r.happyEyes.forEach((m) => show(m, joy));
    show(r.smile, joy);
    show(r.frown, gloom);

    const pulseRate = state === 'EXECUTING' ? 9 : state === 'CONTROL' ? 5 : 2;
    const pulse = 0.5 + 0.5 * Math.sin(t * pulseRate);
    r.mats.coreMat.emissiveIntensity = (locked ? 0.5 : 1.6) + pulse * (state === 'EXECUTING' ? 2.5 : 0.8);
    r.mats.coreMat.emissive.copy(state === 'ERROR' ? COLORS.red : COLORS.orange);
    r.coreGlow.material.opacity = 0.4 + 0.4 * pulse;

    const ringColor = state === 'ERROR' ? COLORS.red : locked ? COLORS.amber : sad > 0.05 ? SAD_BLUE : laugh > 0.05 ? COLORS.amber : state === 'CONTROL' ? COLORS.orange : COLORS.cyan;
    r.mats.ringMat.color.lerp(ringColor, damp(6, dt));
    r.ringA.scale.setScalar(1 + (state === 'CONTROL' ? 0.06 * pulse : 0));
    r.ringB.rotation.z += dt * (state === 'CONTROL' || state === 'EXECUTING' ? 1.6 : 0.25) * (1 + 2 * gait);
    this.rim.color.lerp(ringColor, damp(4, dt));

    this.mixer?.update(dt);
  }

  _eyeFor(state, t) {
    switch (state) {
      case 'TRACKING': return { color: COLORS.cyan, intensity: 3 };
      case 'CONTROL': return { color: COLORS.orange, intensity: 3.4 };
      case 'EXECUTING': return { color: COLORS.white, intensity: 3.2 + Math.sin(t * 12) * 0.6 };
      case 'LOCKED': return { color: COLORS.amber, intensity: 0.9 };
      case 'ERROR': return { color: COLORS.red, intensity: Math.sin(t * 20) > 0 ? 3 : 0.6 };
      default: return { color: COLORS.cyan, intensity: 1.4 };
    }
  }

  _poseArms(dt, t, state, action, hold, gait = 0, phase = 0, laugh = 0, sad = 0, lean = 0, squat = 0, run = 0) {
    const { right, left } = this.robot.arms;
    const k = damp(squat > 0 ? 18 : 12, dt);
    const sway = Math.sin(t * 1.7) * 0.05;
    const rest = (arm) => new THREE.Quaternion().setFromEuler(new THREE.Euler(sway, 0, arm.side * 0.14));

    let rightQ = rest(right);
    let rightElbow = new THREE.Euler(-0.25, 0, 0);
    let leftQ = rest(left);
    let leftElbow = new THREE.Euler(-0.25, 0, 0);

    if (state === 'LOCKED') {
      rightQ = new THREE.Quaternion().setFromEuler(new THREE.Euler(0, 0, -0.04));
      leftQ = new THREE.Quaternion().setFromEuler(new THREE.Euler(0, 0, 0.04));
      rightElbow = leftElbow = new THREE.Euler(0, 0, 0);
    } else if (squat > 0) {
      // squat: arms straight forward for balance while the body sinks
      const fwd = (side) => new THREE.Quaternion().setFromEuler(new THREE.Euler(-1.45, 0, side * 0.12));
      rightQ = rest(right).slerp(fwd(right.side), Math.min(1, squat * 1.6));
      leftQ = rest(left).slerp(fwd(left.side), Math.min(1, squat * 1.6));
      rightElbow = leftElbow = new THREE.Euler(-0.1, 0, 0);
    } else if (action?.name === 'WAVE') {
      const p = action.progress;
      const env = Math.min(1, p * 5, (1 - p) * 5);
      const raised = new THREE.Quaternion().setFromEuler(new THREE.Euler(0, 0, -2.55));
      rightQ = rest(right).slerp(raised, env);
      rightElbow = new THREE.Euler(0, 0, -0.35 * env + Math.sin(p * Math.PI * 8) * 0.55 * env);
    } else if (hold?.name === 'AIM' && this.hasLookTarget) {
      const shoulderWorld = right.shoulder.getWorldPosition(new THREE.Vector3());
      const dirWorld = this.lookTarget.clone().sub(shoulderWorld).normalize();
      const parentQ = right.shoulder.parent.getWorldQuaternion(new THREE.Quaternion());
      const dirLocal = dirWorld.applyQuaternion(parentQ.invert());
      rightQ = new THREE.Quaternion().setFromUnitVectors(DOWN, dirLocal);
      rightElbow = new THREE.Euler(0, 0, 0);
    } else if (hold?.name === 'GRAB_ROTATE') {
      // heavy lifting (video): arms bent up as if hauling a weight, a slow strain
      const strain = Math.sin(t * 3.2) * 0.08;
      rightQ = new THREE.Quaternion().setFromEuler(new THREE.Euler(-1.1 + strain, 0, -0.55));
      leftQ = new THREE.Quaternion().setFromEuler(new THREE.Euler(-1.1 - strain, 0, 0.55));
      rightElbow = leftElbow = new THREE.Euler(-1.35, 0, 0);
    } else if (Math.abs(lean) > 0.05) {
      // spinning: arms fling outward with the yaw rate
      const k = Math.min(1, Math.abs(lean));
      rightQ = rest(right).slerp(new THREE.Quaternion().setFromEuler(new THREE.Euler(0, 0, -1.25)), k);
      leftQ = rest(left).slerp(new THREE.Quaternion().setFromEuler(new THREE.Euler(0, 0, 1.25)), k);
      rightElbow = leftElbow = new THREE.Euler(-0.2 * k, 0, 0);
    } else if (action?.name === 'SPIN_LEFT' || action?.name === 'SPIN_RIGHT') {
      const env = Math.sin(action.progress * Math.PI);
      rightQ = rest(right).slerp(new THREE.Quaternion().setFromEuler(new THREE.Euler(0, 0, -1.2)), env);
      leftQ = rest(left).slerp(new THREE.Quaternion().setFromEuler(new THREE.Euler(0, 0, 1.2)), env);
    } else if (laugh > 0) {
      // laugh: hands on the belly, shoulders bouncing
      const shake = Math.sin(t * 15) * 0.12 * laugh;
      rightQ = rest(right).slerp(new THREE.Quaternion().setFromEuler(new THREE.Euler(-0.75 + shake, 0, 0.45)), laugh);
      leftQ = rest(left).slerp(new THREE.Quaternion().setFromEuler(new THREE.Euler(-0.75 - shake, 0, -0.45)), laugh);
      rightElbow = leftElbow = new THREE.Euler(-1.25 * laugh, 0, 0);
    } else if (sad > 0) {
      // sad: arms hang limp and slightly forward, a slow sway
      const sway = Math.sin(t * 1.3) * 0.05 * sad;
      rightQ = rest(right).slerp(new THREE.Quaternion().setFromEuler(new THREE.Euler(0.12 + sway, 0, -0.03)), sad);
      leftQ = rest(left).slerp(new THREE.Quaternion().setFromEuler(new THREE.Euler(0.12 - sway, 0, 0.03)), sad);
      rightElbow = leftElbow = new THREE.Euler(-0.05, 0, 0);
    } else if (gait > 0) {
      // walk: arms swing in opposition with the stride (phase = distance walked)
      // running: a wider pump with bent elbows
      const swing = Math.sin(phase) * (0.6 + 0.5 * run) * gait;
      rightQ = new THREE.Quaternion().setFromEuler(new THREE.Euler(swing, 0, -0.14));
      leftQ = new THREE.Quaternion().setFromEuler(new THREE.Euler(-swing, 0, 0.14));
      rightElbow = leftElbow = new THREE.Euler(-0.45 - 0.9 * run, 0, 0);
    } else if (state === 'TRACKING') {
      // attentive: slightly raised forearms
      rightElbow = leftElbow = new THREE.Euler(-0.55, 0, 0);
    }

    right.shoulder.quaternion.slerp(rightQ, k);
    left.shoulder.quaternion.slerp(leftQ, k);
    right.elbow.quaternion.slerp(this._q.setFromEuler(rightElbow), k);
    left.elbow.quaternion.slerp(this._q.setFromEuler(leftElbow), k);
  }

  _playClipFor(name) {
    if (!this.mixer || !name) return;
    const key = name.split('_')[0].toLowerCase();
    const clip = this.clips.find((c) => c.name.toLowerCase().includes(key));
    if (!clip) return;
    const a = this.mixer.clipAction(clip);
    a.reset().setLoop(THREE.LoopOnce, 1).play();
  }

  render() {
    this.renderer.render(this.scene, this.camera);
  }

  /** Normalised screen position (0..1, y down) of the robot's head, for the gaze geometry. */
  headScreen() {
    const p = this.robot.head.getWorldPosition(new THREE.Vector3());
    p.y += 0.3;
    p.project(this.camera);
    if (!Number.isFinite(p.x) || !Number.isFinite(p.y)) return null;
    return { x: (p.x + 1) / 2, y: (1 - p.y) / 2 };
  }

  /**
   * Screen positions (container px) of the five marionette anchors: left hand, head, chest
   * core, right hand, hover base, sorted left -> right on screen (so the strings from the
   * five fingertips, also sorted, never cross).
   */
  anchorPoints() {
    const r = this.robot;
    const v = new THREE.Vector3();
    const pts = [
      [r.arms.left.hand, 0],
      [r.head, 0.35],
      [r.coreRing, 0],
      [r.arms.right.hand, 0],
      [r.thruster, -0.05],
    ].map(([obj, dy]) => {
      obj.getWorldPosition(v);
      v.y += dy;
      v.project(this.camera);
      return { x: ((v.x + 1) / 2) * this.width, y: ((1 - v.y) / 2) * this.height };
    });
    return pts.every((p) => Number.isFinite(p.x) && Number.isFinite(p.y)) ? pts.sort((a, b) => a.x - b.x) : null;
  }

  /** Screen position (container px) of the robot's current look/aim target. */
  debugInfo() {
    const p = this.lookTarget.clone().project(this.camera);
    return {
      hasTarget: this.hasLookTarget,
      targetPx: { x: ((p.x + 1) / 2) * this.width, y: ((1 - p.y) / 2) * this.height },
      beamVisible: this.robot.beam.visible,
      yaw: this.robot.root.rotation.y,
    };
  }

  dispose() {
    this.scene.traverse((o) => {
      o.geometry?.dispose?.();
      const mats = Array.isArray(o.material) ? o.material : o.material ? [o.material] : [];
      for (const m of mats) {
        m.map?.dispose?.();
        m.dispose?.();
      }
    });
    this.glowTex.dispose();
    this.envTex.dispose();
    this.pmrem.dispose();
    this.renderer.dispose();
  }
}
