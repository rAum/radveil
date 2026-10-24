// Runs the RadVeil wasm pipeline off the main thread. Two copies of the
// scan are kept: `before` (pristine, for comparison renders) and `work`
// (anonymized in place).
import init, { Scan, warmUp } from "./pkg/radveil_wasm.js";

let before = null;
let work = null;
const ready = init().then(() => warmUp());

function drop() {
  before?.free();
  work?.free();
  before = work = null;
}

function rgba(scan, yaw, pitch, size) {
  const px = scan.render(yaw, pitch, size);
  return { yaw, pitch, size, px: px.buffer };
}

const handlers = {
  async load({ bytes }) {
    drop();
    const data = new Uint8Array(bytes);
    before = new Scan(data);
    work = new Scan(data);
    return { info: JSON.parse(before.info()) };
  },

  async render({ which, yaw, pitch, size }) {
    const scan = which === "after" ? work : before;
    const r = rgba(scan, yaw, pitch, size);
    return { image: r, transfer: [r.px] };
  },

  async anonymize({ mode, region, spareAir, levelFace, partialFace }) {
    const t0 = performance.now();
    const report = JSON.parse(work.anonymize(mode, region, spareAir, levelFace, partialFace));
    const ms = performance.now() - t0;
    // Before/after pairs at the angles the face was detected from.
    const pairs = report.views.map((v) => ({
      landmarks: v.landmarks,
      before: rgba(before, v.yaw_deg, v.pitch_deg, 512),
      after: rgba(work, v.yaw_deg, v.pitch_deg, 512),
    }));
    const file = work.encode(true);
    const transfer = [file.buffer];
    for (const p of pairs) transfer.push(p.before.px, p.after.px);
    return { report, ms, pairs, file: file.buffer, transfer };
  },
};

self.onmessage = async ({ data }) => {
  const { id, type, ...args } = data;
  try {
    await ready;
    const { transfer = [], ...result } = await handlers[type](args);
    self.postMessage({ id, ok: true, result }, transfer);
  } catch (e) {
    self.postMessage({ id, ok: false, error: String(e?.message ?? e) });
  }
};
