/**
 * Maps normalised camera-image coordinates to CSS pixels of a container
 * that shows the video with `object-fit: cover` (centre-cropped), optionally
 * mirrored. Used by the effects layer and the 3D scene so overlays line up
 * exactly with the visible hand.
 *
 * @param {{ videoWidth: number, videoHeight: number, width: number, height: number, mirror: boolean }} p
 */
export function createCoverMapping({ videoWidth, videoHeight, width, height, mirror }) {
  const vw = videoWidth || width;
  const vh = videoHeight || height;
  const s = Math.max(width / vw, height / vh);
  const dw = vw * s;
  const dh = vh * s;
  const ox = (width - dw) / 2;
  const oy = (height - dh) / 2;
  return {
    width,
    height,
    /** pixels per unit of normalised image height */
    unit: dh,
    /** raw (unmirrored) normalised image point -> container px */
    toScreen(p) {
      const x = mirror ? 1 - p.x : p.x;
      return { x: ox + x * dw, y: oy + p.y * dh };
    },
    /** display-space normalised point (already mirrored) -> container px */
    displayToScreen(p) {
      return { x: ox + p.x * dw, y: oy + p.y * dh };
    },
  };
}
