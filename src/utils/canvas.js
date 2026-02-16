// src/visualizers/piwm/utils/canvas.js
export const DEFAULT_IMG_H = 96;
export const DEFAULT_IMG_W = 96;

export function blitUpscale(smallCanvas, bigCanvas) {
  const bctx = bigCanvas.getContext("2d");
  bctx.imageSmoothingEnabled = false;
  bctx.clearRect(0, 0, bigCanvas.width, bigCanvas.height);
  bctx.drawImage(
    smallCanvas,
    0,
    0,
    smallCanvas.width,
    smallCanvas.height,
    0,
    0,
    bigCanvas.width,
    bigCanvas.height
  );
}

/**
 * Draw CHW float tensor to canvases.
 * chwData: Float32Array in CHW format
 * Optional imgW/imgH override
 */
export function drawCHWFloatToCanvases(
  chwData,
  smallCanvas,
  bigCanvas,
  imgW,
  imgH
) {
  const W = imgW ?? smallCanvas.width ?? DEFAULT_IMG_W;
  const H = imgH ?? smallCanvas.height ?? DEFAULT_IMG_H;

  const sctx = smallCanvas.getContext("2d");
  const imageData = sctx.createImageData(W, H);
  const rgba = imageData.data;

  const planeSize = H * W;

  for (let i = 0; i < planeSize; i++) {
    const rr = chwData[0 * planeSize + i];
    const gg = chwData[1 * planeSize + i];
    const bb = chwData[2 * planeSize + i];

    const idx = i * 4;
    rgba[idx + 0] = Math.max(0, Math.min(255, Math.round(rr * 255)));
    rgba[idx + 1] = Math.max(0, Math.min(255, Math.round(gg * 255)));
    rgba[idx + 2] = Math.max(0, Math.min(255, Math.round(bb * 255)));
    rgba[idx + 3] = 255;
  }

  sctx.putImageData(imageData, 0, 0);
  blitUpscale(smallCanvas, bigCanvas);
}

/**
 * Convert canvas back to CHW float.
 */
export function canvasToCHWFloat(smallCanvas, imgW, imgH) {
  const W = imgW ?? smallCanvas.width ?? DEFAULT_IMG_W;
  const H = imgH ?? smallCanvas.height ?? DEFAULT_IMG_H;

  const ctx = smallCanvas.getContext("2d");
  const img = ctx.getImageData(0, 0, W, H).data;

  const planeSize = W * H;
  const chw = new Float32Array(3 * planeSize);

  for (let i = 0; i < planeSize; i++) {
    chw[0 * planeSize + i] = img[i * 4 + 0] / 255;
    chw[1 * planeSize + i] = img[i * 4 + 1] / 255;
    chw[2 * planeSize + i] = img[i * 4 + 2] / 255;
  }

  return chw;
}

