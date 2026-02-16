import { useEffect, useRef } from "react";
import * as ort from "onnxruntime-web";
import { drawCHWFloatToCanvases } from "../utils/canvas";

export function useVaeDecode({
  vaeDec,
  latent,
  queueRef,
  smallRef,
  bigRef,
  onError,
  inputName = "z",
  outputName = "x_recon",
}) {
  const tokenRef = useRef(0);

  useEffect(() => {
    if (!vaeDec || !smallRef.current || !bigRef.current) return;

    const token = ++tokenRef.current;

    queueRef.current(async () => {
      if (token !== tokenRef.current) return;

      const arr = new Float32Array(latent.length);
      for (let i = 0; i < latent.length; i++) arr[i] = latent[i];

      const tensor = new ort.Tensor("float32", arr, [1, latent.length]);

      const out = await vaeDec.run({ [inputName]: tensor });
      if (token !== tokenRef.current) return;

      const result = out[outputName];
      drawCHWFloatToCanvases(result.data, smallRef.current, bigRef.current);
    }).catch((e) => onError?.(String(e)));
  }, [vaeDec, latent, queueRef, smallRef, bigRef, onError, inputName, outputName]);
}

