import { useEffect, useState } from "react";
import * as ort from "onnxruntime-web";

export function useLatentRolloutModels() {
  const [vaeEnc, setVaeEnc] = useState(null);
  const [vaeDec, setVaeDec] = useState(null);
  const [lstm, setLstm] = useState(null);

  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    let alive = true;

    async function load() {
      try {
        setLoading(true);
        const [enc, dec, lstmSession] = await Promise.all([
          ort.InferenceSession.create("/vae_encoder16.onnx"),
          ort.InferenceSession.create("/vae_decoder16.onnx"),
          ort.InferenceSession.create("/lstm_latent_step.onnx"),
        ]);

        if (!alive) return;
        setVaeEnc(enc);
        setVaeDec(dec);
        setLstm(lstmSession);
      } catch (e) {
        console.error(e);
        if (!alive) return;
        setError(String(e));
      } finally {
        if (!alive) return;
        setLoading(false);
      }
    }

    load();
    return () => {
      alive = false;
    };
  }, []);

  return { vaeEnc, vaeDec, lstm, loading, error, setError };
}
