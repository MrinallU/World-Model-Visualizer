import * as ort from "onnxruntime-web";
import { canvasToCHWFloat } from "../utils/canvas";

/**
 * @param {{
 *   vaeEnc: ort.InferenceSession | null,
 *   lstm: ort.InferenceSession | null,
 *   queueRef: { current: (fn:()=>Promise<any>)=>Promise<any> } | null,
 *   gtSmallCanvasRef: React.RefObject<HTMLCanvasElement>,
 *
 *   latent: number[],
 *   setLatent: (fn:(prev:number[])=>number[])=>void,
 *   hData: Float32Array|null,
 *   setHData: (v:Float32Array|null)=>void,
 *   cData: Float32Array|null,
 *   setCData: (v:Float32Array|null)=>void,
 *
 *   setGtState: (fn:(prev:any)=>any)=>void,
 *   transitionModel: (state:any, action:number)=>any,
 *
 *   LATENT_DIM: number,
 *   NUM_LAYERS: number,
 *   HIDDEN_DIM: number,
 *   onError?: (msg:string)=>void
 * }} args
 */
export function useLatentRolloutControls(args) {
  const {
    vaeEnc,
    lstm,
    queueRef,
    gtSmallCanvasRef,

    latent,
    setLatent,
    hData,
    setHData,
    cData,
    setCData,

    setGtState,
    transitionModel,

    LATENT_DIM,
    NUM_LAYERS,
    HIDDEN_DIM,
    onError,
  } = args;

  const runQueued = (fn) => {
    const q = queueRef?.current;
    return q ? q(fn) : fn();
  };

  const syncGTToLatent = async () => {
    if (!vaeEnc || !gtSmallCanvasRef.current) return;

    try {
      await runQueued(async () => {
        const chw = canvasToCHWFloat(gtSmallCanvasRef.current); // Float32Array [3*H*W]
        const xTensor = new ort.Tensor("float32", chw, [1, 3, 96, 96]);

        const out = await vaeEnc.run({ x: xTensor });
        const mu = out["mu"];

        setLatent(() => Array.from(mu.data));
        setHData(null);
        setCData(null);
      });
    } catch (e) {
      console.error(e);
      onError?.(String(e));
    }
  };

  const stepWithAction = async (actionVal) => {
    // 1) GT state update always happens (no ORT needed)
    setGtState((prev) => transitionModel(prev, actionVal));

    // 2) LSTM latent step
    if (!lstm) return;

    try {
      await runQueued(async () => {
        const zArr = new Float32Array(LATENT_DIM);
        for (let i = 0; i < LATENT_DIM; i++) zArr[i] = latent[i];

        const latentTensor = new ort.Tensor("float32", zArr, [1, LATENT_DIM]);
        const actionTensor = new ort.Tensor("float32", new Float32Array([actionVal]), [1, 1]);

        let hArr = hData;
        let cArr = cData;
        if (!hArr || !cArr) {
          hArr = new Float32Array(NUM_LAYERS * 1 * HIDDEN_DIM);
          cArr = new Float32Array(NUM_LAYERS * 1 * HIDDEN_DIM);
        }

        const h0 = new ort.Tensor("float32", hArr, [NUM_LAYERS, 1, HIDDEN_DIM]);
        const c0 = new ort.Tensor("float32", cArr, [NUM_LAYERS, 1, HIDDEN_DIM]);

        const out = await lstm.run({
          latent: latentTensor,
          action: actionTensor,
          h0,
          c0,
        });

        setLatent(() => Array.from(out["next_latent"].data));
        setHData(new Float32Array(out["h1"].data));
        setCData(new Float32Array(out["c1"].data));
      });
    } catch (e) {
      console.error(e);
      onError?.(String(e));
    }
  };

  return { syncGTToLatent, stepWithAction };
}
