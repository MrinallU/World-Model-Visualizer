import React, { useEffect, useRef, useState } from "react";
import * as ort from "onnxruntime-web";

const STATE_DIM = 2;
const LATENT_DIM = 16;
const IMG_H = 96;
const IMG_W = 96;
const SCALE = 4;

// must match how you exported lstm_latent_step.onnx
const NUM_LAYERS = 2;
const HIDDEN_DIM = 128;

function LatentRolloutVisualizer() {
  // ONNX sessions
  const [stateSession, setStateSession] = useState(null);   // decoder_interpretable.onnx
  const [encoderSession, setEncoderSession] = useState(null); // vae_encoder16.onnx
  const [decoderSession, setDecoderSession] = useState(null); // vae_decoder16.onnx
  const [lstmSession, setLstmSession] = useState(null);       // lstm_latent_step.onnx

  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  // ---------------- State latent sliders (position, angle) ----------------
  const [position, setPosition] = useState(0); // state[0]
  const [angle, setAngle] = useState(0);       // state[1]

  // Store the latest state->image float data so we can feed to VAE encoder
  const [stateImageTensor, setStateImageTensor] = useState(null); // Float32Array [3*96*96]

  const stateSmallCanvasRef = useRef(null);
  const stateBigCanvasRef = useRef(null);

  // ---------------- LSTM latent & hidden state ----------------
  const [latent, setLatent] = useState(() => Array(LATENT_DIM).fill(0));
  const [hData, setHData] = useState(null); // Float32Array for h
  const [cData, setCData] = useState(null); // Float32Array for c

  const latentSmallCanvasRef = useRef(null);
  const latentBigCanvasRef = useRef(null);

  async function runLatentDecoder() {
    if (!decoderSession || !latentSmallCanvasRef.current || !latentBigCanvasRef.current) return;

    const zData = new Float32Array(LATENT_DIM);
    for (let i = 0; i < LATENT_DIM; i++) {
      zData[i] = latent[i];
    }
    const zTensor = new ort.Tensor("float32", zData, [1, LATENT_DIM]);

    try {
      const outputs = await decoderSession.run({ z: zTensor });
      const xRecon = outputs["x_recon"]; // [1, 3, 96, 96]
      const data = xRecon.data;

      drawImageToCanvases(
        data,
        latentSmallCanvasRef.current,
        latentBigCanvasRef.current
      );
    } catch (e) {
      console.error(e);
      setError(String(e));
    }
  }

  async function runStateDecoder() {
    if (!stateSession || !stateSmallCanvasRef.current || !stateBigCanvasRef.current) return;

    // Build state vector: [pos, angle]
    const stateArr = new Float32Array(STATE_DIM);
    stateArr[0] = position;
    stateArr[1] = angle;

    const stateTensor = new ort.Tensor("float32", stateArr, [1, STATE_DIM]);

    try {
      const outputs = await stateSession.run({ state: stateTensor });
      const xRecon = outputs["image"]; // [1, 3, 96, 96]
      const data = xRecon.data;        // Float32Array

      // stash this for VAE encoder use
      setStateImageTensor(new Float32Array(data));

      drawImageToCanvases(
        data,
        stateSmallCanvasRef.current,
        stateBigCanvasRef.current
      );
    } catch (e) {
      console.error(e);
      setError(String(e));
    }
  }

  // Run *one* decode for both panels immediately after models load
  const initialDrawAfterLoad = async (stateDec, dec) => {
    try {
      // --- LEFT: state → image ---
      if (stateSmallCanvasRef.current && stateBigCanvasRef.current) {
        const stateArr = new Float32Array(STATE_DIM);
        stateArr[0] = position; // current slider values
        stateArr[1] = angle;

        const stateTensor = new ort.Tensor("float32", stateArr, [1, STATE_DIM]);
        const out = await stateDec.run({ state: stateTensor });
        const img = out["image"]; // [1, 3, 96, 96]

        setStateImageTensor(new Float32Array(img.data));
        drawImageToCanvases(
          img.data,
          stateSmallCanvasRef.current,
          stateBigCanvasRef.current
        );

      }

      // --- RIGHT: latent → image ---
      if (latentSmallCanvasRef.current && latentBigCanvasRef.current) {
        const zArr = new Float32Array(LATENT_DIM);
        for (let i = 0; i < LATENT_DIM; i++) {
          zArr[i] = latent[i]; // current latent sliders (initially all 0)
        }
        const zTensor = new ort.Tensor("float32", zArr, [1, LATENT_DIM]);
        const out2 = await dec.run({ z: zTensor });
        const img2 = out2["x_recon"]; // [1, 3, 96, 96]

        drawImageToCanvases(
          img2.data,
          latentSmallCanvasRef.current,
          latentBigCanvasRef.current
        );
      }
    } catch (e) {
      console.error(e);
      setError(String(e));
    }
  };

  async function loadModels() {
    try {
      setLoading(true);
      const [stateDec, enc, dec, lstm] = await Promise.all([
        ort.InferenceSession.create("/decoder_interpretable.onnx"), // your state→image model
        ort.InferenceSession.create("/vae_encoder16.onnx"),         // image→latent
        ort.InferenceSession.create("/vae_decoder16.onnx"),         // latent→image
        ort.InferenceSession.create("/lstm_latent_step.onnx"),      // (latent, action, h, c)→ next_latent
      ]);

      setStateSession(stateDec);
      setEncoderSession(enc);
      setDecoderSession(dec);
      setLstmSession(lstm);
      await initialDrawAfterLoad(stateDec, dec);
    } catch (e) {
      console.error(e);
      setError(String(e));
    }
    finally {

      setLoading(false);
    }
  }

  // ============================================================
  // Load all ONNX models once
  // ============================================================
  useEffect(() => {
    loadModels();

  }, []);

  // ============================================================
  // Helper: draw CHW float image to canvases
  // ============================================================
  const drawImageToCanvases = (data, smallCanvas, bigCanvas) => {
    if (!smallCanvas || !bigCanvas) return;

    const sctx = smallCanvas.getContext("2d");
    const imageData = sctx.createImageData(IMG_W, IMG_H);
    const rgba = imageData.data;
    const planeSize = IMG_H * IMG_W;

    for (let y = 0; y < IMG_H; y++) {
      for (let x = 0; x < IMG_W; x++) {
        const idxHW = y * IMG_W + x;
        const r = data[0 * planeSize + idxHW];
        const g = data[1 * planeSize + idxHW];
        const b = data[2 * planeSize + idxHW];

        const idxRGBA = idxHW * 4;
        rgba[idxRGBA + 0] = Math.max(0, Math.min(255, Math.round(r * 255)));
        rgba[idxRGBA + 1] = Math.max(0, Math.min(255, Math.round(g * 255)));
        rgba[idxRGBA + 2] = Math.max(0, Math.min(255, Math.round(b * 255)));
        rgba[idxRGBA + 3] = 255;
      }
    }

    sctx.putImageData(imageData, 0, 0);

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
  };

  // ============================================================
  // StateLatentVisualizer behavior (from your code)
  // ============================================================
  useEffect(() => {
    runStateDecoder();
  }, [stateSession, position, angle]);

  const resetState = () => {
    setPosition(0);
    setAngle(0);
  };

  // ============================================================
  // VAE decoder: latent → image (for LSTM panel)
  // ============================================================
  useEffect(() => {


    runLatentDecoder();
  }, [decoderSession, latent]);

  // ============================================================
  // LSTM stuff
  // ============================================================
  const handleLatentSliderChange = (idx, value) => {
    setLatent((prev) => {
      const next = [...prev];
      next[idx] = value;
      return next;
    });
    // optional: reset hidden when user manually edits latent
    // setHData(null);
    // setCData(null);
  };

  const resetLatentAndHidden = () => {
    setLatent(Array(LATENT_DIM).fill(0));
    setHData(null);
    setCData(null);
  };

  const stepWithAction = async (actionVal) => {
    if (!lstmSession) return;

    try {
      const zArr = new Float32Array(LATENT_DIM);
      for (let i = 0; i < LATENT_DIM; i++) {
        zArr[i] = latent[i];
      }
      const latentTensor = new ort.Tensor("float32", zArr, [1, LATENT_DIM]);

      const actionTensor = new ort.Tensor(
        "float32",
        new Float32Array([actionVal]),
        [1, 1]
      );

      let hArr = hData;
      let cArr = cData;
      if (!hArr || !cArr) {
        hArr = new Float32Array(NUM_LAYERS * 1 * HIDDEN_DIM); // zeros
        cArr = new Float32Array(NUM_LAYERS * 1 * HIDDEN_DIM);
      }
      const hTensor = new ort.Tensor("float32", hArr, [NUM_LAYERS, 1, HIDDEN_DIM]);
      const cTensor = new ort.Tensor("float32", cArr, [NUM_LAYERS, 1, HIDDEN_DIM]);

      const outputs = await lstmSession.run({
        latent: latentTensor,
        action: actionTensor,
        h0: hTensor,
        c0: cTensor,
      });

      const nextLatentTensor = outputs["next_latent"]; // [1,16]
      const h1Tensor = outputs["h1"];
      const c1Tensor = outputs["c1"];

      const nextLatentArr = Array.from(nextLatentTensor.data);
      setLatent(nextLatentArr);
      setHData(new Float32Array(h1Tensor.data));
      setCData(new Float32Array(c1Tensor.data));
    } catch (e) {
      console.error(e);
      setError(String(e));
    }
  };

  // ============================================================
  // Sync button: state image → VAE encoder → latent sliders
  // ============================================================
  const syncStateToLatent = async () => {
    if (!encoderSession) return;
    if (!stateImageTensor) {
      setError("No state decoder image available yet to encode.");
      return;
    }

    try {
      // stateImageTensor is [3*96*96]
      const xTensor = new ort.Tensor(
        "float32",
        stateImageTensor,
        [1, 3, IMG_H, IMG_W]
      );

      const outputs = await encoderSession.run({ x: xTensor });
      // assuming export_vae_encoder16.py named output 'mu'
      const mu = outputs["mu"]; // [1,16]
      const newLatent = Array.from(mu.data);

      setLatent(newLatent);
      // reset hidden state: new episode from this latent
      setHData(null);
      setCData(null);
    } catch (e) {
      console.error(e);
      setError(String(e));
    }
  };

  const disabled =
    loading ||
    !stateSession ||
    !encoderSession ||
    !decoderSession ||
    !lstmSession;

  // ============================================================
  // Render
  // ============================================================
  return (
    <div style={{ padding: 16, fontFamily: "sans-serif" }}>
      <h2>LSTM + Interpretable State Visualizer</h2>

      {loading && <p>Loading ONNX models…</p>}
      {error && (
        <p style={{ color: "red", whiteSpace: "pre-wrap" }}>
          Error: {error}
        </p>
      )}

      <div style={{ display: "flex", gap: 32, alignItems: "flex-start" }}>
        {/* LEFT: your StateLatentVisualizer behavior */}
        <div>
          <h3>Interpretable (State → Image)</h3>

          <div
            style={{
              display: "grid",
              gridTemplateColumns: "1fr 1fr",
              gap: 52,
              maxHeight: "100%",
              maxWidth: "100%",
              overflowY: "auto",
            }}
          >
            <div style={{ marginBottom: 24 }}>
              <label
                style={{
                  display: "block",
                  fontSize: 18,
                  marginBottom: 4,
                  fontFamily: "monospace",
                }}
              >
                Position = {position.toFixed(2)}
              </label>
              <input
                type="range"
                min={-2.14}
                max={2.14}
                step={0.01}
                value={position}
                onChange={(e) => setPosition(Number(e.target.value))}
                style={{
                  width: 200,
                  accentColor: "#2563eb",
                }}
              />
            </div>

            <div style={{ marginBottom: 24 }}>
              <label
                style={{
                  display: "block",
                  fontSize: 18,
                  marginBottom: 4,
                  fontFamily: "monospace",
                }}
              >
                Angle = {angle.toFixed(2)}
              </label>
              <input
                type="range"
                min={-3.14159}
                max={3.14159}
                step={0.01}
                value={angle}
                onChange={(e) => setAngle(Number(e.target.value))}
                style={{
                  width: 200,
                  accentColor: "#2563eb",
                }}
              />
            </div>
          </div>

          <p style={{ fontSize: 20, color: "#666", marginTop: 16 }}>
            Decoded image (interpretable):
          </p>
          <canvas
            ref={stateSmallCanvasRef}
            width={IMG_W}
            height={IMG_H}
            style={{ display: "none" }}
          />
          <canvas
            ref={stateBigCanvasRef}
            width={IMG_W * SCALE}
            height={IMG_H * SCALE}
            style={{
              width: `${IMG_W * SCALE}px`,
              height: `${IMG_H * SCALE}px`,
              imageRendering: "pixelated",
              border: "1px solid #ccc",
              backgroundColor: "#000",
            }}
          />
          <br />
          <button
            onClick={resetState}
            style={{ padding: "8px 16px", marginTop: 12, fontSize: 20 }}
          >
            Reset State
          </button>

          <div style={{ marginTop: 16 }}>
            <button
              onClick={syncStateToLatent}
              disabled={disabled}
              style={{ padding: "8px 16px", fontSize: 18 }}
            >
              Sync state image → LSTM latent
            </button>
            <p style={{ fontSize: 12, color: "#666", marginTop: 4 }}>
              Encodes the current interpretable image with the VAE encoder and
              updates the 16-D LSTM latent (and resets hidden state).
            </p>
          </div>
        </div>

        {/* RIGHT: LSTM latent visualizer */}
        <div style={{ display: "flex", gap: 32, flexDirection: "column" }}>
          <h3>LSTM Latent Rollout</h3>
          <div
            style={{
              display: "grid",
              gridTemplateColumns: "1fr 1fr",
              maxHeight: "100%",

              gap: 52,
              maxWidth: "100%",
              overflowY: "auto",
            }}
          >

            {latent.map((value, i) => (
              <div key={i}>
                <label
                  style={{
                    display: "block",
                    fontSize: 18,
                    marginBottom: 4,
                    fontFamily: "monospace",
                    color: "#000",
                  }}
                >
                  z[{i}] = {value.toFixed(2)}
                </label>
                <input
                  type="range"
                  min={-3}
                  max={3}
                  step={0.05}
                  value={value}
                  onChange={(e) =>
                    handleLatentSliderChange(i, Number(e.target.value))
                  }
                  style={{ width: 200, accentColor: "#2563eb" }}
                />
              </div>
            ))}

          </div>
        </div>

        <div>
          <p style={{ fontSize: 20, color: "#666", marginTop: 16 }}>
            Decoded image (LSTM latent):
          </p>
          <canvas
            ref={latentSmallCanvasRef}
            width={IMG_W}
            height={IMG_H}
            style={{ display: "none" }}
          />
          <canvas
            ref={latentBigCanvasRef}
            width={IMG_W * SCALE}
            height={IMG_H * SCALE}
            style={{
              width: `${IMG_W * SCALE}px`,
              height: `${IMG_H * SCALE}px`,
              imageRendering: "pixelated",
              border: "1px solid #ccc",
              backgroundColor: "#000",
            }}
          />

          <div style={{ marginTop: 16, display: "flex", gap: 12 }}>
            <button
              onClick={() => stepWithAction(0)}
              disabled={disabled}
              style={{ padding: "8px 16px", fontSize: 20 }}
            >
              Action: Left
            </button>
            <button
              onClick={() => stepWithAction(1)}
              disabled={disabled}
              style={{ padding: "8px 16px", fontSize: 20 }}
            >
              Action: Right
            </button>
          </div>

          <button
            onClick={resetLatentAndHidden}
            style={{ marginTop: 12, padding: "8px 16px", fontSize: 20 }}
          >
            Reset latent & hidden state
          </button>

        </div>
      </div>


    </div >
  );
}

export default LatentRolloutVisualizer;

