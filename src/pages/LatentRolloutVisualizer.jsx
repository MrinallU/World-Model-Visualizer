// src/LatentRolloutVisualizer.jsx
import React, { useEffect, useMemo, useRef, useState } from "react";

import { useOrtRuntime } from "../hooks/useOrtRuntime";
import { useRunQueue } from "../hooks/useRunQueue";
import { useVaeDecode } from "../hooks/useVaeDecode";
import { useLatentRolloutModels } from "../hooks/useLatentRolloutModels";
import { useLatentRolloutControls } from "../hooks/useLatentRolloutControls";

import { blitUpscale } from "../utils/canvas";
import { renderObservationTo96 } from "../utils/render.js";
import { transitionModel } from "../utils/physics";

import { useUiTheme } from "../components/theme";
import { Card, CardTitleRow } from "../components/Card";
import { Button } from "../components/Button";
import { Dot } from "../components/Pill";
import { CanvasFrame } from "../components/CanvasFrame";
import { PageHeader } from "../components/PageHeader";
import { SliderGrid } from "../components/SliderGrid";

const IMG_H = 96;
const IMG_W = 96;
const SCALE = 4;

const LATENT_DIM = 16;
const NUM_LAYERS = 2;
const HIDDEN_DIM = 128;

// render canvas (match python)
const RENDER_W = 600;
const RENDER_H = 400;

export default function LatentRolloutVisualizer() {
  // ORT runtime config + 1 global ORT queue (same pattern as PIWM)
  useOrtRuntime();
  const ortQueueRef = useRunQueue();

  // theme
  const styles = useUiTheme({ imgW: IMG_W, imgH: IMG_H, scale: SCALE });

  // models
  const { vaeEnc, vaeDec, lstm, loading, error, setError } =
    useLatentRolloutModels();

  // state
  const [gtState, setGtState] = useState({
    x: 0,
    xDot: 0,
    theta: 0,
    thetaDot: 0,
  });
  const [latent, setLatent] = useState(() => Array(LATENT_DIM).fill(0));
  const [hData, setHData] = useState(null);
  const [cData, setCData] = useState(null);

  // refs: GT renderer
  const gtRenderCanvasRef = useRef(null);
  const gtSmallCanvasRef = useRef(null);
  const gtBigCanvasRef = useRef(null);

  // refs: latent decoded output
  const latentSmallCanvasRef = useRef(null);
  const latentBigCanvasRef = useRef(null);

  // GT render effect
  useEffect(() => {
    if (!gtRenderCanvasRef.current || !gtSmallCanvasRef.current || !gtBigCanvasRef.current) return;

    renderObservationTo96({
      position: gtState.x,
      angle: gtState.theta,
      renderCanvas: gtRenderCanvasRef.current,
      smallCanvas96: gtSmallCanvasRef.current,
      renderW: RENDER_W,
      renderH: RENDER_H,
    });

    blitUpscale(gtSmallCanvasRef.current, gtBigCanvasRef.current);
  }, [gtState]);

  // latent -> pixels (reused hook)
  useVaeDecode({
    vaeDec,
    latent,
    queueRef: ortQueueRef,
    smallRef: latentSmallCanvasRef,
    bigRef: latentBigCanvasRef,
    onError: (msg) => setError?.(msg),
  });

  // controls (sync + action stepping) via hook
  const { syncGTToLatent, stepWithAction } = useLatentRolloutControls({
    vaeEnc,
    lstm,
    queueRef: ortQueueRef,
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

    onError: (msg) => setError?.(msg),
  });

  const resetGT = () => setGtState({ x: 0, xDot: 0, theta: 0, thetaDot: 0 });
  const resetLatent = () => {
    setLatent(Array(LATENT_DIM).fill(0));
    setHData(null);
    setCData(null);
  };

  const disabled = loading || !vaeEnc || !vaeDec || !lstm;

  const subtitle = useMemo(
    () => (
      <>
        Compare <span style={styles.kbd}>Ground Truth</span> observation rendering against{" "}
        <span style={styles.kbd}>LSTM</span> latent rollout decoded back to pixels. Use <b>Sync</b> to initialize latent
        from the GT image.
      </>
    ),
    [styles.kbd]
  );

  return (
    <div style={styles.page}>
      <PageHeader
        styles={styles}
        title="Latent Rollout Visualizer"
        subtitle={subtitle}
        callout={
          <>
            <b>Flow:</b> set GT sliders → <b>Sync GT image → latent</b> → step actions → compare decoded drift vs GT.
          </>
        }
      />

      {loading && (
        <Card style={{ ...styles.card, marginBottom: 12 }}>
          <div style={{ fontWeight: 800, marginBottom: 6 }}>Loading ONNX models…</div>
          <div style={styles.smallText}>If loading hangs, double-check model paths and WASM asset delivery.</div>
        </Card>
      )}

      {error && <div style={styles.err}>Error: {error}</div>}

      {/* ✅ Same clean PIWM-style responsive layout: auto-wrap via minmax */}
      <div
        style={{
          ...styles.grid,
          gridTemplateColumns: "repeat(auto-fit, minmax(520px, 1fr))",
        }}
      >
        {/* ===================== GT ===================== */}
        <Card style={styles.card}>
          <CardTitleRow style={styles.titleRow}>
            <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
              <Dot styles={styles} color="#22c55e" />
              <div>
                <div style={{ fontWeight: 850, letterSpacing: -0.2 }}>Ground Truth</div>
                <div style={styles.smallText}>Deterministic renderer driven by the GT state</div>
              </div>
            </div>

            <Button variant="danger" styles={styles} onClick={resetGT}>
              Reset
            </Button>
          </CardTitleRow>

          {/* hidden render canvases */}
          <canvas ref={gtRenderCanvasRef} width={RENDER_W} height={RENDER_H} style={{ display: "none" }} />
          <canvas ref={gtSmallCanvasRef} width={IMG_W} height={IMG_H} style={{ display: "none" }} />

          <CanvasFrame
            canvasRef={gtBigCanvasRef}
            width={IMG_W * SCALE}
            height={IMG_H * SCALE}
            style={styles.canvasFrame}
          />

          <div style={{ display: "flex", gap: 10, marginTop: 12, flexWrap: "wrap" }}>
            <Button variant="primary" styles={styles} onClick={syncGTToLatent} disabled={disabled}>
              Sync GT image → latent
            </Button>

            <div style={styles.smallText}>
              GT full state: (x={gtState.x.toFixed(2)}, xDot={gtState.xDot.toFixed(2)}, θ={gtState.theta.toFixed(2)},
              θDot={gtState.thetaDot.toFixed(2)})
            </div>
          </div>

          {/* Use your SliderGrid in values-mode (2 sliders) */}
          <SliderGrid
            styles={styles}
            title={null}
            description={null}
            columns={2}
            maxHeight={220}
            values={[gtState.x, gtState.theta]}
            labelForIndex={(i) => (i === 0 ? "Position" : "Angle")}
            formatValue={(v) => Number(v).toFixed(2)}
            rangeForIndex={(i) =>
              i === 0
                ? { min: -2.4, max: 2.4, step: 0.01, width: 240 }
                : { min: -3.14159, max: 3.14159, step: 0.01, width: 240 }
            }
            onChangeIndex={(i, val) => {
              if (i === 0) setGtState((prev) => ({ ...prev, x: val }));
              else setGtState((prev) => ({ ...prev, theta: val }));
            }}
          />
        </Card>

        {/* ===================== LSTM ===================== */}
        <Card style={styles.card}>
          <CardTitleRow style={styles.titleRow}>
            <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
              <Dot styles={styles} color="#0ea5e9" />
              <div>
                <div style={{ fontWeight: 850, letterSpacing: -0.2 }}>LSTM Latent Rollout</div>
                <div style={styles.smallText}>Latent transition + VAE decoder back to pixels</div>
              </div>
            </div>

            <Button variant="danger" styles={styles} onClick={resetLatent}>
              Reset latent & hidden
            </Button>
          </CardTitleRow>

          <canvas ref={latentSmallCanvasRef} width={IMG_W} height={IMG_H} style={{ display: "none" }} />
          <CanvasFrame
            canvasRef={latentBigCanvasRef}
            width={IMG_W * SCALE}
            height={IMG_H * SCALE}
            style={styles.canvasFrame}
          />

          <div style={{ display: "flex", gap: 10, marginTop: 12, flexWrap: "wrap" }}>
            <Button variant="primary" styles={styles} onClick={() => stepWithAction(0)} disabled={disabled}>
              Action: Left
            </Button>
            <Button variant="primary" styles={styles} onClick={() => stepWithAction(1)} disabled={disabled}>
              Action: Right
            </Button>
          </div>

          {/* SliderGrid in latent-mode */}
          <SliderGrid
            styles={styles}
            latent={latent}
            onChangeLatent={(i, val) => {
              setLatent((prev) => {
                const next = [...prev];
                next[i] = val;
                return next;
              });
            }}
          />
        </Card>
      </div>
    </div>
  );
}

