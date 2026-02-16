import React, { useMemo, useRef, useState } from "react";

import { useOrtRuntime } from "../hooks/useOrtRuntime";
import { useRunQueue } from "../hooks/useRunQueue";
import { useVaeDecoderOnly } from "../hooks/useVaeDecoderOnly";
import { useVaeDecode } from "../hooks/useVaeDecode";

import { useUiTheme } from "../components/theme";
import { Card, CardTitleRow } from "../components/Card";
import { Button } from "../components/Button";
import { Dot } from "../components/Pill";
import { CanvasFrame } from "../components/CanvasFrame";
import { PageHeader } from "../components/PageHeader";
import { SliderGrid } from "../components/SliderGrid";

const STATE_DIM = 3; // [x, y, angle]
const SCALE = 4;
const IMG_H = 100,
  IMG_W = 150;

const OBS_LOW = [-2.5, -2.5, -10.0, -10.0, -6.2831855, -10.0, -0.0, -0.0];
const OBS_HIGH = [2.5, 2.5, 10.0, 10.0, 6.2831855, 10.0, 1.0, 1.0];

// We use indices [0, 1, 4] => x, y, angle
const IDX_MAP = [0, 1, 4];

function randUniform(a, b) {
  return a + Math.random() * (b - a);
}

export default function LunarLander() {
  useOrtRuntime();

  const styles = useUiTheme({ imgW: IMG_W, imgH: IMG_H, scale: SCALE });

  const ortQueueRef = useRunQueue();

  const { vaeDec: session, loading, error, setError } = useVaeDecoderOnly("/lunar_lander_decoder.onnx");

  const [state, setState] = useState(() => Array(STATE_DIM).fill(0));

  const smallCanvasRef = useRef(null);
  const bigCanvasRef = useRef(null);

  useVaeDecode({
    vaeDec: session,
    latent: state,
    queueRef: ortQueueRef,
    smallRef: smallCanvasRef,
    bigRef: bigCanvasRef,
    onError: (msg) => setError?.(msg),
    inputName: "state",
    outputName: "recon",
  });

  const disabled = loading || !session;

  const resetState = () => setState(Array(STATE_DIM).fill(0));

  const randomState = () => {
    const xMin = OBS_LOW[0], xMax = OBS_HIGH[0];
    const yMin = OBS_LOW[1], yMax = OBS_HIGH[1];
    const aMin = OBS_LOW[4], aMax = OBS_HIGH[4];
    setState([randUniform(xMin, xMax), randUniform(yMin, yMax), randUniform(aMin, aMax)]);
  };

  const onChangeState = (i, val) => {
    setState((prev) => {
      const next = [...prev];
      next[i] = val;
      return next;
    });
  };

  const subtitle = useMemo(
    () => (
      <>
        Probe the decoder by manipulating <span style={styles.kbd}>[x, y, angle]</span> directly.
      </>
    ),
    [styles.kbd]
  );

  const rangeForIndex = (i) => {
    const obsIdx = IDX_MAP[i]; // 0, 1, 4
    const min = OBS_LOW[obsIdx];
    const max = OBS_HIGH[obsIdx];

    if (obsIdx === 0 || obsIdx === 1) return { min, max, step: 0.01 };   // x,y
    if (obsIdx === 4) return { min, max, step: 0.01 };                   // angle
    return { min, max, step: 0.05 };
  };

  const labelForIndex = (i) => {
    if (i === 0) return "x";
    if (i === 1) return "y";
    if (i === 2) return "angle";
    return `s[${i}]`;
  };

  const formatValue = (v, i) => {
    if (i === 0 || i === 1) return Number(v).toFixed(3);
    return Number(v).toFixed(3);
  };

  return (
    <div style={styles.page}>
      <PageHeader
        styles={styles}
        title="Decoder Input Visualizer"
        subtitle={subtitle}
        callout={
          <>
            <b>Flow:</b> drag sliders → decoder renders instantly. <b>Random</b> samples within Gym observation bounds.
          </>
        }
      />

      {loading && (
        <Card style={{ ...styles.card, marginBottom: 12 }}>
          <div style={{ fontWeight: 800, marginBottom: 6 }}>Loading ONNX model…</div>
          <div style={styles.smallText}>
            If this takes unusually long, check{" "}
            <span style={styles.kbd}>public/lunar_lander_decoder.onnx</span> exists and WASM assets aren’t blocked.
          </div>
        </Card>
      )}

      {error && <div style={styles.err}>Error: {error}</div>}

      <div style={{ ...styles.grid, gridTemplateColumns: "repeat(auto-fit, minmax(520px, 1fr))" }}>
        <Card style={styles.card}>
          <CardTitleRow style={styles.titleRow}>
            <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
              <Dot styles={styles} color="#0ea5e9" />
              <div>
                <div style={{ fontWeight: 850, letterSpacing: -0.2 }}>Input controls</div>
                <div style={styles.smallText}>Edit [x, y, angle] and observe decoded pixels</div>
              </div>
            </div>
          </CardTitleRow>

          <div style={{ display: "flex", gap: 10, marginTop: 12, flexWrap: "wrap" }}>
            <Button styles={styles} variant="primary" onClick={randomState} disabled={disabled}>
              Random
            </Button>
            <Button variant="danger" styles={styles} onClick={resetState} disabled={disabled}>
              Reset
            </Button>
          </div>

          <SliderGrid
            styles={styles}
            latent={state}
            onChangeLatent={onChangeState}
            title="State controls"
            labelForIndex={labelForIndex}
            formatValue={formatValue}
            rangeForIndex={rangeForIndex}
            description={
              <>
                Ranges match the Gym observation bounds:
                <span style={styles.kbd}> x,y ∈ [-2.5, 2.5]</span>,{" "}
                <span style={styles.kbd}> angle ∈ [-2π, 2π]</span>.
              </>
            }
          />
        </Card>

        {/* ===================== Output ===================== */}
        <Card style={styles.card}>
          <CardTitleRow style={styles.titleRow}>
            <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
              <Dot styles={styles} color="#22c55e" />
              <div>
                <div style={{ fontWeight: 850, letterSpacing: -0.2 }}>Decoded image</div>
                <div style={styles.smallText}>
                  Decoder output rendered at {IMG_W}×{IMG_H} then upscaled
                </div>
              </div>
            </div>
          </CardTitleRow>

          <canvas ref={smallCanvasRef} width={IMG_W} height={IMG_H} style={{ display: "none" }} />

          <CanvasFrame
            canvasRef={bigCanvasRef}
            width={IMG_W * SCALE}
            height={IMG_H * SCALE}
            style={styles.canvasFrame}
          />
        </Card>
      </div>
    </div>
  );
}

