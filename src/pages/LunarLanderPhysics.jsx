import React, { useEffect, useMemo, useRef, useState } from "react";

import { IMG_H, IMG_W } from "../utils/canvas";
import { LunarLanderPhysics } from "../utils/LunarLanderPhysics";

import { useUiTheme } from "../components/theme";
import { Card, CardTitleRow } from "../components/Card";
import { Button } from "../components/Button";
import { Dot } from "../components/Pill";
import { CanvasFrame } from "../components/CanvasFrame";
import { PageHeader } from "../components/PageHeader";

const SCALE = 4;

// Discrete actions: 0=idle, 1=left, 2=main, 3=right
const A_IDLE = 0;
const A_LEFT = 1;
const A_MAIN = 2;
const A_RIGHT = 3;

function fmtObs(obs) {
  if (!obs || !obs.length) return "";
  return obs.map((v) => (Number.isFinite(v) ? v.toFixed(3) : "nan")).join(", ");
}

export default function LunarLander() {
  const styles = useUiTheme({ imgW: IMG_W, imgH: IMG_H, scale: SCALE });

  const env = useMemo(() => new LunarLanderPhysics(), []);
  const [obs, setObs] = useState(() => env.reset());
  const [totalReward, setTotalReward] = useState(0);
  const [lastReward, setLastReward] = useState(0);
  const [done, setDone] = useState(false);
  const [paused, setPaused] = useState(false);

  const smallCanvasRef = useRef(null);
  const bigCanvasRef = useRef(null);

  const inputRef = useRef({ left: false, right: false, main: false });

  const pickAction = () => {
    const k = inputRef.current;
    if (k.main) return A_MAIN;
    if (k.left && !k.right) return A_LEFT;
    if (k.right && !k.left) return A_RIGHT;
    return A_IDLE;
  };

  const reset = () => {
    const o = env.reset();
    setObs(o);
    setTotalReward(0);
    setLastReward(0);
    setDone(false);
    setPaused(false);
  };

  // Keyboard controls
  useEffect(() => {
    const down = (e) => {
      if (e.code === "ArrowLeft" || e.code === "KeyA") inputRef.current.left = true;
      if (e.code === "ArrowRight" || e.code === "KeyD") inputRef.current.right = true;
      if (e.code === "ArrowUp" || e.code === "Space" || e.code === "KeyW") inputRef.current.main = true;

      if (e.code === "KeyR") reset();
      if (e.code === "KeyP") setPaused((p) => !p);
    };
    const up = (e) => {
      if (e.code === "ArrowLeft" || e.code === "KeyA") inputRef.current.left = false;
      if (e.code === "ArrowRight" || e.code === "KeyD") inputRef.current.right = false;
      if (e.code === "ArrowUp" || e.code === "Space" || e.code === "KeyW") inputRef.current.main = false;
    };

    window.addEventListener("keydown", down);
    window.addEventListener("keyup", up);
    return () => {
      window.removeEventListener("keydown", down);
      window.removeEventListener("keyup", up);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Draw (small canvas -> upscale to big canvas)
  const draw = () => {
    const small = smallCanvasRef.current;
    const big = bigCanvasRef.current;
    if (!small || !big) return;

    const sctx = small.getContext("2d");
    const bctx = big.getContext("2d");
    if (!sctx || !bctx) return;

    sctx.clearRect(0, 0, IMG_W, IMG_H);

    const groundYPx = IMG_H - 14;
    const metersToPx = 8;
    const centerX = IMG_W / 2;

    const worldToScreen = (x, y) => {
      const sx = centerX + x * metersToPx;
      const sy = groundYPx - y * metersToPx;
      return [sx, sy];
    };

    // ground line
    sctx.beginPath();
    sctx.moveTo(0, groundYPx);
    sctx.lineTo(IMG_W, groundYPx);
    sctx.stroke();

    // lander body pose (expects env.lander to be a planck body)
    const lander = env.lander;
    if (lander && lander.getPosition) {
      const p = lander.getPosition();
      const a = lander.getAngle();

      const [cx, cy] = worldToScreen(p.x, p.y);

      sctx.save();
      sctx.translate(cx, cy);
      sctx.rotate(-a);

      const halfW = 0.6 * metersToPx;
      const halfH = 0.9 * metersToPx;

      sctx.strokeRect(-halfW, -halfH, 2 * halfW, 2 * halfH);

      // “nose” indicator
      sctx.beginPath();
      sctx.moveTo(0, -halfH);
      sctx.lineTo(0, -halfH - 6);
      sctx.stroke();

      sctx.restore();
    }

    // upscale
    bctx.clearRect(0, 0, IMG_W * SCALE, IMG_H * SCALE);
    bctx.imageSmoothingEnabled = false;
    bctx.drawImage(small, 0, 0, IMG_W * SCALE, IMG_H * SCALE);
  };

  // Main loop
  useEffect(() => {
    let raf = 0;

    const tick = () => {
      if (!paused && !done) {
        const a = pickAction();
        const res = env.step(a);

        // expects { obs, reward, done, info }
        if (res && res.obs) setObs(res.obs);
        if (res && typeof res.reward === "number") {
          setLastReward(res.reward);
          setTotalReward((t) => t + res.reward);
        }
        if (res && typeof res.done === "boolean") setDone(false);
      }

      draw();
      raf = requestAnimationFrame(tick);
    };

    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [paused, done, env]);

  const subtitle = useMemo(
    () => (
      <>
        Browser-native Lunar Lander: physics in <span style={styles.kbd}>JS</span>, rendered on{" "}
        <span style={styles.kbd}>canvas</span>, with Gym-like{" "}
        <span style={styles.kbd}>reset()</span>/<span style={styles.kbd}>step(action)</span>.
      </>
    ),
    [styles.kbd]
  );

  return (
    <div style={styles.page}>
      <PageHeader
        styles={styles}
        title="Lunar Lander"
        subtitle={subtitle}
        callout={
          <>
            <b>Controls:</b> <span style={styles.kbd}>←/→</span> (A/D) side,{" "}
            <span style={styles.kbd}>↑</span>/<span style={styles.kbd}>Space</span> (W) main,{" "}
            <span style={styles.kbd}>P</span> pause, <span style={styles.kbd}>R</span> reset.
          </>
        }
      />

      <div style={{ ...styles.grid, gridTemplateColumns: "repeat(auto-fit, minmax(520px, 1fr))" }}>
        <Card style={styles.card}>
          <CardTitleRow style={styles.titleRow}>
            <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
              <Dot styles={styles} color="#0ea5e9" />
              <div>
                <div style={{ fontWeight: 850, letterSpacing: -0.2 }}>Controls + telemetry</div>
                <div style={styles.smallText}>Keyboard or press-and-hold buttons</div>
              </div>
            </div>
          </CardTitleRow>

          <div style={{ display: "flex", gap: 10, marginTop: 12, flexWrap: "wrap" }}>
            <Button styles={styles} variant="primary" onClick={() => setPaused((p) => !p)}>
              {paused ? "Resume" : "Pause"}
            </Button>
            <Button styles={styles} variant="danger" onClick={reset}>
              Reset
            </Button>
          </div>

          <div style={{ display: "flex", gap: 10, marginTop: 12, flexWrap: "wrap" }}>
            <Button
              styles={styles}
              onMouseDown={() => (inputRef.current.left = true)}
              onMouseUp={() => (inputRef.current.left = false)}
              onMouseLeave={() => (inputRef.current.left = false)}
              onTouchStart={() => (inputRef.current.left = true)}
              onTouchEnd={() => (inputRef.current.left = false)}
              disabled={done}
            >
              Hold Left
            </Button>

            <Button
              styles={styles}
              onMouseDown={() => (inputRef.current.main = true)}
              onMouseUp={() => (inputRef.current.main = false)}
              onMouseLeave={() => (inputRef.current.main = false)}
              onTouchStart={() => (inputRef.current.main = true)}
              onTouchEnd={() => (inputRef.current.main = false)}
              disabled={done}
            >
              Hold Main
            </Button>

            <Button
              styles={styles}
              onMouseDown={() => (inputRef.current.right = true)}
              onMouseUp={() => (inputRef.current.right = false)}
              onMouseLeave={() => (inputRef.current.right = false)}
              onTouchStart={() => (inputRef.current.right = true)}
              onTouchEnd={() => (inputRef.current.right = false)}
              disabled={done}
            >
              Hold Right
            </Button>
          </div>

          <div style={{ marginTop: 14 }}>
            <div style={styles.smallText}>
              <b>Status:</b> {done ? "done" : paused ? "paused" : "running"}{" "}
              <span style={{ opacity: 0.7 }}>•</span> <b>Last r:</b> {lastReward.toFixed(3)}{" "}
              <span style={{ opacity: 0.7 }}>•</span> <b>Total:</b> {totalReward.toFixed(2)}
            </div>

            <div style={{ ...styles.smallText, marginTop: 10 }}>
              <b>obs:</b> {fmtObs(obs)}
            </div>
          </div>
        </Card>

        <Card style={styles.card}>
          <CardTitleRow style={styles.titleRow}>
            <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
              <Dot styles={styles} color="#22c55e" />
              <div>
                <div style={{ fontWeight: 850, letterSpacing: -0.2 }}>Simulation view</div>
                <div style={styles.smallText}>Rendered at {IMG_W}×{IMG_H}, upscaled ×{SCALE}</div>
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

