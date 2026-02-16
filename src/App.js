import React, { useMemo } from "react";
import { Routes, Route, NavLink, Navigate, useLocation } from "react-router-dom";

import VaeLatentVisualizer from "./pages/VaeLatentVisualizer.jsx";
import StateLatentVisualizer from "./pages/StateLatentVisualizer";
import LatentRolloutVisualizer from "./pages/LatentRolloutVisualizer";
import SemiInterpretableVisualizer from "./pages/SemiInterpretableVisualizer";
import PIWMVisualizer from "./pages/PIWMVisualizer";
import Guide from "./pages/Guide";
import Home from "./pages/Home";

// ✅ add this
import LunarLander from "./pages/LunarLander";

import { useUiTheme } from "./components/theme";

function TopNav() {
  const t = useUiTheme();
  const location = useLocation();

  const items = useMemo(() => t.navItems, [t.navItems]);

  const subtitle = useMemo(() => {
    const path = location.pathname;
    if (path.startsWith("/piwm")) return "PIWM Model";
    if (path.startsWith("/home")) return "Home";
    if (path.startsWith("/latent")) return "VAE encoder/decoder Latent Space";
    if (path.startsWith("/rollout")) return "Latent Rollouts";
    if (path.startsWith("/state")) return "Interpretable State Mapping";
    if (path.startsWith("/semi")) return "Semi-interpretable State Mapping";
    if (path.startsWith("/lunar")) return "Lunar Lander";
    return "User Guide";
  }, [location.pathname]);

  return (
    <div style={t.topWrap}>
      <div style={t.topBar}>
        <div style={t.brand}>
          <div style={t.logo} aria-hidden />
          <div style={t.brandText}>
            <p style={t.brandTitle}>World Model Visualizers</p>
            <p style={t.brandSub}>{subtitle}</p>
          </div>
        </div>

        <div style={t.navRow} aria-label="Primary navigation">
          {items.map((it) =>
            it.external ? (
              <a
                key={it.to}
                href={it.to}
                target="_blank"
                rel="noopener noreferrer"
                style={t.externalLink}
              >
                {it.label}
                <span style={t.externalIcon} aria-hidden>
                  ↗
                </span>
              </a>
            ) : (
              <NavLink
                key={it.to}
                to={it.to}
                style={({ isActive }) => ({
                  ...t.linkBase,
                  ...(isActive ? t.linkActive : t.linkInactive),
                  transform: isActive ? "translateY(-1px)" : "translateY(0px)",
                })}
              >
                <span style={t.dot(it.dot)} aria-hidden />
                {it.label}
              </NavLink>
            )
          )}

          {/* ✅ add nav item here if you don't want to touch t.navItems */}
          <NavLink
            to="/lunar"
            style={({ isActive }) => ({
              ...t.linkBase,
              ...(isActive ? t.linkActive : t.linkInactive),
              transform: isActive ? "translateY(-1px)" : "translateY(0px)",
            })}
          >
            <span style={t.dot("#22c55e")} aria-hidden />
            Lunar
          </NavLink>
        </div>
      </div>
    </div>
  );
}

function App() {
  const t = useUiTheme();

  return (
    <div style={t.appShell}>
      <TopNav />

      <div style={t.contentWrap}>
        <Routes>
          <Route path="/" element={<Navigate to="/home" replace />} />
          <Route path="/guide" element={<Guide />} />
          <Route path="/home" element={<Home />} />
          <Route path="/latent" element={<VaeLatentVisualizer />} />
          <Route path="/state" element={<StateLatentVisualizer />} />
          <Route path="/semi" element={<SemiInterpretableVisualizer />} />
          <Route path="/rollout" element={<LatentRolloutVisualizer />} />
          <Route path="/piwm" element={<PIWMVisualizer />} />
          <Route path="/lunar" element={<LunarLander />} />

          <Route path="*" element={<Navigate to="/guide" replace />} />
        </Routes>
      </div>
    </div>
  );
}

export default App;

