// src/App.jsx
import React from "react";
import { Routes, Route, NavLink, Navigate } from "react-router-dom";
import VaeLatentVisualizer from "./VaeLatentVisualizer";
import StateLatentVisualizer from "./StateLatentVisualizer";

function App() {
  return (
    <div style={{ fontFamily: "sans-serif", fontSize: "25px" }}>
      {/* Simple top nav */}
      <nav
        style={{
          display: "flex",
          gap: 56,
          padding: "12px 16px",
          borderBottom: "1px solid #ddd",
          marginBottom: 16,
        }}
      >
        <NavLink
          to="/latent"
          style={({ isActive }) => ({
            textDecoration: "none",
            fontWeight: isActive ? "700" : "500",
            color: isActive ? "#2563eb" : "#444",
          })}
        >
          Latent Visualizer
        </NavLink>

        <NavLink
          to="/state"
          style={({ isActive }) => ({
            textDecoration: "none",
            fontWeight: isActive ? "700" : "500",
            color: isActive ? "#2563eb" : "#444",
          })}
        >
          Interpretable Visualizer
        </NavLink>
      </nav>

      {/* Route content */}
      <Routes>
        {/* default → latent page */}
        <Route path="/" element={<Navigate to="/latent" replace />} />
        <Route path="/latent" element={<VaeLatentVisualizer />} />
        <Route path="/state" element={<StateLatentVisualizer />} />

        {/* catch-all → latent */}
        <Route path="*" element={<Navigate to="/latent" replace />} />
      </Routes>
    </div>
  );
}

export default App;

