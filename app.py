"""Interface Streamlit do simulador didático de queda livre."""

from __future__ import annotations

import html
import json
import math

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components
from plotly.subplots import make_subplots

from atmosphere import AtmosphereConfig
from objects import OBJECT_PRESETS, make_custom_object
from physics import SimulationConfig, simulate_fall
from wind import WindConfig


ANIMATION_FRAME_DURATION_MS = 55
ANIMATION_PLOTLY_CONFIG = {
    "displayModeBar": True,
    "displaylogo": False,
    "scrollZoom": False,
    "doubleClick": False,
    "modeBarButtonsToRemove": [
        "zoom2d",
        "pan2d",
        "select2d",
        "lasso2d",
        "zoomIn2d",
        "zoomOut2d",
        "autoScale2d",
        "resetScale2d",
    ],
}

BG_MAIN = "#07111c"
BG_CARD = "#0f1d2b"
BG_CARD_SECONDARY = "#0b1723"
BORDER_CARD = "#21384d"
TEXT_PRIMARY = "#f1f6fb"
TEXT_SECONDARY = "#b8c7d8"
GRID_COLOR = "rgba(120, 150, 180, 0.18)"
AXIS_COLOR = "rgba(180, 200, 220, 0.35)"

PURPLE = "#8b5cf6"
BLUE = "#2687ff"
CYAN = "#22d3ee"
GREEN = "#4ade80"
PINK = "#ff6f91"
OBJECT_COLOR = "#ff7043"
GRAVITY_COLOR = "#ff3b30"


st.set_page_config(
    page_title="Simulador de Queda Livre",
    page_icon="🪂",
    layout="wide",
)

st.markdown(
    """
    <style>
    .block-container {
        padding-top: 1.25rem;
        padding-bottom: 2rem;
        max-width: 1500px;
    }

    .hero-card {
        background: linear-gradient(135deg, #0F172A 0%, #1D4ED8 58%, #38BDF8 100%);
        padding: 26px 32px;
        border-radius: 22px;
        color: #FFFFFF;
        box-shadow: 0 18px 45px rgba(15, 23, 42, 0.24);
        margin-bottom: 18px;
        border: 1px solid rgba(255, 255, 255, 0.18);
    }

    .hero-title {
        font-size: 2.35rem;
        font-weight: 850;
        margin-bottom: 8px;
        letter-spacing: -0.03em;
        color: #FFFFFF;
    }

    .hero-subtitle {
        font-size: 1.08rem;
        color: rgba(255, 255, 255, 0.94);
        max-width: 1050px;
        line-height: 1.45;
    }

    .section-card {
        background: var(--secondary-background-color);
        border: 1px solid rgba(148, 163, 184, 0.35);
        border-radius: 18px;
        padding: 18px 22px;
        box-shadow: 0 10px 28px rgba(15, 23, 42, 0.10);
        margin: 14px 0 18px 0;
    }

    .section-title {
        font-size: 1.28rem;
        font-weight: 780;
        color: var(--text-color);
        margin-bottom: 4px;
    }

    .section-subtitle {
        font-size: 0.98rem;
        color: var(--text-color);
        opacity: 0.82;
        line-height: 1.45;
    }

    div[data-testid="stMetric"] {
        background: var(--secondary-background-color);
        border: 1px solid rgba(148, 163, 184, 0.35);
        border-radius: 16px;
        padding: 14px 16px;
        box-shadow: 0 8px 20px rgba(15, 23, 42, 0.08);
    }

    div[data-testid="stMetricLabel"] {
        color: var(--text-color) !important;
        opacity: 0.72;
    }

    div[data-testid="stMetricValue"] {
        color: var(--text-color) !important;
        font-weight: 780;
    }

    [data-testid="stSidebar"] {
        background: var(--secondary-background-color);
        border-right: 1px solid rgba(148, 163, 184, 0.25);
    }

    .tip-box {
        background: rgba(59, 130, 246, 0.12);
        border: 1px solid rgba(59, 130, 246, 0.28);
        border-left: 5px solid #3B82F6;
        color: var(--text-color);
        padding: 14px 18px;
        border-radius: 14px;
        margin: 12px 0 18px 0;
        line-height: 1.45;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def format_br(value: float, decimals: int = 1) -> str:
    """Formata número com vírgula decimal."""
    return f"{value:.{decimals}f}".replace(".", ",")


def sample_simulation_frames(df: pd.DataFrame, max_frames: int = 120) -> pd.DataFrame:
    """Reduz a quantidade de pontos da simulação para animações leves."""
    if len(df) <= max_frames:
        return df.reset_index(drop=True)

    indices = np.linspace(0, len(df) - 1, max_frames, dtype=int)
    return df.iloc[np.unique(indices)].reset_index(drop=True)


def build_animation_panel_html(
    df: pd.DataFrame,
    object_name: str,
    initial_height: float,
    initial_vx: float,
    gravity: float,
) -> str:
    """Cria o painel visual de animação usando HTML, CSS, SVG e JavaScript."""
    anim_df = sample_simulation_frames(df, max_frames=180)
    records = anim_df[
        ["t", "y", "x", "distancia_horizontal", "vx", "vy", "speed"]
    ].round(4).to_dict(orient="records")

    max_height = max(float(initial_height), float(df["y"].max()), 1.0)
    max_distance = max(float(df["distancia_horizontal"].max()), 1e-9)
    impact = df.iloc[-1]
    escaped_name = html.escape(object_name)
    vertical_fall = abs(initial_vx) < 1e-9

    template = r'''
    <!DOCTYPE html>
    <html lang="pt-BR">
    <head>
      <meta charset="UTF-8" />
      <style>
        :root {
          --bg-main: #05080d;
          --bg-panel: #07111c;
          --bg-card: #0d1b2a;
          --bg-card-secondary: #102235;
          --border-soft: #1d3346;
          --border-strong: #2c5c8a;
          --text-primary: #f1f6fb;
          --text-secondary: #b8c7d8;
          --text-muted: #7f91a6;
          --scene-bg: #eaf7ff;
          --scene-grid: rgba(70, 100, 130, 0.12);
          --axis-color: #0c2a44;
          --ground: #7cb342;
          --ground-border: #2e5c22;
          --ground-text: #234b1c;
          --ball-fill: #ff8a24;
          --ball-border: #e65100;
          --vx-color: #2687ff;
          --vy-color: #ff3f73;
          --v-result-color: #b765ff;
          --gravity-color: #ff3b30;
          --trajectory-color: #0f4cab;
        }

        * { box-sizing: border-box; }

        body {
          margin: 0;
          font-family: Inter, Arial, sans-serif;
          background: var(--bg-main);
          color: var(--text-primary);
        }

        .freefall-animation-section {
          display: grid;
          grid-template-columns: 2.4fr 1fr;
          gap: 16px;
          width: 100%;
          padding: 16px;
          background: var(--bg-main);
        }

        .animation-card,
        .fall-panel {
          background:
            radial-gradient(circle at top left, rgba(34, 211, 238, 0.08), transparent 35%),
            linear-gradient(180deg, #08131f 0%, #05080d 100%);
          border: 1px solid var(--border-strong);
          border-radius: 14px;
          padding: 16px;
          box-shadow: 0 10px 30px rgba(0, 0, 0, 0.35);
        }

        .animation-header {
          display: flex;
          align-items: center;
          justify-content: space-between;
          gap: 16px;
          margin-bottom: 16px;
        }

        .animation-title-area,
        .fall-panel-title,
        .section-title-row {
          display: flex;
          align-items: center;
          gap: 12px;
        }

        .animation-title-area h2,
        .fall-panel-title h2 {
          font-size: 20px;
          font-weight: 700;
          color: var(--text-primary);
          margin: 0;
        }

        .animation-icon,
        .panel-icon,
        .section-icon {
          width: 40px;
          height: 40px;
          border-radius: 10px;
          background: rgba(38, 135, 255, 0.14);
          color: var(--vx-color);
          display: flex;
          align-items: center;
          justify-content: center;
          font-size: 22px;
          flex: 0 0 auto;
        }

        .section-icon {
          width: 30px;
          height: 30px;
          font-size: 17px;
          border-radius: 8px;
        }

        .animation-controls {
          display: flex;
          align-items: center;
          gap: 10px;
          flex-wrap: wrap;
          justify-content: flex-end;
        }

        .animation-controls button,
        .speed-control {
          background: linear-gradient(180deg, #13283c, #0d1b2a);
          border: 1px solid #2a4864;
          color: var(--text-primary);
          border-radius: 8px;
          padding: 10px 16px;
          font-size: 14px;
          cursor: pointer;
          min-height: 44px;
        }

        .animation-controls button:hover,
        .speed-control:hover {
          border-color: var(--vx-color);
          background: linear-gradient(180deg, #173454, #0d1b2a);
        }

        .speed-control {
          display: flex;
          flex-direction: column;
          align-items: center;
          gap: 1px;
          padding: 7px 18px;
        }

        .speed-control select {
          appearance: none;
          background: transparent;
          color: var(--text-primary);
          border: none;
          font-size: 16px;
          outline: none;
          text-align: center;
          cursor: pointer;
        }

        .speed-control span {
          font-size: 10px;
          color: var(--text-secondary);
        }

        .scene-container {
          position: relative;
          width: 100%;
          height: 620px;
          background: var(--scene-bg);
          border-radius: 12px;
          overflow: hidden;
          border: 1px solid rgba(180, 210, 240, 0.6);
        }

        svg { width: 100%; height: 100%; display: block; }

        .trajectory {
          stroke: var(--trajectory-color);
          stroke-width: 4;
          fill: none;
          stroke-linecap: round;
          filter: drop-shadow(0 0 4px rgba(15, 76, 171, 0.25));
        }

        .start-point {
          fill: var(--trajectory-color);
          stroke: #0a367a;
          stroke-width: 2;
        }

        .axis-line { stroke: var(--axis-color); stroke-width: 2; }
        .axis-label { fill: var(--axis-color); font-size: 15px; font-weight: 700; }
        .tick-label { fill: var(--axis-color); font-size: 14px; }
        .grid-line { stroke: rgba(70, 100, 130, 0.14); stroke-width: 1; stroke-dasharray: 4 4; }

        .ground { fill: var(--ground); }
        .ground-top-line { stroke: var(--ground-border); stroke-width: 3; }
        .ground-label { fill: var(--ground-text); font-size: 22px; font-weight: 800; letter-spacing: 1px; }

        .falling-object {
          fill: var(--ball-fill);
          stroke: var(--ball-border);
          stroke-width: 4;
          filter: drop-shadow(0 0 8px rgba(255, 138, 36, 0.45));
        }

        .vector-vx { stroke: var(--vx-color); stroke-width: 3; marker-end: url(#arrow-blue); }
        .vector-vy { stroke: var(--vy-color); stroke-width: 3; marker-end: url(#arrow-pink); }
        .vector-result { stroke: var(--v-result-color); stroke-width: 3; marker-end: url(#arrow-purple); }
        .vector-g { stroke: var(--gravity-color); stroke-width: 3; marker-end: url(#arrow-red); }

        .label-vx { fill: var(--vx-color); font-size: 16px; font-weight: 700; }
        .label-vy { fill: var(--vy-color); font-size: 16px; font-weight: 700; }
        .label-result { fill: var(--v-result-color); font-size: 16px; font-weight: 700; }
        .label-g { fill: var(--gravity-color); font-size: 16px; font-weight: 700; }

        .vector-legend {
          position: absolute;
          left: 24px;
          bottom: 20px;
          display: grid;
          grid-template-columns: repeat(2, auto);
          gap: 10px 18px;
          background: rgba(255, 255, 255, 0.55);
          border: 1px solid rgba(80, 120, 160, 0.18);
          border-radius: 10px;
          padding: 12px 16px;
          backdrop-filter: blur(6px);
        }

        .vector-legend-item {
          display: flex;
          align-items: center;
          gap: 8px;
          font-size: 14px;
          font-weight: 700;
          color: #0c2a44;
        }

        .legend-line {
          width: 34px;
          height: 3px;
          border-radius: 999px;
          display: inline-block;
        }

        .fall-panel { color: var(--text-primary); padding: 18px; }
        .fall-panel-title { margin-bottom: 20px; }

        .panel-section {
          border: 1px solid var(--border-strong);
          border-radius: 12px;
          padding: 14px;
          margin-bottom: 18px;
          background: rgba(7, 17, 28, 0.46);
        }

        .section-title-row {
          margin-bottom: 12px;
          color: var(--text-primary);
          font-size: 17px;
          font-weight: 700;
        }

        .section-title-row::after {
          content: "";
          flex: 1;
          height: 1px;
          background: var(--border-strong);
          margin-left: 8px;
        }

        .data-row,
        .feature-row {
          display: flex;
          align-items: center;
          justify-content: space-between;
          gap: 12px;
          background: linear-gradient(180deg, #102235, #0d1b2a);
          border: 1px solid #263f59;
          border-radius: 9px;
          padding: 12px 14px;
          margin-bottom: 8px;
        }

        .data-label,
        .feature-label {
          display: flex;
          align-items: center;
          gap: 10px;
          color: #e6eef7;
          font-size: 14px;
        }

        .data-icon,
        .feature-icon {
          width: 26px;
          text-align: center;
          font-size: 22px;
        }

        .data-row strong {
          font-size: 18px;
          font-weight: 800;
          color: var(--text-primary);
          white-space: nowrap;
        }

        .value-vx { color: var(--vx-color) !important; }
        .value-vy { color: var(--vy-color) !important; }
        .value-result { color: var(--v-result-color) !important; }
        .value-g { color: var(--gravity-color) !important; }

        .check-icon {
          width: 22px;
          height: 22px;
          border-radius: 50%;
          background: #4ade80;
          color: #052e16;
          display: flex;
          align-items: center;
          justify-content: center;
          font-weight: 900;
        }

        @media (max-width: 1000px) {
          .freefall-animation-section { grid-template-columns: 1fr; }
          .animation-header { align-items: flex-start; flex-direction: column; }
          .animation-controls { justify-content: flex-start; }
          .scene-container { height: 520px; }
        }
      </style>
    </head>
    <body>
      <div class="freefall-animation-section">
        <div class="animation-card">
          <div class="animation-header">
            <div class="animation-title-area">
              <div class="animation-icon">⌁</div>
              <h2>Animação visual da queda livre com vetores</h2>
            </div>
            <div class="animation-controls">
              <button id="playBtn">▶ Play</button>
              <button id="pauseBtn">⏸ Pausar</button>
              <button id="resetBtn">↻ Reiniciar</button>
              <div class="speed-control">
                <select id="speedSelect">
                  <option value="0.5">0.5x</option>
                  <option value="1" selected>1.0x</option>
                  <option value="1.5">1.5x</option>
                  <option value="2">2.0x</option>
                </select>
                <span>Velocidade</span>
              </div>
            </div>
          </div>

          <div class="scene-container">
            <svg viewBox="0 0 1000 600" preserveAspectRatio="none">
              <defs>
                <marker id="arrow-blue" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto" markerUnits="strokeWidth"><path d="M0,0 L0,6 L9,3 z" fill="#2687ff" /></marker>
                <marker id="arrow-pink" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto" markerUnits="strokeWidth"><path d="M0,0 L0,6 L9,3 z" fill="#ff3f73" /></marker>
                <marker id="arrow-purple" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto" markerUnits="strokeWidth"><path d="M0,0 L0,6 L9,3 z" fill="#b765ff" /></marker>
                <marker id="arrow-red" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto" markerUnits="strokeWidth"><path d="M0,0 L0,6 L9,3 z" fill="#ff3b30" /></marker>
              </defs>

              <g id="gridLayer"></g>
              <line class="axis-line" x1="70" y1="500" x2="70" y2="70" />
              <path class="axis-line" d="M70 70 L62 84 M70 70 L78 84" fill="none" />
              <text class="axis-label" x="38" y="52">Altura (m)</text>

              <path id="trajectoryPath" class="trajectory" d="" />
              <circle id="startPoint" class="start-point" cx="0" cy="0" r="7" />

              <rect class="ground" x="70" y="500" width="910" height="70" />
              <line class="ground-top-line" x1="70" y1="500" x2="980" y2="500" />
              <text class="ground-label" x="525" y="545" text-anchor="middle">SOLO</text>

              <circle id="fallingObject" class="falling-object" cx="0" cy="0" r="24" />

              <line id="vxVector" class="vector-vx" x1="0" y1="0" x2="0" y2="0" />
              <text id="vxLabel" class="label-vx" x="0" y="0">Vx</text>

              <line id="vyVector" class="vector-vy" x1="0" y1="0" x2="0" y2="0" />
              <text id="vyLabel" class="label-vy" x="0" y="0">Vy</text>

              <line id="resultVector" class="vector-result" x1="0" y1="0" x2="0" y2="0" />
              <text id="resultLabel" class="label-result" x="0" y="0">|v|</text>

              <line id="gVector" class="vector-g" x1="0" y1="0" x2="0" y2="0" />
              <text id="gLabel" class="label-g" x="0" y="0">g</text>
            </svg>

            <div class="vector-legend">
              <div class="vector-legend-item"><span class="legend-line" style="background:#2687ff"></span>Vx</div>
              <div class="vector-legend-item"><span class="legend-line" style="background:#ff3f73"></span>Vy</div>
              <div class="vector-legend-item"><span class="legend-line" style="background:#b765ff"></span>|v|</div>
              <div class="vector-legend-item"><span class="legend-line" style="background:#ff3b30"></span>g</div>
            </div>
          </div>
        </div>

        <div class="fall-panel">
          <div class="fall-panel-title"><div class="panel-icon">▣</div><h2>Painel da queda</h2></div>

          <div class="panel-section">
            <div class="section-title-row"><div class="section-icon">⌁</div><span>Dados da queda</span></div>
            <div class="data-row"><div class="data-label"><span class="data-icon">◷</span><span>Tempo (t)</span></div><strong id="timeValue">0,00 s</strong></div>
            <div class="data-row"><div class="data-label"><span class="data-icon">↕</span><span>Altura (h)</span></div><strong id="heightValue">0 m</strong></div>
            <div class="data-row"><div class="data-label"><span class="data-icon">↔</span><span>Distância horizontal (x)</span></div><strong id="distanceValue">0 m</strong></div>
            <div class="data-row"><div class="data-label"><span class="data-icon value-vx">→</span><span>Componente horizontal (Vx)</span></div><strong id="vxValue" class="value-vx">0 m/s</strong></div>
            <div class="data-row"><div class="data-label"><span class="data-icon value-vy">↓</span><span>Componente vertical (Vy)</span></div><strong id="vyValue" class="value-vy">0 m/s</strong></div>
            <div class="data-row"><div class="data-label"><span class="data-icon value-result">↗</span><span>Velocidade resultante |v|</span></div><strong id="speedValue" class="value-result">0 m/s</strong></div>
            <div class="data-row"><div class="data-label"><span class="data-icon value-g">g</span><span>Aceleração da gravidade (g)</span></div><strong id="gValue" class="value-g">9,8 m/s²</strong></div>
          </div>

          <div class="panel-section">
            <div class="section-title-row"><div class="section-icon">☆</div><span>Características</span></div>
            <div class="feature-row"><div class="feature-label"><span class="feature-icon" style="color:#4ade80">∅</span><span>Queda livre</span></div><span class="check-icon">✓</span></div>
            <div class="feature-row"><div class="feature-label"><span class="feature-icon" style="color:#2687ff">←</span><span id="horizontalFeature">Sem componente horizontal</span></div><span class="check-icon">✓</span></div>
            <div class="feature-row"><div class="feature-label"><span class="feature-icon" style="color:#fde047">↕</span><span>Movimento vertical</span></div><span class="check-icon">✓</span></div>
            <div class="feature-row"><div class="feature-label"><span class="feature-icon" style="color:#38bdf8">⌁</span><span>Aceleração constante</span></div><span class="check-icon">✓</span></div>
          </div>
        </div>
      </div>

      <script>
        const data = __DATA__;
        const h0 = __H0__;
        const maxDistance = __MAX_DISTANCE__;
        const gravity = __GRAVITY__;
        const verticalFall = __VERTICAL_FALL__;

        const topY = 100;
        const groundY = 500;
        const leftX = 150;
        const rightX = 820;
        const visualWidth = rightX - leftX;
        const finalT = Math.max(data[data.length - 1].t, 0.001);
        let frameIndex = 0;
        let timer = null;
        let speedFactor = 1;

        const gridLayer = document.getElementById('gridLayer');
        const trajectoryPath = document.getElementById('trajectoryPath');
        const startPoint = document.getElementById('startPoint');
        const fallingObject = document.getElementById('fallingObject');
        const vxVector = document.getElementById('vxVector');
        const vyVector = document.getElementById('vyVector');
        const resultVector = document.getElementById('resultVector');
        const gVector = document.getElementById('gVector');
        const vxLabel = document.getElementById('vxLabel');
        const vyLabel = document.getElementById('vyLabel');
        const resultLabel = document.getElementById('resultLabel');
        const gLabel = document.getElementById('gLabel');

        const formatNumber = (value, decimals = 1) => Number(value).toFixed(decimals).replace('.', ',');

        function mapY(height) {
          const safeHeight = Math.max(0, Math.min(h0, height));
          return topY + ((h0 - safeHeight) / h0) * (groundY - topY);
        }

        function mapX(row) {
          if (maxDistance < 0.001) {
            const p = Math.max(0, Math.min(1, row.t / finalT));
            return leftX + 0.78 * visualWidth * p;
          }
          const p = Math.max(0, Math.min(1, row.distancia_horizontal / maxDistance));
          return leftX + 0.78 * visualWidth * p;
        }

        function buildGrid() {
          gridLayer.innerHTML = '';
          const tickValues = [100, 80, 60, 40, 20, 0];
          tickValues.forEach((value) => {
            const normalizedValue = value > h0 ? h0 : value;
            const y = mapY(normalizedValue);
            const line = document.createElementNS('http://www.w3.org/2000/svg', 'line');
            line.setAttribute('class', 'grid-line');
            line.setAttribute('x1', '70');
            line.setAttribute('x2', '980');
            line.setAttribute('y1', y);
            line.setAttribute('y2', y);
            gridLayer.appendChild(line);

            const tick = document.createElementNS('http://www.w3.org/2000/svg', 'line');
            tick.setAttribute('class', 'axis-line');
            tick.setAttribute('x1', '62');
            tick.setAttribute('x2', '78');
            tick.setAttribute('y1', y);
            tick.setAttribute('y2', y);
            gridLayer.appendChild(tick);

            const label = document.createElementNS('http://www.w3.org/2000/svg', 'text');
            label.setAttribute('class', 'tick-label');
            label.setAttribute('x', '48');
            label.setAttribute('y', y + 5);
            label.setAttribute('text-anchor', 'end');
            label.textContent = String(value);
            gridLayer.appendChild(label);
          });
        }

        function pointFor(row) {
          return { x: mapX(row), y: mapY(row.y) };
        }

        function buildPath(index) {
          const points = data.slice(0, index + 1).map(pointFor);
          if (points.length === 0) return '';
          let path = `M ${points[0].x} ${points[0].y}`;
          for (let i = 1; i < points.length; i++) {
            path += ` L ${points[i].x} ${points[i].y}`;
          }
          return path;
        }

        function setLine(line, x1, y1, x2, y2) {
          line.setAttribute('x1', x1);
          line.setAttribute('y1', y1);
          line.setAttribute('x2', x2);
          line.setAttribute('y2', y2);
        }

        function updatePanel(row) {
          document.getElementById('timeValue').textContent = `${formatNumber(row.t, 2)} s`;
          document.getElementById('heightValue').textContent = `${formatNumber(Math.max(0, row.y), 0)} m`;
          document.getElementById('distanceValue').textContent = `${formatNumber(row.distancia_horizontal, 0)} m`;
          document.getElementById('vxValue').textContent = `${formatNumber(row.vx, 1)} m/s`;
          document.getElementById('vyValue').textContent = `${formatNumber(row.vy, 1)} m/s`;
          document.getElementById('speedValue').textContent = `${formatNumber(row.speed, 1)} m/s`;
          document.getElementById('gValue').textContent = `${formatNumber(gravity, 1)} m/s²`;
          document.getElementById('horizontalFeature').textContent = verticalFall ? 'Sem componente horizontal' : 'Com componente horizontal';
        }

        function drawFrame(index) {
          frameIndex = Math.max(0, Math.min(data.length - 1, index));
          const row = data[frameIndex];
          const p = pointFor(row);
          const path = buildPath(frameIndex);
          trajectoryPath.setAttribute('d', path);

          const initialPoint = pointFor(data[0]);
          startPoint.setAttribute('cx', initialPoint.x);
          startPoint.setAttribute('cy', initialPoint.y);

          fallingObject.setAttribute('cx', p.x);
          fallingObject.setAttribute('cy', p.y);

          const vxLen = Math.max(26, Math.min(85, Math.abs(row.vx) * 4));
          const vyLen = Math.max(45, Math.min(120, Math.abs(row.vy) * 2.2));
          const resultXLen = verticalFall ? 72 : Math.max(40, Math.min(110, Math.abs(row.vx) * 4));
          const resultYLen = Math.max(55, Math.min(125, Math.abs(row.vy) * 2.2));
          const signX = row.vx < 0 ? -1 : 1;

          setLine(vxVector, p.x, p.y, p.x + signX * vxLen, p.y);
          vxLabel.setAttribute('x', p.x + signX * vxLen + 8);
          vxLabel.setAttribute('y', p.y - 10);

          setLine(vyVector, p.x, p.y + 12, p.x, p.y + vyLen);
          vyLabel.setAttribute('x', p.x + 12);
          vyLabel.setAttribute('y', p.y + vyLen + 8);

          setLine(resultVector, p.x + 8, p.y + 8, p.x + signX * resultXLen, p.y + resultYLen);
          resultLabel.setAttribute('x', p.x + signX * resultXLen + 8);
          resultLabel.setAttribute('y', p.y + resultYLen + 8);

          setLine(gVector, p.x - 42, p.y + 20, p.x - 42, p.y + 100);
          gLabel.setAttribute('x', p.x - 58);
          gLabel.setAttribute('y', p.y + 108);

          updatePanel(row);
        }

        function play() {
          if (timer) return;
          timer = setInterval(() => {
            if (frameIndex >= data.length - 1) {
              pause();
              return;
            }
            drawFrame(frameIndex + 1);
          }, Math.max(16, 45 / speedFactor));
        }

        function pause() {
          if (timer) {
            clearInterval(timer);
            timer = null;
          }
        }

        function reset() {
          pause();
          drawFrame(0);
        }

        document.getElementById('playBtn').addEventListener('click', play);
        document.getElementById('pauseBtn').addEventListener('click', pause);
        document.getElementById('resetBtn').addEventListener('click', reset);
        document.getElementById('speedSelect').addEventListener('change', (event) => {
          speedFactor = Number(event.target.value);
          if (timer) {
            pause();
            play();
          }
        });

        buildGrid();
        drawFrame(data.length - 1);
      </script>
    </body>
    </html>
    '''

    return (
        template
        .replace("__DATA__", json.dumps(records, ensure_ascii=False))
        .replace("__H0__", json.dumps(max_height))
        .replace("__MAX_DISTANCE__", json.dumps(max_distance))
        .replace("__GRAVITY__", json.dumps(float(gravity)))
        .replace("__VERTICAL_FALL__", "true" if vertical_fall else "false")
        .replace("__OBJECT_NAME__", escaped_name)
    )


def add_dashboard_background(fig: go.Figure) -> None:
    """Adiciona fundo principal, barra superior e cards dos gráficos."""
    fig.add_shape(
        type="rect",
        xref="x domain",
        yref="y domain",
        x0=0,
        x1=1,
        y0=0.10,
        y1=0.90,
        fillcolor=BG_CARD,
        line=dict(color="#1d3346", width=1),
        layer="below",
    )

    for separator_x in [0.25, 0.50, 0.75]:
        fig.add_shape(
            type="line",
            xref="x domain",
            yref="y domain",
            x0=separator_x,
            x1=separator_x,
            y0=0.25,
            y1=0.75,
            line=dict(color="#1d3346", width=1),
            layer="above",
        )

    for xref, yref in [("x2 domain", "y2 domain"), ("x3 domain", "y3 domain"), ("x4 domain", "y4 domain")]:
        fig.add_shape(
            type="rect",
            xref=xref,
            yref=yref,
            x0=-0.14,
            x1=1.14,
            y0=-0.28,
            y1=1.34,
            fillcolor=BG_CARD,
            line=dict(color=BORDER_CARD, width=1),
            layer="below",
        )
        fig.add_shape(
            type="rect",
            xref=xref,
            yref=yref,
            x0=-0.14,
            x1=1.14,
            y0=-0.28,
            y1=1.34,
            fillcolor="rgba(12, 25, 38, 0.35)",
            line=dict(width=0),
            layer="below",
        )


def add_summary_annotations(fig: go.Figure, initial_height: float, impact_time: float, initial_vx: float) -> None:
    """Adiciona os quatro cards superiores de informação."""
    vertical_fall = "Sim" if abs(initial_vx) < 1e-9 else "Não"

    items = [
        (0.055, 0.085, "↑", "Altura inicial", f"{format_br(initial_height, 1)} m", BLUE, "rgba(38, 135, 255, 0.18)"),
        (0.305, 0.335, "◷", "Tempo de simulação", f"{format_br(impact_time, 2)} s", GREEN, "rgba(74, 222, 128, 0.16)"),
        (0.555, 0.585, "→", "Velocidade horizontal (Vx)", f"{format_br(initial_vx, 2)} m/s", CYAN, "rgba(34, 211, 238, 0.16)"),
        (0.805, 0.835, "↓", "Queda livre vertical", vertical_fall, PINK, "rgba(255, 111, 145, 0.16)"),
    ]

    for x_icon, x_text, icon, title, value, color, bg in items:
        fig.add_annotation(
            xref="x domain",
            yref="y domain",
            x=x_icon,
            y=0.50,
            text=f"<b>{icon}</b>",
            showarrow=False,
            font=dict(size=20, color=color),
            bgcolor=bg,
            bordercolor=bg,
            borderpad=8,
        )
        fig.add_annotation(
            xref="x domain",
            yref="y domain",
            x=x_text,
            y=0.61,
            text=title,
            showarrow=False,
            xanchor="left",
            align="left",
            font=dict(size=12, color="#d7e2ee"),
        )
        fig.add_annotation(
            xref="x domain",
            yref="y domain",
            x=x_text,
            y=0.39,
            text=f"<b>{value}</b>",
            showarrow=False,
            xanchor="left",
            align="left",
            font=dict(size=18, color=color),
        )


def add_chart_header(fig: go.Figure, xref: str, yref: str, icon: str, title: str, icon_color: str, icon_bg: str, legend_html: str) -> None:
    """Adiciona cabeçalho e legenda local dentro de um card de gráfico."""
    fig.add_annotation(
        xref=xref,
        yref=yref,
        x=-0.08,
        y=1.23,
        text=f"<b>{icon}</b>",
        showarrow=False,
        xanchor="left",
        yanchor="middle",
        font=dict(size=15, color=icon_color),
        bgcolor=icon_bg,
        bordercolor=icon_bg,
        borderpad=7,
    )
    fig.add_annotation(
        xref=xref,
        yref=yref,
        x=0.02,
        y=1.23,
        text=f"<b>{title}</b>",
        showarrow=False,
        xanchor="left",
        yanchor="middle",
        align="left",
        font=dict(size=14, color=TEXT_PRIMARY),
    )
    fig.add_annotation(
        xref=xref,
        yref=yref,
        x=0.02,
        y=1.10,
        text=legend_html,
        showarrow=False,
        xanchor="left",
        yanchor="top",
        align="left",
        font=dict(size=12, color="#dce7f3"),
    )


def add_chart_headers(fig: go.Figure) -> None:
    """Adiciona cabeçalhos aos três gráficos."""
    velocity_legend = (
        f"<span style='color:{PURPLE}'><b>■</b></span> |v| &nbsp;&nbsp;&nbsp;"
        f"<span style='color:{BLUE}'><b>■</b></span> Vx &nbsp;&nbsp;&nbsp;"
        f"<span style='color:{PINK}'><b>■</b></span> Vy"
    )
    height_legend = f"<span style='color:{CYAN}'><b>■</b></span> Altura"
    distance_legend = f"<span style='color:{GREEN}'><b>■</b></span> Distância horizontal"

    add_chart_header(fig, "x2 domain", "y2 domain", "↗", "Componentes da velocidade × tempo", PURPLE, "rgba(139, 92, 246, 0.18)", velocity_legend)
    add_chart_header(fig, "x3 domain", "y3 domain", "⌁", "Altura × tempo", CYAN, "rgba(34, 211, 238, 0.16)", height_legend)
    add_chart_header(fig, "x4 domain", "y4 domain", "→", "Distância horizontal × tempo", GREEN, "rgba(74, 222, 128, 0.16)", distance_legend)


def style_chart_axes(fig: go.Figure, row: int, col: int, x_title: str, y_title: str) -> None:
    """Padroniza eixos, grade e texto dos gráficos do painel."""
    fig.update_xaxes(
        title=x_title,
        title_font=dict(size=13, color="#edf4fb"),
        tickfont=dict(size=12, color="#dbe7f3"),
        color="#dbe7f3",
        gridcolor=GRID_COLOR,
        gridwidth=1,
        griddash="dash",
        zerolinecolor=AXIS_COLOR,
        linecolor=AXIS_COLOR,
        showline=True,
        fixedrange=True,
        row=row,
        col=col,
    )
    fig.update_yaxes(
        title=y_title,
        title_font=dict(size=13, color="#edf4fb"),
        tickfont=dict(size=12, color="#dbe7f3"),
        color="#dbe7f3",
        gridcolor=GRID_COLOR,
        gridwidth=1,
        griddash="dash",
        zerolinecolor=AXIS_COLOR,
        linecolor=AXIS_COLOR,
        showline=True,
        fixedrange=True,
        row=row,
        col=col,
    )


def make_value_label_trace(row: pd.Series) -> go.Scatter:
    """Cria etiquetas dos valores atuais no gráfico das velocidades."""
    return go.Scatter(
        x=[row["t"], row["t"], row["t"]],
        y=[row["speed"], row["vx"], row["vy"]],
        mode="text",
        text=[
            format_br(float(row["speed"]), 1),
            format_br(float(row["vx"]), 1),
            format_br(float(row["vy"]), 1),
        ],
        textposition=["middle right", "middle right", "middle right"],
        textfont=dict(color=TEXT_PRIMARY, size=11),
        hoverinfo="skip",
        showlegend=False,
    )


def build_professional_dashboard(full_df: pd.DataFrame, initial_height: float, initial_vx: float) -> go.Figure:
    """Constrói o painel solicitado, com barra superior e três gráficos sincronizados."""
    anim_df = sample_simulation_frames(full_df, max_frames=120)
    initial = anim_df.iloc[0]
    final = anim_df.iloc[-1]

    impact_time = float(final["t"])
    x_max = max(5.0, math.ceil(impact_time * 10) / 10)
    height_max = max(120.0, float(full_df["y"].max()) * 1.2)
    distance_max = float(full_df["distancia_horizontal"].max())
    distance_y_min = -1.5 if distance_max < 1e-9 else 0.0
    distance_y_max = 1.5 if distance_max < 1e-9 else max(1.5, distance_max * 1.15)

    fig = make_subplots(
        rows=2,
        cols=3,
        specs=[[{"colspan": 3}, None, None], [{}, {}, {}]],
        row_heights=[0.22, 0.78],
        horizontal_spacing=0.06,
        vertical_spacing=0.22,
    )

    fig.add_trace(go.Scatter(x=[0], y=[0], mode="markers", marker=dict(opacity=0), showlegend=False, hoverinfo="skip"), row=1, col=1)

    fig.add_trace(go.Scatter(x=[initial["t"]], y=[initial["speed"]], mode="lines", line=dict(color=PURPLE, width=3), hoverinfo="skip", showlegend=False), row=2, col=1)
    fig.add_trace(go.Scatter(x=[initial["t"]], y=[initial["vx"]], mode="lines", line=dict(color=BLUE, width=3), hoverinfo="skip", showlegend=False), row=2, col=1)
    fig.add_trace(go.Scatter(x=[initial["t"]], y=[initial["vy"]], mode="lines", line=dict(color=PINK, width=3), hoverinfo="skip", showlegend=False), row=2, col=1)
    fig.add_trace(
        go.Scatter(
            x=[initial["t"], initial["t"], initial["t"]],
            y=[initial["speed"], initial["vx"], initial["vy"]],
            mode="markers",
            marker=dict(size=9, color=[PURPLE, BLUE, PINK], line=dict(width=1, color=TEXT_PRIMARY)),
            hoverinfo="skip",
            showlegend=False,
        ),
        row=2,
        col=1,
    )
    fig.add_trace(make_value_label_trace(initial), row=2, col=1)

    fig.add_trace(go.Scatter(x=[initial["t"]], y=[initial["y"]], mode="lines", line=dict(color=CYAN, width=3), line_shape="spline", hoverinfo="skip", showlegend=False), row=2, col=2)
    fig.add_trace(go.Scatter(x=[initial["t"]], y=[initial["y"]], mode="markers", marker=dict(size=9, color=CYAN, line=dict(width=1, color=TEXT_PRIMARY)), hoverinfo="skip", showlegend=False), row=2, col=2)

    fig.add_trace(go.Scatter(x=[initial["t"]], y=[initial["distancia_horizontal"]], mode="lines", line=dict(color=GREEN, width=3), hoverinfo="skip", showlegend=False), row=2, col=3)
    fig.add_trace(go.Scatter(x=[initial["t"]], y=[initial["distancia_horizontal"]], mode="markers", marker=dict(size=9, color=GREEN, line=dict(width=1, color=TEXT_PRIMARY)), hoverinfo="skip", showlegend=False), row=2, col=3)

    frames: list[go.Frame] = []
    slider_steps = []

    for index, row in anim_df.iterrows():
        chart_df = anim_df.iloc[: index + 1]
        frame_name = str(index)
        frames.append(
            go.Frame(
                name=frame_name,
                data=[
                    go.Scatter(x=[0], y=[0], mode="markers", marker=dict(opacity=0), showlegend=False, hoverinfo="skip"),
                    go.Scatter(x=chart_df["t"], y=chart_df["speed"], mode="lines", line=dict(color=PURPLE, width=3), hoverinfo="skip", showlegend=False),
                    go.Scatter(x=chart_df["t"], y=chart_df["vx"], mode="lines", line=dict(color=BLUE, width=3), hoverinfo="skip", showlegend=False),
                    go.Scatter(x=chart_df["t"], y=chart_df["vy"], mode="lines", line=dict(color=PINK, width=3), hoverinfo="skip", showlegend=False),
                    go.Scatter(
                        x=[row["t"], row["t"], row["t"]],
                        y=[row["speed"], row["vx"], row["vy"]],
                        mode="markers",
                        marker=dict(size=9, color=[PURPLE, BLUE, PINK], line=dict(width=1, color=TEXT_PRIMARY)),
                        hoverinfo="skip",
                        showlegend=False,
                    ),
                    make_value_label_trace(row),
                    go.Scatter(x=chart_df["t"], y=chart_df["y"], mode="lines", line=dict(color=CYAN, width=3), line_shape="spline", hoverinfo="skip", showlegend=False),
                    go.Scatter(x=[row["t"]], y=[row["y"]], mode="markers", marker=dict(size=9, color=CYAN, line=dict(width=1, color=TEXT_PRIMARY)), hoverinfo="skip", showlegend=False),
                    go.Scatter(x=chart_df["t"], y=chart_df["distancia_horizontal"], mode="lines", line=dict(color=GREEN, width=3), hoverinfo="skip", showlegend=False),
                    go.Scatter(x=[row["t"]], y=[row["distancia_horizontal"]], mode="markers", marker=dict(size=9, color=GREEN, line=dict(width=1, color=TEXT_PRIMARY)), hoverinfo="skip", showlegend=False),
                ],
            )
        )
        if index % 6 == 0 or index == len(anim_df) - 1:
            slider_steps.append(
                {
                    "method": "animate",
                    "label": f"{row['t']:.1f}s",
                    "args": [
                        [frame_name],
                        {"mode": "immediate", "frame": {"duration": 0, "redraw": True}, "transition": {"duration": 0}},
                    ],
                }
            )

    fig.frames = frames

    add_dashboard_background(fig)
    add_summary_annotations(fig, initial_height=initial_height, impact_time=impact_time, initial_vx=initial_vx)
    add_chart_headers(fig)

    fig.update_layout(
        template="plotly_dark",
        height=650,
        margin=dict(l=28, r=28, t=24, b=72),
        paper_bgcolor=BG_MAIN,
        plot_bgcolor=BG_CARD_SECONDARY,
        font=dict(color=TEXT_PRIMARY, family="Inter, Arial, sans-serif"),
        showlegend=False,
        dragmode=False,
        updatemenus=[
            {
                "type": "buttons",
                "showactive": False,
                "direction": "left",
                "x": 0.02,
                "y": -0.09,
                "xanchor": "left",
                "yanchor": "top",
                "bgcolor": "#1D4ED8",
                "bordercolor": "#1E40AF",
                "borderwidth": 2,
                "font": {"color": "#FFFFFF", "size": 14},
                "buttons": [
                    {
                        "label": "▶ Play",
                        "method": "animate",
                        "args": [
                            None,
                            {"frame": {"duration": ANIMATION_FRAME_DURATION_MS, "redraw": True}, "fromcurrent": True, "transition": {"duration": 0}, "mode": "immediate"},
                        ],
                    },
                    {
                        "label": "⏸ Pausar",
                        "method": "animate",
                        "args": [[None], {"frame": {"duration": 0, "redraw": False}, "transition": {"duration": 0}, "mode": "immediate"}],
                    },
                ],
            }
        ],
        sliders=[
            {
                "active": 0,
                "x": 0.18,
                "y": -0.09,
                "xanchor": "left",
                "yanchor": "top",
                "len": 0.76,
                "pad": {"t": 0, "b": 0},
                "bgcolor": "#1d3346",
                "activebgcolor": BLUE,
                "bordercolor": BORDER_CARD,
                "currentvalue": {"prefix": "Tempo: ", "suffix": "", "font": {"size": 14, "color": TEXT_PRIMARY}},
                "steps": slider_steps,
            }
        ],
    )

    fig.update_xaxes(range=[0, 1], showgrid=False, zeroline=False, showticklabels=False, fixedrange=True, row=1, col=1)
    fig.update_yaxes(range=[0, 1], showgrid=False, zeroline=False, showticklabels=False, fixedrange=True, row=1, col=1)

    style_chart_axes(fig, row=2, col=1, x_title="Tempo (s)", y_title="Velocidade (m/s)")
    style_chart_axes(fig, row=2, col=2, x_title="Tempo (s)", y_title="Altura (m)")
    style_chart_axes(fig, row=2, col=3, x_title="Tempo (s)", y_title="Distância horizontal (m)")

    fig.update_xaxes(range=[0, x_max], dtick=1, row=2, col=1)
    fig.update_yaxes(range=[-60, 60], dtick=20, row=2, col=1)

    fig.update_xaxes(range=[0, x_max], dtick=1, row=2, col=2)
    fig.update_yaxes(range=[0, height_max], row=2, col=2)

    fig.update_xaxes(range=[0, x_max], dtick=1, row=2, col=3)
    fig.update_yaxes(range=[distance_y_min, distance_y_max], row=2, col=3)

    return fig


def apply_dark_chart_style(fig: go.Figure) -> go.Figure:
    """Aplica a mesma identidade visual aos gráficos completos."""
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=BG_MAIN,
        plot_bgcolor=BG_CARD_SECONDARY,
        font=dict(color=TEXT_PRIMARY),
        legend=dict(
            font=dict(color=TEXT_PRIMARY),
            bgcolor="rgba(0,0,0,0)",
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0,
        ),
    )
    fig.update_xaxes(
        gridcolor=GRID_COLOR,
        griddash="dash",
        zerolinecolor=AXIS_COLOR,
        color=TEXT_PRIMARY,
        title_font=dict(color=TEXT_PRIMARY),
        tickfont=dict(color=TEXT_SECONDARY),
    )
    fig.update_yaxes(
        gridcolor=GRID_COLOR,
        griddash="dash",
        zerolinecolor=AXIS_COLOR,
        color=TEXT_PRIMARY,
        title_font=dict(color=TEXT_PRIMARY),
        tickfont=dict(color=TEXT_SECONDARY),
    )
    return fig


st.markdown(
    """
    <div class="hero-card">
        <div class="hero-title">Simulador didático de queda livre</div>
        <div class="hero-subtitle">
            Visualize o corpo em queda, os vetores físicos e acompanhe, logo abaixo, os gráficos de velocidade,
            altura e distância horizontal no modelo ideal, sem resistência do ar e sem vento.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("⚙️ Configuração da queda")

    preset_name = st.selectbox(
        "Objeto visual",
        options=list(OBJECT_PRESETS.keys()) + ["Personalizado"],
        index=1 if "Bola de futebol" in OBJECT_PRESETS else 0,
        help="Na queda livre ideal, a massa e o formato não alteram o tempo de queda. O objeto serve apenas como representação visual.",
    )

    if preset_name == "Personalizado":
        object_name = st.text_input("Nome do objeto", value="Objeto")
        falling_object = make_custom_object(name=object_name, mass=1.0, area=1.0, drag_coefficient=0.0)
    else:
        preset = OBJECT_PRESETS[preset_name]
        falling_object = make_custom_object(name=preset.name, mass=1.0, area=1.0, drag_coefficient=0.0)

    st.header("📌 Condições iniciais")
    initial_height = st.number_input("Altura inicial (m)", min_value=0.1, value=100.0, step=10.0)
    initial_vx = st.number_input(
        "Velocidade horizontal inicial (m/s)",
        value=0.0,
        step=1.0,
        help="Use zero para queda livre vertical. Valores diferentes de zero geram deslocamento horizontal linear.",
    )
    initial_vy = st.number_input(
        "Velocidade vertical inicial (m/s)",
        value=0.0,
        step=1.0,
        help="Valor positivo para cima e negativo para baixo.",
    )
    gravity = st.number_input("Aceleração da gravidade g (m/s²)", min_value=0.0, value=9.81, step=0.01)

    st.header("🧮 Precisão numérica")
    dt = st.number_input("Passo de tempo dt (s)", min_value=0.001, max_value=1.0, value=0.01, step=0.001, format="%.3f")
    max_time = st.number_input("Tempo máximo de simulação (s)", min_value=1.0, value=300.0, step=10.0)

atmosphere_config = AtmosphereConfig(model="constant", rho0=0.0)
wind_config = WindConfig(model="none")
sim_config = SimulationConfig(
    initial_height=initial_height,
    initial_vx=initial_vx,
    initial_vy=initial_vy,
    gravity=gravity,
    dt=dt,
    max_time=max_time,
    use_air_resistance=False,
)

try:
    data = simulate_fall(
        falling_object=falling_object,
        sim_config=sim_config,
        atmosphere_config=atmosphere_config,
        wind_config=wind_config,
    )
except ValueError as error:
    st.error(str(error))
    st.stop()

if not data:
    st.warning("A simulação não gerou dados. Verifique a altura inicial e os parâmetros numéricos.")
    st.stop()

df = pd.DataFrame(data)
df["distancia_horizontal"] = (df["x"] - df["x"].iloc[0]).abs()
impact = df.iloc[-1]

metric_cols = st.columns(5)
metric_cols[0].metric("Tempo de queda", f"{impact['t']:.2f} s")
metric_cols[1].metric("Velocidade final", f"{impact['speed']:.2f} m/s")
metric_cols[2].metric("Vx final", f"{impact['vx']:.2f} m/s")
metric_cols[3].metric("Vy final", f"{impact['vy']:.2f} m/s")
metric_cols[4].metric("Distância horizontal", f"{impact['distancia_horizontal']:.2f} m")

st.markdown(
    """
    <div class="section-card">
        <div class="section-title">Painel de animação da queda</div>
        <div class="section-subtitle">
            A animação mostra a trajetória, o objeto, o solo, os vetores físicos e os dados da queda em tempo real.
            Este painel fica acima dos gráficos para facilitar a leitura didática do movimento.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

animation_html = build_animation_panel_html(
    df=df,
    object_name=falling_object.name,
    initial_height=initial_height,
    initial_vx=initial_vx,
    gravity=gravity,
)
components.html(animation_html, height=760, scrolling=False)

st.markdown(
    """
    <div class="section-card">
        <div class="section-title">Painel gráfico sincronizado</div>
        <div class="section-subtitle">
            Os três gráficos abaixo continuam com o estilo profissional solicitado. Eles mostram velocidade, altura e distância horizontal,
            com a possibilidade de acompanhar a construção das curvas pelo botão Play do painel gráfico.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

panel_fig = build_professional_dashboard(
    full_df=df,
    initial_height=initial_height,
    initial_vx=initial_vx,
)
st.plotly_chart(panel_fig, use_container_width=True, config=ANIMATION_PLOTLY_CONFIG)

st.markdown(
    """
    <div class="tip-box">
        <b>Observação:</b> se <b>Vx = 0</b>, a distância horizontal permanece em zero. Se <b>Vx ≠ 0</b>, o gráfico da distância horizontal é uma reta crescente.
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="section-card">
        <div class="section-title">Gráficos completos</div>
        <div class="section-subtitle">
            Estes gráficos mostram o resultado final completo da simulação, sem depender do controle de animação.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

left, middle, right = st.columns(3)

with left:
    st.subheader("Componentes da velocidade × tempo")
    velocity_df = df[["t", "speed", "vx", "vy"]].rename(
        columns={
            "speed": "|v| velocidade resultante",
            "vx": "Vx componente horizontal",
            "vy": "Vy componente vertical",
        }
    )
    fig_speed = px.line(
        velocity_df,
        x="t",
        y=["|v| velocidade resultante", "Vx componente horizontal", "Vy componente vertical"],
        labels={"t": "Tempo (s)", "value": "Velocidade (m/s)", "variable": "Grandeza"},
        color_discrete_sequence=[PURPLE, BLUE, PINK],
    )
    st.plotly_chart(apply_dark_chart_style(fig_speed), use_container_width=True)

with middle:
    st.subheader("Altura × tempo")
    fig_height = px.line(df, x="t", y="y", labels={"t": "Tempo (s)", "y": "Altura (m)"})
    fig_height.update_traces(line=dict(color=CYAN, width=4))
    st.plotly_chart(apply_dark_chart_style(fig_height), use_container_width=True)

with right:
    st.subheader("Distância horizontal × tempo")
    fig_distance = px.line(
        df,
        x="t",
        y="distancia_horizontal",
        labels={"t": "Tempo (s)", "distancia_horizontal": "Distância horizontal (m)"},
    )
    fig_distance.update_traces(line=dict(color=GREEN, width=4))
    st.plotly_chart(apply_dark_chart_style(fig_distance), use_container_width=True)

with st.expander("Ver tabela da simulação"):
    st.dataframe(df, use_container_width=True)
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("Baixar dados em CSV", data=csv, file_name="dados_queda_livre.csv", mime="text/csv")

with st.expander("Interpretação física"):
    st.markdown(
        """
        - A velocidade resultante é formada pelas componentes horizontal e vertical.
        - A componente horizontal `Vx` permanece constante quando não há resistência do ar.
        - A distância horizontal é `dₓ = |x - x₀|`.
        - Se `Vx = 0`, não há deslocamento horizontal; se `Vx ≠ 0`, `dₓ` cresce linearmente com o tempo.
        - A componente vertical `Vy` muda continuamente por causa da aceleração da gravidade.
        - A altura diminui de forma não linear, pois o movimento vertical é acelerado.
        """
    )
