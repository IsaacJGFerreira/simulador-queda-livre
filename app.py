"""Interface Streamlit do simulador didático de queda livre."""

from __future__ import annotations

import html
import json

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

from atmosphere import AtmosphereConfig
from objects import OBJECT_PRESETS, make_custom_object
from physics import SimulationConfig, simulate_fall
from wind import WindConfig


BG_MAIN = "#07111c"
BG_CARD_SECONDARY = "#0b1723"
TEXT_PRIMARY = "#f1f6fb"
TEXT_SECONDARY = "#b8c7d8"
GRID_COLOR = "rgba(120, 150, 180, 0.18)"
AXIS_COLOR = "rgba(180, 200, 220, 0.35)"

PURPLE = "#8b5cf6"
BLUE = "#2687ff"
CYAN = "#22d3ee"
GREEN = "#4ade80"
PINK = "#ff6f91"


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


def sample_simulation_frames(df: pd.DataFrame, max_frames: int = 180) -> pd.DataFrame:
    """Reduz a quantidade de pontos da simulação para animações leves."""
    if len(df) <= max_frames:
        return df.reset_index(drop=True)

    indices = np.linspace(0, len(df) - 1, max_frames, dtype=int)
    return df.iloc[np.unique(indices)].reset_index(drop=True)


def build_synced_panel_html(
    df: pd.DataFrame,
    object_name: str,
    initial_height: float,
    initial_vx: float,
    gravity: float,
) -> str:
    """Cria um único painel HTML onde animação e gráficos usam o mesmo controle de tempo."""
    anim_df = sample_simulation_frames(df, max_frames=180)
    records = anim_df[["t", "y", "x", "distancia_horizontal", "vx", "vy", "speed"]].round(4).to_dict(orient="records")

    max_height = max(float(initial_height), float(df["y"].max()), 1.0)
    min_x = float(df["x"].min())
    max_x = float(df["x"].max())
    vertical_fall = abs(initial_vx) < 1e-9
    escaped_name = html.escape(object_name)

    template = r'''
    <!DOCTYPE html>
    <html lang="pt-BR">
    <head>
      <meta charset="UTF-8" />
      <style>
        :root {
          --bg-main: #05080d;
          --bg-card: #0d1b2a;
          --bg-card-secondary: #102235;
          --border-soft: #1d3346;
          --border-strong: #2c5c8a;
          --text-primary: #f1f6fb;
          --text-secondary: #b8c7d8;
          --scene-bg: #eaf7ff;
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
          --cyan: #22d3ee;
          --green: #4ade80;
          --purple: #8b5cf6;
          --grid: rgba(120, 150, 180, 0.18);
        }
        * { box-sizing: border-box; }
        body { margin: 0; font-family: Inter, Arial, sans-serif; background: var(--bg-main); color: var(--text-primary); }

        .simulation-root {
          width: 100%;
          padding: 16px;
          background: var(--bg-main);
          border-radius: 16px;
        }
        .animation-section {
          display: grid;
          grid-template-columns: 2.4fr 1fr;
          gap: 16px;
          width: 100%;
        }
        .animation-card, .fall-panel, .charts-panel {
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
        .animation-title-area, .fall-panel-title, .section-title-row, .charts-title-row {
          display: flex;
          align-items: center;
          gap: 12px;
        }
        .animation-title-area h2, .fall-panel-title h2, .charts-title-row h2 {
          font-size: 20px;
          font-weight: 700;
          margin: 0;
          color: var(--text-primary);
        }
        .animation-icon, .panel-icon, .section-icon, .charts-icon {
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
        .section-icon { width: 30px; height: 30px; font-size: 17px; border-radius: 8px; }
        .animation-controls { display: flex; align-items: center; gap: 10px; flex-wrap: wrap; justify-content: flex-end; }
        .animation-controls button, .speed-control {
          background: linear-gradient(180deg, #13283c, #0d1b2a);
          border: 1px solid #2a4864;
          color: var(--text-primary);
          border-radius: 8px;
          padding: 10px 16px;
          font-size: 14px;
          cursor: pointer;
          min-height: 44px;
        }
        .animation-controls button:hover, .speed-control:hover { border-color: var(--vx-color); background: linear-gradient(180deg, #173454, #0d1b2a); }
        .speed-control { display: flex; flex-direction: column; align-items: center; gap: 1px; padding: 7px 18px; }
        .speed-control select { appearance: none; background: transparent; color: var(--text-primary); border: none; font-size: 16px; outline: none; text-align: center; cursor: pointer; }
        .speed-control span { font-size: 10px; color: var(--text-secondary); }
        .time-slider-row {
          display: grid;
          grid-template-columns: auto 1fr auto;
          gap: 14px;
          align-items: center;
          margin: 0 0 14px 0;
          padding: 10px 12px;
          background: rgba(16, 34, 53, 0.72);
          border: 1px solid #263f59;
          border-radius: 10px;
        }
        .time-badge { color: var(--text-primary); font-size: 14px; font-weight: 700; white-space: nowrap; }
        input[type="range"] { width: 100%; accent-color: var(--vx-color); }
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
        .trajectory { stroke: var(--trajectory-color); stroke-width: 4; fill: none; stroke-linecap: round; filter: drop-shadow(0 0 4px rgba(15, 76, 171, 0.25)); }
        .start-point { fill: var(--trajectory-color); stroke: #0a367a; stroke-width: 2; }
        .axis-line { stroke: var(--axis-color); stroke-width: 2; }
        .axis-label { fill: var(--axis-color); font-size: 15px; font-weight: 700; }
        .tick-label { fill: var(--axis-color); font-size: 14px; }
        .grid-line { stroke: rgba(70, 100, 130, 0.14); stroke-width: 1; stroke-dasharray: 4 4; }
        .ground { fill: var(--ground); }
        .ground-top-line { stroke: var(--ground-border); stroke-width: 3; }
        .ground-label { fill: var(--ground-text); font-size: 22px; font-weight: 800; letter-spacing: 1px; }
        .falling-object { fill: var(--ball-fill); stroke: var(--ball-border); stroke-width: 4; filter: drop-shadow(0 0 8px rgba(255, 138, 36, 0.45)); }
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
        .vector-legend-item { display: flex; align-items: center; gap: 8px; font-size: 14px; font-weight: 700; color: #0c2a44; }
        .legend-line { width: 34px; height: 3px; border-radius: 999px; display: inline-block; }
        .fall-panel { color: var(--text-primary); padding: 18px; }
        .fall-panel-title { margin-bottom: 20px; }
        .panel-section { border: 1px solid var(--border-strong); border-radius: 12px; padding: 14px; margin-bottom: 18px; background: rgba(7, 17, 28, 0.46); }
        .section-title-row { margin-bottom: 12px; color: var(--text-primary); font-size: 17px; font-weight: 700; }
        .section-title-row::after { content: ""; flex: 1; height: 1px; background: var(--border-strong); margin-left: 8px; }
        .data-row, .feature-row {
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
        .data-label, .feature-label { display: flex; align-items: center; gap: 10px; color: #e6eef7; font-size: 14px; }
        .data-icon, .feature-icon { width: 26px; text-align: center; font-size: 22px; }
        .data-row strong { font-size: 18px; font-weight: 800; color: var(--text-primary); white-space: nowrap; }
        .value-vx { color: var(--vx-color) !important; }
        .value-vy { color: var(--vy-color) !important; }
        .value-result { color: var(--v-result-color) !important; }
        .value-g { color: var(--gravity-color) !important; }
        .check-icon { width: 22px; height: 22px; border-radius: 50%; background: #4ade80; color: #052e16; display: flex; align-items: center; justify-content: center; font-weight: 900; }

        .charts-panel { margin-top: 16px; }
        .charts-title-row { margin-bottom: 8px; }
        .charts-subtitle { color: var(--text-secondary); font-size: 13px; margin: 0 0 14px 52px; line-height: 1.4; }
        .summary-bar {
          display: grid;
          grid-template-columns: repeat(4, 1fr);
          background: linear-gradient(180deg, #0f1d2b, #0b1723);
          border: 1px solid #1d3346;
          border-radius: 10px;
          padding: 14px 24px;
          margin-bottom: 16px;
        }
        .summary-item { display: flex; align-items: center; gap: 12px; min-width: 0; }
        .summary-item + .summary-item { border-left: 1px solid #1d3346; padding-left: 24px; }
        .summary-icon {
          width: 34px; height: 34px; border-radius: 999px;
          display: flex; align-items: center; justify-content: center;
          font-size: 18px; font-weight: 800;
        }
        .summary-title { font-size: 12px; color: #d7e2ee; margin-bottom: 3px; }
        .summary-value { font-size: 18px; font-weight: 600; }
        .charts-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 14px; }
        .chart-card {
          background: linear-gradient(180deg, #102031, #0c1926);
          border: 1px solid #21384d;
          border-radius: 10px;
          padding: 18px;
          min-height: 360px;
        }
        .chart-header { display: flex; align-items: center; gap: 10px; margin-bottom: 12px; }
        .chart-icon { width: 28px; height: 28px; border-radius: 7px; display: flex; align-items: center; justify-content: center; }
        .chart-icon.purple { background: rgba(139, 92, 246, 0.18); color: var(--purple); }
        .chart-icon.cyan { background: rgba(34, 211, 238, 0.16); color: var(--cyan); }
        .chart-icon.green { background: rgba(74, 222, 128, 0.16); color: var(--green); }
        .chart-title { font-size: 14px; font-weight: 700; color: var(--text-primary); }
        .chart-legend { display: flex; gap: 24px; margin: 12px 0 12px 0; font-size: 12px; color: #dce7f3; flex-wrap: wrap; }
        .chart-legend span { display: flex; align-items: center; gap: 6px; }
        .chart-legend i { width: 6px; height: 6px; border-radius: 1px; display: inline-block; }
        .chart-svg { width: 100%; height: 270px; overflow: visible; }
        .chart-grid { stroke: rgba(120, 150, 180, 0.18); stroke-width: 1; stroke-dasharray: 4 4; }
        .chart-axis { stroke: rgba(180, 200, 220, 0.35); stroke-width: 1; }
        .chart-label, .chart-tick { fill: #dbe7f3; font-size: 12px; }
        .chart-axis-label { fill: #edf4fb; font-size: 13px; font-weight: 600; }
        .chart-line { fill: none; stroke-width: 3; stroke-linecap: round; stroke-linejoin: round; }
        .value-bubble { fill: rgba(15, 29, 43, 0.92); stroke-width: 1; }
        .bubble-text { fill: #f1f6fb; font-size: 11px; font-weight: 700; }

        @media (max-width: 1000px) {
          .animation-section { grid-template-columns: 1fr; }
          .animation-header { align-items: flex-start; flex-direction: column; }
          .animation-controls { justify-content: flex-start; }
          .scene-container { height: 520px; }
          .charts-grid { grid-template-columns: 1fr; }
          .summary-bar { grid-template-columns: repeat(2, 1fr); gap: 12px; }
          .summary-item + .summary-item { border-left: none; padding-left: 0; }
        }
      </style>
    </head>
    <body>
      <div class="simulation-root">
        <div class="animation-section">
          <div class="animation-card">
            <div class="animation-header">
              <div class="animation-title-area"><div class="animation-icon">⌁</div><h2>Animação visual da queda livre com vetores</h2></div>
              <div class="animation-controls">
                <button id="playBtn">▶ Play</button>
                <button id="pauseBtn">⏸ Pausar</button>
                <button id="resetBtn">↻ Reiniciar</button>
                <div class="speed-control">
                  <select id="speedSelect"><option value="0.5">0.5x</option><option value="1" selected>1.0x</option><option value="1.5">1.5x</option><option value="2">2.0x</option></select>
                  <span>Velocidade</span>
                </div>
              </div>
            </div>
            <div class="time-slider-row">
              <span class="time-badge">Tempo: <span id="timeBadge">0,00 s</span></span>
              <input id="timeSlider" type="range" min="0" max="0" step="1" value="0" />
              <span class="time-badge">Quadro <span id="frameBadge">1</span></span>
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
                <line id="gVector" class="vector-g" x1="905" y1="105" x2="905" y2="175" />
                <text id="gLabel" class="label-g" x="890" y="195">g</text>
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

        <div class="charts-panel">
          <div class="charts-title-row"><div class="charts-icon">▤</div><h2>Gráficos sincronizados com a animação</h2></div>
          <p class="charts-subtitle">O mesmo botão Play, Pausar, Reiniciar e a mesma barra de tempo controlam a animação e os gráficos simultaneamente.</p>

          <div class="summary-bar">
            <div class="summary-item"><div class="summary-icon" style="color:#2687ff;background:rgba(38,135,255,.18)">↑</div><div><div class="summary-title">Altura inicial</div><div class="summary-value" style="color:#2687ff">__INITIAL_HEIGHT__ m</div></div></div>
            <div class="summary-item"><div class="summary-icon" style="color:#4ade80;background:rgba(74,222,128,.16)">◷</div><div><div class="summary-title">Tempo de simulação</div><div class="summary-value" style="color:#4ade80">__FINAL_TIME__ s</div></div></div>
            <div class="summary-item"><div class="summary-icon" style="color:#22d3ee;background:rgba(34,211,238,.16)">→</div><div><div class="summary-title">Velocidade horizontal (Vx)</div><div class="summary-value" style="color:#22d3ee">__INITIAL_VX__ m/s</div></div></div>
            <div class="summary-item"><div class="summary-icon" style="color:#ff6f91;background:rgba(255,111,145,.16)">↓</div><div><div class="summary-title">Queda livre vertical</div><div class="summary-value" style="color:#ff6f91">__VERTICAL_LABEL__</div></div></div>
          </div>

          <div class="charts-grid">
            <div class="chart-card">
              <div class="chart-header"><div class="chart-icon purple">↗</div><div class="chart-title">Componentes da velocidade × tempo</div></div>
              <div class="chart-legend"><span><i style="background:#8b5cf6"></i>|v|</span><span><i style="background:#2687ff"></i>Vx</span><span><i style="background:#ff6f91"></i>Vy</span></div>
              <svg id="velocityChart" class="chart-svg" viewBox="0 0 360 260"></svg>
            </div>
            <div class="chart-card">
              <div class="chart-header"><div class="chart-icon cyan">⌁</div><div class="chart-title">Altura × tempo</div></div>
              <div class="chart-legend"><span><i style="background:#22d3ee"></i>Altura</span></div>
              <svg id="heightChart" class="chart-svg" viewBox="0 0 360 260"></svg>
            </div>
            <div class="chart-card">
              <div class="chart-header"><div class="chart-icon green">→</div><div class="chart-title">Distância horizontal × tempo</div></div>
              <div class="chart-legend"><span><i style="background:#4ade80"></i>Distância horizontal</span></div>
              <svg id="distanceChart" class="chart-svg" viewBox="0 0 360 260"></svg>
            </div>
          </div>
        </div>
      </div>

      <script>
        const data = __DATA__;
        const h0 = __H0__;
        const minX = __MIN_X__;
        const maxX = __MAX_X__;
        const gravity = __GRAVITY__;
        const verticalFall = __VERTICAL_FALL__;
        const topY = 100;
        const groundY = 500;
        const leftX = 150;
        const rightX = 820;
        const visualWidth = rightX - leftX;
        const centerX = (leftX + rightX) / 2;
        const finalTime = data[data.length - 1].t;
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
        const timeSlider = document.getElementById('timeSlider');

        const formatNumber = (value, decimals = 1) => Number(value).toFixed(decimals).replace('.', ',');
        timeSlider.max = data.length - 1;

        function mapY(height) {
          const safeHeight = Math.max(0, Math.min(h0, height));
          return topY + ((h0 - safeHeight) / h0) * (groundY - topY);
        }
        function mapX(row) {
          const span = Math.abs(maxX - minX);
          if (span < 0.001) return centerX;
          const padding = Math.max(span * 0.18, 0.1);
          const low = minX - padding;
          const high = maxX + padding;
          const normalized = (row.x - low) / (high - low);
          return leftX + Math.max(0, Math.min(1, normalized)) * visualWidth;
        }
        function buildGrid() {
          gridLayer.innerHTML = '';
          const tickValues = [100, 80, 60, 40, 20, 0];
          tickValues.forEach((value) => {
            const normalizedValue = value > h0 ? h0 : value;
            const y = mapY(normalizedValue);
            const line = document.createElementNS('http://www.w3.org/2000/svg', 'line');
            line.setAttribute('class', 'grid-line'); line.setAttribute('x1', '70'); line.setAttribute('x2', '980'); line.setAttribute('y1', y); line.setAttribute('y2', y); gridLayer.appendChild(line);
            const tick = document.createElementNS('http://www.w3.org/2000/svg', 'line');
            tick.setAttribute('class', 'axis-line'); tick.setAttribute('x1', '62'); tick.setAttribute('x2', '78'); tick.setAttribute('y1', y); tick.setAttribute('y2', y); gridLayer.appendChild(tick);
            const label = document.createElementNS('http://www.w3.org/2000/svg', 'text');
            label.setAttribute('class', 'tick-label'); label.setAttribute('x', '48'); label.setAttribute('y', y + 5); label.setAttribute('text-anchor', 'end'); label.textContent = String(value); gridLayer.appendChild(label);
          });
        }
        function pointFor(row) { return { x: mapX(row), y: mapY(row.y) }; }
        function buildPath(index) {
          const points = data.slice(0, index + 1).map(pointFor);
          if (points.length === 0) return '';
          let path = `M ${points[0].x} ${points[0].y}`;
          for (let i = 1; i < points.length; i++) path += ` L ${points[i].x} ${points[i].y}`;
          return path;
        }
        function setLine(line, x1, y1, x2, y2) { line.setAttribute('x1', x1); line.setAttribute('y1', y1); line.setAttribute('x2', x2); line.setAttribute('y2', y2); }
        function updatePanel(row) {
          document.getElementById('timeValue').textContent = `${formatNumber(row.t, 2)} s`;
          document.getElementById('heightValue').textContent = `${formatNumber(Math.max(0, row.y), 0)} m`;
          document.getElementById('distanceValue').textContent = `${formatNumber(row.distancia_horizontal, 0)} m`;
          document.getElementById('vxValue').textContent = `${formatNumber(row.vx, 1)} m/s`;
          document.getElementById('vyValue').textContent = `${formatNumber(row.vy, 1)} m/s`;
          document.getElementById('speedValue').textContent = `${formatNumber(row.speed, 1)} m/s`;
          document.getElementById('gValue').textContent = `${formatNumber(gravity, 1)} m/s²`;
          document.getElementById('horizontalFeature').textContent = verticalFall ? 'Sem componente horizontal' : 'Com componente horizontal';
          document.getElementById('timeBadge').textContent = `${formatNumber(row.t, 2)} s`;
          document.getElementById('frameBadge').textContent = `${frameIndex + 1}/${data.length}`;
        }
        function setGravityVectorStatic() {
          setLine(gVector, 905, 105, 905, 175);
          gLabel.setAttribute('x', 890); gLabel.setAttribute('y', 195);
        }

        function svgEl(name) { return document.createElementNS('http://www.w3.org/2000/svg', name); }
        function drawChartBase(svg, yMin, yMax, yTicks, yLabel) {
          svg.innerHTML = '';
          const W = 360, H = 260, L = 46, R = 12, T = 14, B = 38;
          const PW = W - L - R, PH = H - T - B;
          function xScale(t) { return L + (t / Math.max(finalTime, 0.001)) * PW; }
          function yScale(v) { return T + ((yMax - v) / (yMax - yMin)) * PH; }
          for (let i = 0; i <= 5; i++) {
            const t = finalTime * i / 5;
            const x = xScale(t);
            const line = svgEl('line'); line.setAttribute('class', 'chart-grid'); line.setAttribute('x1', x); line.setAttribute('x2', x); line.setAttribute('y1', T); line.setAttribute('y2', T + PH); svg.appendChild(line);
            const txt = svgEl('text'); txt.setAttribute('class', 'chart-tick'); txt.setAttribute('x', x); txt.setAttribute('y', H - 14); txt.setAttribute('text-anchor', 'middle'); txt.textContent = formatNumber(t, 0); svg.appendChild(txt);
          }
          yTicks.forEach((v) => {
            const y = yScale(v);
            const line = svgEl('line'); line.setAttribute('class', 'chart-grid'); line.setAttribute('x1', L); line.setAttribute('x2', L + PW); line.setAttribute('y1', y); line.setAttribute('y2', y); svg.appendChild(line);
            const txt = svgEl('text'); txt.setAttribute('class', 'chart-tick'); txt.setAttribute('x', L - 8); txt.setAttribute('y', y + 4); txt.setAttribute('text-anchor', 'end'); txt.textContent = formatNumber(v, Math.abs(v) < 2 && yMax <= 2 ? 1 : 0); svg.appendChild(txt);
          });
          const axisX = svgEl('line'); axisX.setAttribute('class', 'chart-axis'); axisX.setAttribute('x1', L); axisX.setAttribute('x2', L + PW); axisX.setAttribute('y1', T + PH); axisX.setAttribute('y2', T + PH); svg.appendChild(axisX);
          const axisY = svgEl('line'); axisY.setAttribute('class', 'chart-axis'); axisY.setAttribute('x1', L); axisY.setAttribute('x2', L); axisY.setAttribute('y1', T); axisY.setAttribute('y2', T + PH); svg.appendChild(axisY);
          const labelX = svgEl('text'); labelX.setAttribute('class', 'chart-axis-label'); labelX.setAttribute('x', L + PW / 2); labelX.setAttribute('y', H - 2); labelX.setAttribute('text-anchor', 'middle'); labelX.textContent = 'Tempo (s)'; svg.appendChild(labelX);
          const labelY = svgEl('text'); labelY.setAttribute('class', 'chart-axis-label'); labelY.setAttribute('x', 12); labelY.setAttribute('y', T + PH / 2); labelY.setAttribute('text-anchor', 'middle'); labelY.setAttribute('transform', `rotate(-90 12 ${T + PH / 2})`); labelY.textContent = yLabel; svg.appendChild(labelY);
          return { xScale, yScale, L, R, T, B, PW, PH };
        }
        function pathFrom(values, xScale, yScale, key) {
          if (values.length === 0) return '';
          let d = `M ${xScale(values[0].t)} ${yScale(values[0][key])}`;
          for (let i = 1; i < values.length; i++) d += ` L ${xScale(values[i].t)} ${yScale(values[i][key])}`;
          return d;
        }
        function addLine(svg, d, color) {
          const path = svgEl('path'); path.setAttribute('class', 'chart-line'); path.setAttribute('d', d); path.setAttribute('stroke', color); svg.appendChild(path); return path;
        }
        function addPoint(svg, x, y, color) {
          const c = svgEl('circle'); c.setAttribute('cx', x); c.setAttribute('cy', y); c.setAttribute('r', 4.5); c.setAttribute('fill', color); c.setAttribute('stroke', '#f1f6fb'); c.setAttribute('stroke-width', '1'); svg.appendChild(c);
        }
        function addBubble(svg, x, y, value, color) {
          const text = formatNumber(value, 1);
          const width = Math.max(34, text.length * 7 + 12);
          const rect = svgEl('rect'); rect.setAttribute('class', 'value-bubble'); rect.setAttribute('x', Math.min(x + 8, 348 - width)); rect.setAttribute('y', y - 10); rect.setAttribute('width', width); rect.setAttribute('height', 20); rect.setAttribute('rx', 4); rect.setAttribute('stroke', color); svg.appendChild(rect);
          const txt = svgEl('text'); txt.setAttribute('class', 'bubble-text'); txt.setAttribute('x', Math.min(x + 8, 348 - width) + width / 2); txt.setAttribute('y', y + 4); txt.setAttribute('text-anchor', 'middle'); txt.textContent = text; svg.appendChild(txt);
        }
        function updateCharts(index) {
          const currentData = data.slice(0, index + 1);
          const row = data[index];
          const velocitySvg = document.getElementById('velocityChart');
          const heightSvg = document.getElementById('heightChart');
          const distanceSvg = document.getElementById('distanceChart');
          const vBase = drawChartBase(velocitySvg, -60, 60, [-60, -40, -20, 0, 20, 40, 60], 'Velocidade (m/s)');
          addLine(velocitySvg, pathFrom(currentData, vBase.xScale, vBase.yScale, 'speed'), '#8b5cf6');
          addLine(velocitySvg, pathFrom(currentData, vBase.xScale, vBase.yScale, 'vx'), '#2687ff');
          addLine(velocitySvg, pathFrom(currentData, vBase.xScale, vBase.yScale, 'vy'), '#ff6f91');
          [['speed','#8b5cf6'], ['vx','#2687ff'], ['vy','#ff6f91']].forEach(([key, color]) => { const x = vBase.xScale(row.t), y = vBase.yScale(row[key]); addPoint(velocitySvg, x, y, color); addBubble(velocitySvg, x, y, row[key], color); });

          const hMax = Math.max(120, h0 * 1.15);
          const hBase = drawChartBase(heightSvg, 0, hMax, [0, 20, 40, 60, 80, 100, 120].filter(v => v <= hMax), 'Altura (m)');
          addLine(heightSvg, pathFrom(currentData, hBase.xScale, hBase.yScale, 'y'), '#22d3ee');
          addPoint(heightSvg, hBase.xScale(row.t), hBase.yScale(row.y), '#22d3ee');

          const maxD = Math.max(...data.map(d => d.distancia_horizontal));
          const dMin = maxD < 0.001 ? -1.5 : 0;
          const dMax = maxD < 0.001 ? 1.5 : Math.max(1.5, maxD * 1.15);
          const dTicks = maxD < 0.001 ? [-1.5, -1, -0.5, 0, 0.5, 1, 1.5] : [0, dMax / 4, dMax / 2, 3 * dMax / 4, dMax];
          const dBase = drawChartBase(distanceSvg, dMin, dMax, dTicks, 'Distância horizontal (m)');
          addLine(distanceSvg, pathFrom(currentData, dBase.xScale, dBase.yScale, 'distancia_horizontal'), '#4ade80');
          addPoint(distanceSvg, dBase.xScale(row.t), dBase.yScale(row.distancia_horizontal), '#4ade80');
        }
        function drawFrame(index) {
          frameIndex = Math.max(0, Math.min(data.length - 1, index));
          const row = data[frameIndex];
          const p = pointFor(row);
          trajectoryPath.setAttribute('d', buildPath(frameIndex));
          const initialPoint = pointFor(data[0]);
          startPoint.setAttribute('cx', initialPoint.x); startPoint.setAttribute('cy', initialPoint.y);
          fallingObject.setAttribute('cx', p.x); fallingObject.setAttribute('cy', p.y);
          const hasHorizontal = Math.abs(row.vx) > 0.001;
          const vxLen = hasHorizontal ? Math.max(28, Math.min(90, Math.abs(row.vx) * 4)) : 0;
          const vyLen = Math.max(45, Math.min(125, Math.abs(row.vy) * 2.2));
          const signX = row.vx < 0 ? -1 : 1;
          if (hasHorizontal) {
            vxVector.style.display = 'block'; vxLabel.style.display = 'block';
            setLine(vxVector, p.x, p.y, p.x + signX * vxLen, p.y);
            vxLabel.setAttribute('x', p.x + signX * vxLen + 8); vxLabel.setAttribute('y', p.y - 10);
          } else { vxVector.style.display = 'none'; vxLabel.style.display = 'none'; }
          setLine(vyVector, p.x, p.y + 12, p.x, p.y + vyLen);
          vyLabel.setAttribute('x', p.x + 12); vyLabel.setAttribute('y', p.y + vyLen + 8);
          if (hasHorizontal) {
            const resultXLen = Math.max(40, Math.min(110, Math.abs(row.vx) * 4));
            const resultYLen = Math.max(55, Math.min(125, Math.abs(row.vy) * 2.2));
            setLine(resultVector, p.x + 8, p.y + 8, p.x + signX * resultXLen, p.y + resultYLen);
            resultLabel.setAttribute('x', p.x + signX * resultXLen + 8); resultLabel.setAttribute('y', p.y + resultYLen + 8);
          } else {
            const resultYLen = Math.max(55, Math.min(125, Math.abs(row.vy) * 2.2));
            setLine(resultVector, p.x + 22, p.y + 8, p.x + 22, p.y + resultYLen);
            resultLabel.setAttribute('x', p.x + 34); resultLabel.setAttribute('y', p.y + resultYLen + 8);
          }
          setGravityVectorStatic();
          timeSlider.value = frameIndex;
          updatePanel(row);
          updateCharts(frameIndex);
        }
        function play() {
          if (timer) return;
          timer = setInterval(() => {
            if (frameIndex >= data.length - 1) { pause(); return; }
            drawFrame(frameIndex + 1);
          }, Math.max(16, 45 / speedFactor));
        }
        function pause() { if (timer) { clearInterval(timer); timer = null; } }
        function reset() { pause(); drawFrame(0); }
        document.getElementById('playBtn').addEventListener('click', play);
        document.getElementById('pauseBtn').addEventListener('click', pause);
        document.getElementById('resetBtn').addEventListener('click', reset);
        document.getElementById('speedSelect').addEventListener('change', (event) => { speedFactor = Number(event.target.value); if (timer) { pause(); play(); } });
        timeSlider.addEventListener('input', (event) => { pause(); drawFrame(Number(event.target.value)); });
        buildGrid();
        drawFrame(0);
      </script>
    </body>
    </html>
    '''

    return (
        template
        .replace("__DATA__", json.dumps(records, ensure_ascii=False))
        .replace("__H0__", json.dumps(max_height))
        .replace("__MIN_X__", json.dumps(min_x))
        .replace("__MAX_X__", json.dumps(max_x))
        .replace("__GRAVITY__", json.dumps(float(gravity)))
        .replace("__VERTICAL_FALL__", "true" if vertical_fall else "false")
        .replace("__OBJECT_NAME__", escaped_name)
        .replace("__INITIAL_HEIGHT__", format_br(float(initial_height), 1))
        .replace("__FINAL_TIME__", format_br(float(df.iloc[-1]["t"]), 2))
        .replace("__INITIAL_VX__", format_br(float(initial_vx), 2))
        .replace("__VERTICAL_LABEL__", "Sim" if vertical_fall else "Não")
    )


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
            Visualize o corpo em queda, os vetores físicos e os gráficos de velocidade, altura e distância horizontal
            caminhando juntos no mesmo controle de tempo.
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
        <div class="section-title">Painel único: animação e gráficos sincronizados</div>
        <div class="section-subtitle">
            Agora o Play, Pausar, Reiniciar e a barra de tempo controlam a animação e os três gráficos ao mesmo tempo.
            Ao voltar na barra de tempo, a animação e os gráficos voltam juntos.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

synced_html = build_synced_panel_html(
    df=df,
    object_name=falling_object.name,
    initial_height=initial_height,
    initial_vx=initial_vx,
    gravity=gravity,
)
components.html(synced_html, height=1260, scrolling=False)

st.markdown(
    """
    <div class="tip-box">
        <b>Observação:</b> se <b>Vx = 0</b>, o corpo cai em linha vertical e a distância horizontal permanece em zero.
        Se <b>Vx ≠ 0</b>, o corpo se desloca lateralmente e o gráfico da distância horizontal cresce linearmente.
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
        - A animação e os gráficos usam o mesmo índice de tempo.
        - Se `Vx = 0`, o corpo não se desloca para os lados e a trajetória fica vertical.
        - Se `Vx > 0`, o corpo se desloca para a direita; se `Vx < 0`, para a esquerda.
        - O vetor `g` representa a aceleração da gravidade e fica fixo no painel.
        - A componente horizontal `Vx` permanece constante quando não há resistência do ar.
        - A componente vertical `Vy` muda continuamente por causa da aceleração da gravidade.
        """
    )
