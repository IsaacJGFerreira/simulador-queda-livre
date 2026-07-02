"""Interface Streamlit do simulador didático de queda livre."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
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

VECTOR_RESULTANT_COLOR = "#7E22CE"
VECTOR_X_COLOR = "#2563EB"
VECTOR_Y_COLOR = "#DC2626"
HEIGHT_COLOR = "#0891B2"
OBJECT_COLOR = "#FF7043"


st.set_page_config(
    page_title="Simulador de Queda Livre",
    page_icon="🪂",
    layout="wide",
)


def sample_simulation_frames(df: pd.DataFrame, max_frames: int = 120) -> pd.DataFrame:
    """Reduz a quantidade de pontos da simulação para criar animações leves."""
    if len(df) <= max_frames:
        return df.reset_index(drop=True)

    indices = np.linspace(0, len(df) - 1, max_frames, dtype=int)
    indices = np.unique(indices)
    return df.iloc[indices].reset_index(drop=True)


def format_panel_text(row: pd.Series, object_name: str, gravity: float) -> str:
    """Texto do painel lateral da animação didática."""
    return (
        f"<b>{object_name}</b><br><br>"
        f"⏱ <b>Tempo:</b> {row['t']:.2f} s<br>"
        f"↕ <b>Altura:</b> {row['y']:.2f} m<br>"
        f"⚡ <b>|v|:</b> {row['speed']:.2f} m/s<br>"
        f"<span style='color:{VECTOR_X_COLOR}'><b>vₓ:</b></span> {row['vx']:.2f} m/s<br>"
        f"<span style='color:{VECTOR_Y_COLOR}'><b>vᵧ:</b></span> {row['vy']:.2f} m/s<br>"
        f"➡ <b>Distância horizontal:</b> {row['x']:.2f} m<br>"
        f"⬇ <b>Aceleração:</b> {gravity:.2f} m/s²<br>"
        f"<br><b>Leitura dos vetores:</b><br>"
        f"<span style='color:{VECTOR_RESULTANT_COLOR}'><b>roxo:</b></span> velocidade resultante<br>"
        f"<span style='color:{VECTOR_X_COLOR}'><b>azul:</b></span> componente horizontal<br>"
        f"<span style='color:{VECTOR_Y_COLOR}'><b>vermelho:</b></span> componente vertical<br>"
        f"<br><b>Gráficos:</b> são desenhados junto com a queda."
    )


def marker_symbol_for_horizontal(value: float) -> str:
    """Escolhe o símbolo da ponta do vetor horizontal."""
    if value > 0:
        return "triangle-right"
    if value < 0:
        return "triangle-left"
    return "circle"


def marker_symbol_for_vertical(value: float) -> str:
    """Escolhe o símbolo da ponta do vetor vertical."""
    if value > 0:
        return "triangle-up"
    if value < 0:
        return "triangle-down"
    return "circle"


def vector_geometry(
    row: pd.Series,
    vector_scale: float,
    scene_width: float,
    scene_y_max: float,
    max_speed: float,
    max_abs_vx: float,
    max_abs_vy: float,
) -> dict[str, object]:
    """Calcula posição, sentido, tamanho e largura dos vetores em um quadro."""
    x = float(row["x"])
    y = float(row["y"])
    vx = float(row["vx"])
    vy = float(row["vy"])
    speed = float(row["speed"])

    v_end = (x + vx * vector_scale, y + vy * vector_scale)
    vx_end = (x + vx * vector_scale, y)
    vy_end = (x, y + vy * vector_scale)

    resultant_width = 2 + 4 * (speed / max_speed if max_speed > 0 else 0)
    vx_width = 2 + 3 * (abs(vx) / max_abs_vx if max_abs_vx > 0 else 0)
    vy_width = 2 + 3 * (abs(vy) / max_abs_vy if max_abs_vy > 0 else 0)

    return {
        "origin": (x, y),
        "v_end": v_end,
        "vx_end": vx_end,
        "vy_end": vy_end,
        "resultant_width": resultant_width,
        "vx_width": vx_width,
        "vy_width": vy_width,
        "vx_symbol": marker_symbol_for_horizontal(vx),
        "vy_symbol": marker_symbol_for_vertical(vy),
    }


def build_vector_traces(row: pd.Series, vector_data: dict[str, object]) -> list[go.Scatter]:
    """Cria os traços dos vetores para a animação."""
    x, y = vector_data["origin"]
    vx_end = vector_data["vx_end"]
    vy_end = vector_data["vy_end"]
    v_end = vector_data["v_end"]

    return [
        go.Scatter(
            x=[x, v_end[0]],
            y=[y, v_end[1]],
            mode="lines",
            line=dict(width=vector_data["resultant_width"], color=VECTOR_RESULTANT_COLOR),
            hoverinfo="skip",
        ),
        go.Scatter(
            x=[v_end[0]],
            y=[v_end[1]],
            mode="markers+text",
            text=["v"],
            textposition="top center",
            textfont=dict(size=15, color=VECTOR_RESULTANT_COLOR),
            marker=dict(size=14, color=VECTOR_RESULTANT_COLOR, symbol="diamond"),
            hovertemplate=f"|v| = {row['speed']:.2f} m/s<extra></extra>",
        ),
        go.Scatter(
            x=[x, vx_end[0]],
            y=[y, vx_end[1]],
            mode="lines",
            line=dict(width=vector_data["vx_width"], color=VECTOR_X_COLOR, dash="dot"),
            hoverinfo="skip",
        ),
        go.Scatter(
            x=[vx_end[0]],
            y=[vx_end[1]],
            mode="markers+text",
            text=["vₓ"],
            textposition="top center",
            textfont=dict(size=15, color=VECTOR_X_COLOR),
            marker=dict(
                size=14,
                color=VECTOR_X_COLOR,
                symbol=vector_data["vx_symbol"],
            ),
            hovertemplate=f"vₓ = {row['vx']:.2f} m/s<extra></extra>",
        ),
        go.Scatter(
            x=[x, vy_end[0]],
            y=[y, vy_end[1]],
            mode="lines",
            line=dict(width=vector_data["vy_width"], color=VECTOR_Y_COLOR, dash="dot"),
            hoverinfo="skip",
        ),
        go.Scatter(
            x=[vy_end[0]],
            y=[vy_end[1]],
            mode="markers+text",
            text=["vᵧ"],
            textposition="middle right",
            textfont=dict(size=15, color=VECTOR_Y_COLOR),
            marker=dict(
                size=14,
                color=VECTOR_Y_COLOR,
                symbol=vector_data["vy_symbol"],
            ),
            hovertemplate=f"vᵧ = {row['vy']:.2f} m/s<extra></extra>",
        ),
    ]


def build_synchronized_animation_figure(
    full_df: pd.DataFrame,
    object_name: str,
    gravity: float,
) -> go.Figure:
    """Cria uma animação única com cena, vetores e gráficos sincronizados."""
    anim_df = sample_simulation_frames(full_df, max_frames=120)

    scene_y_max = max(float(anim_df["y"].max()), 1.0)
    scene_y_min = min(float(anim_df["y"].min()), 0.0)
    ground_bottom = -0.10 * scene_y_max

    scene_x_min = float(anim_df["x"].min())
    scene_x_max = float(anim_df["x"].max())

    if abs(scene_x_max - scene_x_min) < 1.0:
        center_x = (scene_x_max + scene_x_min) / 2
        scene_x_min = center_x - 2.0
        scene_x_max = center_x + 2.0
    else:
        margin = 0.15 * (scene_x_max - scene_x_min)
        scene_x_min -= margin
        scene_x_max += margin

    scene_width = scene_x_max - scene_x_min
    max_speed = max(float(anim_df["speed"].max()), 1e-9)
    max_abs_vx = max(float(anim_df["vx"].abs().max()), 1e-9)
    max_abs_vy = max(float(anim_df["vy"].abs().max()), 1e-9)
    vector_reference_length = max(0.12 * scene_y_max, min(0.24 * scene_y_max, 0.20 * scene_width))
    vector_scale = vector_reference_length / max_speed

    scene_x_min -= 1.15 * vector_reference_length
    scene_x_max += 1.15 * vector_reference_length
    scene_y_min = min(ground_bottom - 0.60 * vector_reference_length, scene_y_min)
    scene_y_top = scene_y_max * 1.12 + 0.25 * vector_reference_length

    time_max = max(float(anim_df["t"].max()), 1e-9)
    height_max = max(float(anim_df["y"].max()), 1.0)
    velocity_min = min(float(anim_df["vy"].min()), float(anim_df["vx"].min()), 0.0)
    velocity_max = max(float(anim_df["speed"].max()), float(anim_df["vx"].max()), float(anim_df["vy"].max()), 1.0)
    velocity_margin = max(1.0, 0.10 * (velocity_max - velocity_min))

    initial = anim_df.iloc[0]
    initial_vector_data = vector_geometry(
        row=initial,
        vector_scale=vector_scale,
        scene_width=scene_x_max - scene_x_min,
        scene_y_max=scene_y_max,
        max_speed=max_speed,
        max_abs_vx=max_abs_vx,
        max_abs_vy=max_abs_vy,
    )

    fig = make_subplots(
        rows=2,
        cols=2,
        row_heights=[0.62, 0.38],
        column_widths=[0.66, 0.34],
        horizontal_spacing=0.07,
        vertical_spacing=0.15,
        subplot_titles=(
            "Cena da queda com vetores",
            "Painel da queda",
            "Altura × tempo",
            "Componentes da velocidade × tempo",
        ),
    )

    # 0 - Rastro da trajetória
    fig.add_trace(
        go.Scatter(
            x=[initial["x"]],
            y=[initial["y"]],
            mode="lines",
            line=dict(width=4, color="#0D47A1"),
            name="Rastro",
            hoverinfo="skip",
        ),
        row=1,
        col=1,
    )

    # 1 - Objeto
    fig.add_trace(
        go.Scatter(
            x=[initial["x"]],
            y=[initial["y"]],
            mode="markers",
            name="Objeto",
            marker=dict(size=34, color=OBJECT_COLOR, line=dict(width=3, color="#BF360C")),
            hovertemplate=(
                "Tempo: %{customdata[0]:.2f} s<br>"
                "Altura: %{y:.2f} m<br>"
                "Velocidade: %{customdata[1]:.2f} m/s<extra></extra>"
            ),
            customdata=[[initial["t"], initial["speed"]]],
        ),
        row=1,
        col=1,
    )

    # 2 a 7 - Vetores da velocidade
    for trace in build_vector_traces(initial, initial_vector_data):
        fig.add_trace(trace, row=1, col=1)

    # 8 - Painel lateral
    fig.add_trace(
        go.Scatter(
            x=[0.05],
            y=[0.52],
            mode="text",
            name="Painel",
            text=[format_panel_text(initial, object_name, gravity)],
            textposition="middle left",
            textfont=dict(size=15, color="#1F2937"),
            hoverinfo="skip",
        ),
        row=1,
        col=2,
    )

    # 9 - Gráfico altura × tempo sendo construído
    fig.add_trace(
        go.Scatter(
            x=[initial["t"]],
            y=[initial["y"]],
            mode="lines",
            line=dict(width=4, color=HEIGHT_COLOR),
            name="Altura",
            hovertemplate="t = %{x:.2f} s<br>y = %{y:.2f} m<extra></extra>",
        ),
        row=2,
        col=1,
    )

    # 10 - Marcador atual no gráfico altura × tempo
    fig.add_trace(
        go.Scatter(
            x=[initial["t"]],
            y=[initial["y"]],
            mode="markers",
            marker=dict(size=12, color=OBJECT_COLOR, line=dict(width=2, color="#BF360C")),
            name="Altura atual",
            hoverinfo="skip",
        ),
        row=2,
        col=1,
    )

    # 11, 12, 13 - Componentes da velocidade sendo construídas
    fig.add_trace(
        go.Scatter(
            x=[initial["t"]],
            y=[initial["speed"]],
            mode="lines",
            line=dict(width=4, color=VECTOR_RESULTANT_COLOR),
            name="|v|",
            hovertemplate="t = %{x:.2f} s<br>|v| = %{y:.2f} m/s<extra></extra>",
        ),
        row=2,
        col=2,
    )
    fig.add_trace(
        go.Scatter(
            x=[initial["t"]],
            y=[initial["vx"]],
            mode="lines",
            line=dict(width=3, color=VECTOR_X_COLOR),
            name="vₓ",
            hovertemplate="t = %{x:.2f} s<br>vₓ = %{y:.2f} m/s<extra></extra>",
        ),
        row=2,
        col=2,
    )
    fig.add_trace(
        go.Scatter(
            x=[initial["t"]],
            y=[initial["vy"]],
            mode="lines",
            line=dict(width=3, color=VECTOR_Y_COLOR),
            name="vᵧ",
            hovertemplate="t = %{x:.2f} s<br>vᵧ = %{y:.2f} m/s<extra></extra>",
        ),
        row=2,
        col=2,
    )

    # 14 - Marcadores atuais das velocidades
    fig.add_trace(
        go.Scatter(
            x=[initial["t"], initial["t"], initial["t"]],
            y=[initial["speed"], initial["vx"], initial["vy"]],
            mode="markers",
            marker=dict(
                size=10,
                color=[VECTOR_RESULTANT_COLOR, VECTOR_X_COLOR, VECTOR_Y_COLOR],
                line=dict(width=1, color="#111827"),
            ),
            name="Velocidade atual",
            hoverinfo="skip",
        ),
        row=2,
        col=2,
    )

    frames: list[go.Frame] = []
    slider_steps = []

    for index, row in anim_df.iterrows():
        trail_df = anim_df.iloc[: index + 1]
        frame_name = str(index)
        vector_data = vector_geometry(
            row=row,
            vector_scale=vector_scale,
            scene_width=scene_x_max - scene_x_min,
            scene_y_max=scene_y_max,
            max_speed=max_speed,
            max_abs_vx=max_abs_vx,
            max_abs_vy=max_abs_vy,
        )
        chart_df = anim_df.iloc[: index + 1]

        frames.append(
            go.Frame(
                name=frame_name,
                data=[
                    go.Scatter(x=trail_df["x"], y=trail_df["y"], mode="lines", line=dict(width=4, color="#0D47A1"), hoverinfo="skip"),
                    go.Scatter(
                        x=[row["x"]],
                        y=[row["y"]],
                        mode="markers",
                        marker=dict(size=34, color=OBJECT_COLOR, line=dict(width=3, color="#BF360C")),
                        hovertemplate=(
                            "Tempo: %{customdata[0]:.2f} s<br>"
                            "Altura: %{y:.2f} m<br>"
                            "Velocidade: %{customdata[1]:.2f} m/s<extra></extra>"
                        ),
                        customdata=[[row["t"], row["speed"]]],
                    ),
                    *build_vector_traces(row, vector_data),
                    go.Scatter(
                        x=[0.05],
                        y=[0.52],
                        mode="text",
                        text=[format_panel_text(row, object_name, gravity)],
                        textposition="middle left",
                        textfont=dict(size=15, color="#1F2937"),
                        hoverinfo="skip",
                    ),
                    go.Scatter(x=chart_df["t"], y=chart_df["y"], mode="lines", line=dict(width=4, color=HEIGHT_COLOR)),
                    go.Scatter(x=[row["t"]], y=[row["y"]], mode="markers", marker=dict(size=12, color=OBJECT_COLOR, line=dict(width=2, color="#BF360C"))),
                    go.Scatter(x=chart_df["t"], y=chart_df["speed"], mode="lines", line=dict(width=4, color=VECTOR_RESULTANT_COLOR)),
                    go.Scatter(x=chart_df["t"], y=chart_df["vx"], mode="lines", line=dict(width=3, color=VECTOR_X_COLOR)),
                    go.Scatter(x=chart_df["t"], y=chart_df["vy"], mode="lines", line=dict(width=3, color=VECTOR_Y_COLOR)),
                    go.Scatter(
                        x=[row["t"], row["t"], row["t"]],
                        y=[row["speed"], row["vx"], row["vy"]],
                        mode="markers",
                        marker=dict(
                            size=10,
                            color=[VECTOR_RESULTANT_COLOR, VECTOR_X_COLOR, VECTOR_Y_COLOR],
                            line=dict(width=1, color="#111827"),
                        ),
                    ),
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
                        {
                            "mode": "immediate",
                            "frame": {"duration": 0, "redraw": True},
                            "transition": {"duration": 0},
                        },
                    ],
                }
            )

    fig.frames = frames

    fig.add_shape(
        type="rect",
        x0=scene_x_min,
        x1=scene_x_max,
        y0=0,
        y1=scene_y_top,
        fillcolor="#EAF7FF",
        line=dict(width=0),
        layer="below",
        row=1,
        col=1,
    )
    fig.add_shape(
        type="rect",
        x0=scene_x_min,
        x1=scene_x_max,
        y0=scene_y_min,
        y1=0,
        fillcolor="#7CB342",
        line=dict(width=0),
        layer="below",
        row=1,
        col=1,
    )
    fig.add_shape(
        type="line",
        x0=scene_x_min,
        x1=scene_x_max,
        y0=0,
        y1=0,
        line=dict(width=5, color="#33691E"),
        layer="above",
        row=1,
        col=1,
    )
    fig.add_annotation(
        x=(scene_x_min + scene_x_max) / 2,
        y=scene_y_min * 0.55,
        text="<b>SOLO</b>",
        showarrow=False,
        font=dict(size=15, color="#1B5E20"),
        row=1,
        col=1,
    )

    fig.update_layout(
        height=840,
        margin=dict(l=20, r=20, t=70, b=50),
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1.0),
        dragmode=False,
        plot_bgcolor="white",
        paper_bgcolor="white",
        updatemenus=[
            {
                "type": "buttons",
                "showactive": False,
                "direction": "left",
                "x": 0.02,
                "y": -0.08,
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
                            {
                                "frame": {"duration": ANIMATION_FRAME_DURATION_MS, "redraw": True},
                                "fromcurrent": True,
                                "transition": {"duration": 0},
                                "mode": "immediate",
                            },
                        ],
                    },
                    {
                        "label": "⏸ Pausar",
                        "method": "animate",
                        "args": [
                            [None],
                            {
                                "frame": {"duration": 0, "redraw": False},
                                "transition": {"duration": 0},
                                "mode": "immediate",
                            },
                        ],
                    },
                ],
            }
        ],
        sliders=[
            {
                "active": 0,
                "x": 0.18,
                "y": -0.08,
                "xanchor": "left",
                "yanchor": "top",
                "len": 0.76,
                "pad": {"t": 0, "b": 0},
                "bgcolor": "#E5E7EB",
                "activebgcolor": "#1D4ED8",
                "bordercolor": "#CBD5E1",
                "currentvalue": {"prefix": "Tempo: ", "suffix": "", "font": {"size": 14, "color": "#111827"}},
                "steps": slider_steps,
            }
        ],
    )

    fig.update_xaxes(range=[scene_x_min, scene_x_max], showgrid=False, zeroline=False, title="", showticklabels=False, fixedrange=True, row=1, col=1)
    fig.update_yaxes(range=[scene_y_min, scene_y_top], showgrid=False, zeroline=False, title="Altura (m)", fixedrange=True, row=1, col=1)
    fig.update_xaxes(range=[0, 1], showgrid=False, zeroline=False, title="", showticklabels=False, fixedrange=True, row=1, col=2)
    fig.update_yaxes(range=[0, 1], showgrid=False, zeroline=False, title="", showticklabels=False, fixedrange=True, row=1, col=2)
    fig.update_xaxes(range=[0, time_max * 1.02], title="Tempo (s)", fixedrange=True, row=2, col=1)
    fig.update_yaxes(range=[0, height_max * 1.08], title="Altura (m)", fixedrange=True, row=2, col=1)
    fig.update_xaxes(range=[0, time_max * 1.02], title="Tempo (s)", fixedrange=True, row=2, col=2)
    fig.update_yaxes(range=[velocity_min - velocity_margin, velocity_max + velocity_margin], title="Velocidade (m/s)", fixedrange=True, row=2, col=2)

    return fig


st.title("Simulador de Queda Livre")
st.caption("Versão didática: movimento vertical sob ação exclusiva da gravidade.")

with st.sidebar:
    st.header("Configuração da queda")

    preset_name = st.selectbox(
        "Objeto visual",
        options=list(OBJECT_PRESETS.keys()) + ["Personalizado"],
        index=1 if "Bola de futebol" in OBJECT_PRESETS else 0,
        help="Na queda livre ideal, a massa e o formato não alteram o tempo de queda. O objeto serve apenas como representação visual.",
    )

    if preset_name == "Personalizado":
        object_name = st.text_input("Nome do objeto", value="Objeto")
        falling_object = make_custom_object(
            name=object_name,
            mass=1.0,
            area=1.0,
            drag_coefficient=0.0,
        )
    else:
        preset = OBJECT_PRESETS[preset_name]
        falling_object = make_custom_object(
            name=preset.name,
            mass=1.0,
            area=1.0,
            drag_coefficient=0.0,
        )

    st.header("Condições iniciais")

    initial_height = st.number_input(
        "Altura inicial (m)",
        min_value=0.1,
        value=100.0,
        step=10.0,
    )

    initial_vx = st.number_input(
        "Velocidade horizontal inicial (m/s)",
        value=0.0,
        step=1.0,
        help="Use zero para queda vertical pura. Valores diferentes de zero mostram um lançamento horizontal.",
    )

    initial_vy = st.number_input(
        "Velocidade vertical inicial (m/s)",
        value=0.0,
        step=1.0,
        help="Valor positivo para cima e negativo para baixo.",
    )

    gravity = st.number_input(
        "Aceleração da gravidade g (m/s²)",
        min_value=0.0,
        value=9.81,
        step=0.01,
    )

    st.header("Precisão numérica")

    dt = st.number_input(
        "Passo de tempo dt (s)",
        min_value=0.001,
        max_value=1.0,
        value=0.01,
        step=0.001,
        format="%.3f",
    )

    max_time = st.number_input(
        "Tempo máximo de simulação (s)",
        min_value=1.0,
        value=300.0,
        step=10.0,
    )

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
impact = df.iloc[-1]

metric_cols = st.columns(5)
metric_cols[0].metric("Tempo de queda", f"{impact['t']:.2f} s")
metric_cols[1].metric("Velocidade final", f"{impact['speed']:.2f} m/s")
metric_cols[2].metric("vₓ final", f"{impact['vx']:.2f} m/s")
metric_cols[3].metric("vᵧ final", f"{impact['vy']:.2f} m/s")
metric_cols[4].metric("Gravidade", f"{gravity:.2f} m/s²")

st.info(
    "Nesta versão didática, a resistência do ar e o vento foram removidos. "
    "O movimento é calculado considerando apenas a ação da gravidade. "
    "Agora os gráficos são desenhados junto com a animação."
)

st.divider()
st.subheader("Animação sincronizada com gráficos")
st.caption(
    "A queda, os vetores, o gráfico altura × tempo e o gráfico das componentes da velocidade "
    "são construídos ao mesmo tempo. Use o Play do gráfico para visualizar a evolução."
)

animation_fig = build_synchronized_animation_figure(
    full_df=df,
    object_name=falling_object.name,
    gravity=gravity,
)
st.plotly_chart(
    animation_fig,
    use_container_width=True,
    config=ANIMATION_PLOTLY_CONFIG,
)

st.divider()
left, right = st.columns(2)

with left:
    st.subheader("Altura × tempo — gráfico completo")
    fig_y = px.line(df, x="t", y="y", labels={"t": "Tempo (s)", "y": "Altura (m)"})
    st.plotly_chart(fig_y, use_container_width=True)

    st.subheader("Componentes da velocidade × tempo — gráfico completo")
    velocity_df = df[["t", "speed", "vx", "vy"]].rename(
        columns={
            "speed": "|v| velocidade resultante",
            "vx": "vₓ componente horizontal",
            "vy": "vᵧ componente vertical",
        }
    )
    fig_speed = px.line(
        velocity_df,
        x="t",
        y=["|v| velocidade resultante", "vₓ componente horizontal", "vᵧ componente vertical"],
        labels={"t": "Tempo (s)", "value": "Velocidade (m/s)", "variable": "Grandeza"},
    )
    st.plotly_chart(fig_speed, use_container_width=True)

with right:
    st.subheader("Trajetória — gráfico completo")
    fig_traj = px.line(df, x="x", y="y", labels={"x": "Posição horizontal (m)", "y": "Altura (m)"})
    fig_traj.update_yaxes(scaleanchor="x", scaleratio=1)
    st.plotly_chart(fig_traj, use_container_width=True)

    st.subheader("Aceleração vertical × tempo — gráfico completo")
    fig_ay = px.line(df, x="t", y="ay", labels={"t": "Tempo (s)", "ay": "Aceleração vertical (m/s²)"})
    st.plotly_chart(fig_ay, use_container_width=True)

with st.expander("Ver tabela da simulação"):
    st.dataframe(df, use_container_width=True)

    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Baixar dados em CSV",
        data=csv,
        file_name="dados_queda_livre.csv",
        mime="text/csv",
    )

with st.expander("Interpretação física"):
    st.markdown(
        """
        - A velocidade resultante é formada pelas componentes horizontal e vertical.
        - A componente horizontal `vₓ` permanece constante quando não há resistência do ar.
        - A componente vertical `vᵧ` muda continuamente por causa da aceleração da gravidade.
        - O gráfico altura × tempo é desenhado junto com a queda, mostrando a diminuição da altura.
        - O gráfico das velocidades mostra `|v|`, `vₓ` e `vᵧ` sendo construídos no mesmo instante da animação.
        - Se `vₓ = 0`, o movimento é uma queda vertical pura; se `vₓ ≠ 0`, o movimento se torna bidimensional.
        """
    )
