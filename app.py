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
OBJECT_COLOR = "#FF7043"
DARK_BG = "#0E1117"
DARK_GRID = "#374151"
DARK_TEXT = "#F9FAFB"


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
        f"<br><b>Vetores:</b><br>"
        f"<span style='color:{VECTOR_RESULTANT_COLOR}'><b>roxo:</b></span> velocidade resultante<br>"
        f"<span style='color:{VECTOR_X_COLOR}'><b>azul:</b></span> componente horizontal<br>"
        f"<span style='color:{VECTOR_Y_COLOR}'><b>vermelho:</b></span> componente vertical"
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
    max_speed: float,
    max_abs_vx: float,
    max_abs_vy: float,
) -> dict[str, object]:
    """Calcula posição, sentido, tamanho e espessura dos vetores."""
    x = float(row["x"])
    y = float(row["y"])
    vx = float(row["vx"])
    vy = float(row["vy"])
    speed = float(row["speed"])

    return {
        "origin": (x, y),
        "v_end": (x + vx * vector_scale, y + vy * vector_scale),
        "vx_end": (x + vx * vector_scale, y),
        "vy_end": (x, y + vy * vector_scale),
        "resultant_width": 2 + 4 * (speed / max_speed if max_speed > 0 else 0),
        "vx_width": 2 + 3 * (abs(vx) / max_abs_vx if max_abs_vx > 0 else 0),
        "vy_width": 2 + 3 * (abs(vy) / max_abs_vy if max_abs_vy > 0 else 0),
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
            showlegend=False,
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
            showlegend=False,
        ),
        go.Scatter(
            x=[x, vx_end[0]],
            y=[y, vx_end[1]],
            mode="lines",
            line=dict(width=vector_data["vx_width"], color=VECTOR_X_COLOR, dash="dot"),
            hoverinfo="skip",
            showlegend=False,
        ),
        go.Scatter(
            x=[vx_end[0]],
            y=[vx_end[1]],
            mode="markers+text",
            text=["vₓ"],
            textposition="top center",
            textfont=dict(size=15, color=VECTOR_X_COLOR),
            marker=dict(size=14, color=VECTOR_X_COLOR, symbol=vector_data["vx_symbol"]),
            hovertemplate=f"vₓ = {row['vx']:.2f} m/s<extra></extra>",
            showlegend=False,
        ),
        go.Scatter(
            x=[x, vy_end[0]],
            y=[y, vy_end[1]],
            mode="lines",
            line=dict(width=vector_data["vy_width"], color=VECTOR_Y_COLOR, dash="dot"),
            hoverinfo="skip",
            showlegend=False,
        ),
        go.Scatter(
            x=[vy_end[0]],
            y=[vy_end[1]],
            mode="markers+text",
            text=["vᵧ"],
            textposition="middle right",
            textfont=dict(size=15, color=VECTOR_Y_COLOR),
            marker=dict(size=14, color=VECTOR_Y_COLOR, symbol=vector_data["vy_symbol"]),
            hovertemplate=f"vᵧ = {row['vy']:.2f} m/s<extra></extra>",
            showlegend=False,
        ),
    ]


def build_animation_with_dark_graphs(
    full_df: pd.DataFrame,
    object_name: str,
    gravity: float,
) -> go.Figure:
    """Animação no estilo anterior, mas com gráficos escuros construídos simultaneamente."""
    anim_df = sample_simulation_frames(full_df, max_frames=120)

    scene_y_max = max(float(anim_df["y"].max()), 1.0)
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

    scene_x_min -= 1.10 * vector_reference_length
    scene_x_max += 1.10 * vector_reference_length
    scene_y_min = min(ground_bottom - 0.50 * vector_reference_length, -0.05 * scene_y_max)
    scene_y_top = scene_y_max * 1.10 + 0.25 * vector_reference_length

    time_max = max(float(anim_df["t"].max()), 1e-9)
    velocity_min = min(float(anim_df["vy"].min()), float(anim_df["vx"].min()), 0.0)
    velocity_max = max(float(anim_df["speed"].max()), float(anim_df["vx"].max()), float(anim_df["vy"].max()), 1.0)
    velocity_margin = max(1.0, 0.10 * (velocity_max - velocity_min))
    accel_min = min(float(anim_df["ay"].min()), -gravity, 0.0)
    accel_max = max(float(anim_df["ay"].max()), -gravity, 0.0)
    accel_margin = max(0.5, 0.20 * (accel_max - accel_min if accel_max != accel_min else abs(gravity)))

    initial = anim_df.iloc[0]
    initial_vector_data = vector_geometry(
        row=initial,
        vector_scale=vector_scale,
        max_speed=max_speed,
        max_abs_vx=max_abs_vx,
        max_abs_vy=max_abs_vy,
    )

    fig = make_subplots(
        rows=2,
        cols=2,
        row_heights=[0.62, 0.38],
        column_widths=[0.62, 0.38],
        horizontal_spacing=0.07,
        vertical_spacing=0.16,
        subplot_titles=(
            "Animação visual da queda livre com vetores",
            "Painel da queda",
            "Componentes da velocidade × tempo",
            "Aceleração vertical × tempo",
        ),
    )

    fig.add_trace(go.Scatter(x=[initial["x"]], y=[initial["y"]], mode="lines", line=dict(width=4, color="#0D47A1"), name="Rastro", hoverinfo="skip", showlegend=False), row=1, col=1)
    fig.add_trace(
        go.Scatter(
            x=[initial["x"]],
            y=[initial["y"]],
            mode="markers",
            name="Objeto",
            marker=dict(size=34, color=OBJECT_COLOR, line=dict(width=3, color="#BF360C")),
            hovertemplate="Tempo: %{customdata[0]:.2f} s<br>Altura: %{y:.2f} m<br>Velocidade: %{customdata[1]:.2f} m/s<extra></extra>",
            customdata=[[initial["t"], initial["speed"]]],
            showlegend=False,
        ),
        row=1,
        col=1,
    )

    for trace in build_vector_traces(initial, initial_vector_data):
        fig.add_trace(trace, row=1, col=1)

    fig.add_trace(
        go.Scatter(
            x=[0.05],
            y=[0.50],
            mode="text",
            text=[format_panel_text(initial, object_name, gravity)],
            textposition="middle left",
            textfont=dict(size=15, color="#1F2937"),
            hoverinfo="skip",
            showlegend=False,
        ),
        row=1,
        col=2,
    )

    fig.add_trace(go.Scatter(x=[initial["t"]], y=[initial["speed"]], mode="lines", line=dict(width=4, color=VECTOR_RESULTANT_COLOR), name="|v| velocidade resultante"), row=2, col=1)
    fig.add_trace(go.Scatter(x=[initial["t"]], y=[initial["vx"]], mode="lines", line=dict(width=3, color=VECTOR_X_COLOR), name="vₓ componente horizontal"), row=2, col=1)
    fig.add_trace(go.Scatter(x=[initial["t"]], y=[initial["vy"]], mode="lines", line=dict(width=3, color="#FCA5A5"), name="vᵧ componente vertical"), row=2, col=1)
    fig.add_trace(
        go.Scatter(
            x=[initial["t"], initial["t"], initial["t"]],
            y=[initial["speed"], initial["vx"], initial["vy"]],
            mode="markers",
            marker=dict(size=9, color=[VECTOR_RESULTANT_COLOR, VECTOR_X_COLOR, "#FCA5A5"], line=dict(width=1, color="#F9FAFB")),
            name="valores atuais",
            hoverinfo="skip",
            showlegend=False,
        ),
        row=2,
        col=1,
    )

    fig.add_trace(go.Scatter(x=[initial["t"]], y=[initial["ay"]], mode="lines", line=dict(width=4, color="#60A5FA"), name="aᵧ aceleração vertical"), row=2, col=2)
    fig.add_trace(go.Scatter(x=[initial["t"]], y=[initial["ay"]], mode="markers", marker=dict(size=10, color="#60A5FA", line=dict(width=1, color="#F9FAFB")), name="aᵧ atual", hoverinfo="skip", showlegend=False), row=2, col=2)

    frames: list[go.Frame] = []
    slider_steps = []

    for index, row in anim_df.iterrows():
        trail_df = anim_df.iloc[: index + 1]
        chart_df = anim_df.iloc[: index + 1]
        frame_name = str(index)
        vector_data = vector_geometry(
            row=row,
            vector_scale=vector_scale,
            max_speed=max_speed,
            max_abs_vx=max_abs_vx,
            max_abs_vy=max_abs_vy,
        )

        frames.append(
            go.Frame(
                name=frame_name,
                data=[
                    go.Scatter(x=trail_df["x"], y=trail_df["y"], mode="lines", line=dict(width=4, color="#0D47A1"), hoverinfo="skip", showlegend=False),
                    go.Scatter(
                        x=[row["x"]],
                        y=[row["y"]],
                        mode="markers",
                        marker=dict(size=34, color=OBJECT_COLOR, line=dict(width=3, color="#BF360C")),
                        hovertemplate="Tempo: %{customdata[0]:.2f} s<br>Altura: %{y:.2f} m<br>Velocidade: %{customdata[1]:.2f} m/s<extra></extra>",
                        customdata=[[row["t"], row["speed"]]],
                        showlegend=False,
                    ),
                    *build_vector_traces(row, vector_data),
                    go.Scatter(x=[0.05], y=[0.50], mode="text", text=[format_panel_text(row, object_name, gravity)], textposition="middle left", textfont=dict(size=15, color="#1F2937"), hoverinfo="skip", showlegend=False),
                    go.Scatter(x=chart_df["t"], y=chart_df["speed"], mode="lines", line=dict(width=4, color=VECTOR_RESULTANT_COLOR), name="|v| velocidade resultante"),
                    go.Scatter(x=chart_df["t"], y=chart_df["vx"], mode="lines", line=dict(width=3, color=VECTOR_X_COLOR), name="vₓ componente horizontal"),
                    go.Scatter(x=chart_df["t"], y=chart_df["vy"], mode="lines", line=dict(width=3, color="#FCA5A5"), name="vᵧ componente vertical"),
                    go.Scatter(x=[row["t"], row["t"], row["t"]], y=[row["speed"], row["vx"], row["vy"]], mode="markers", marker=dict(size=9, color=[VECTOR_RESULTANT_COLOR, VECTOR_X_COLOR, "#FCA5A5"], line=dict(width=1, color="#F9FAFB")), hoverinfo="skip", showlegend=False),
                    go.Scatter(x=chart_df["t"], y=chart_df["ay"], mode="lines", line=dict(width=4, color="#60A5FA"), name="aᵧ aceleração vertical"),
                    go.Scatter(x=[row["t"]], y=[row["ay"]], mode="markers", marker=dict(size=10, color="#60A5FA", line=dict(width=1, color="#F9FAFB")), hoverinfo="skip", showlegend=False),
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

    fig.add_shape(type="rect", x0=scene_x_min, x1=scene_x_max, y0=0, y1=scene_y_top, fillcolor="#EAF7FF", line=dict(width=0), layer="below", row=1, col=1)
    fig.add_shape(type="rect", x0=scene_x_min, x1=scene_x_max, y0=scene_y_min, y1=0, fillcolor="#7CB342", line=dict(width=0), layer="below", row=1, col=1)
    fig.add_shape(type="line", x0=scene_x_min, x1=scene_x_max, y0=0, y1=0, line=dict(width=5, color="#33691E"), layer="above", row=1, col=1)
    fig.add_shape(type="rect", x0=0, x1=1, y0=0, y1=1, fillcolor="#FFFFFF", line=dict(width=2, color="#CBD5E1"), layer="below", row=1, col=2)
    fig.add_annotation(x=(scene_x_min + scene_x_max) / 2, y=scene_y_min * 0.55, text="<b>SOLO</b>", showarrow=False, font=dict(size=15, color="#1B5E20"), row=1, col=1)

    for subplot_ref in [
        ("x3 domain", "y3 domain"),
        ("x4 domain", "y4 domain"),
    ]:
        fig.add_shape(
            type="rect",
            xref=subplot_ref[0],
            yref=subplot_ref[1],
            x0=0,
            x1=1,
            y0=0,
            y1=1,
            fillcolor=DARK_BG,
            line=dict(width=0),
            layer="below",
        )

    fig.update_layout(
        height=880,
        margin=dict(l=20, r=20, t=70, b=55),
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1.0, font=dict(color=DARK_TEXT)),
        dragmode=False,
        plot_bgcolor=DARK_BG,
        paper_bgcolor=DARK_BG,
        font=dict(color=DARK_TEXT),
        updatemenus=[
            {
                "type": "buttons",
                "showactive": False,
                "direction": "left",
                "x": 0.02,
                "y": -0.07,
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
                        "args": [[None], {"frame": {"duration": 0, "redraw": False}, "transition": {"duration": 0}, "mode": "immediate"}],
                    },
                ],
            }
        ],
        sliders=[
            {
                "active": 0,
                "x": 0.18,
                "y": -0.07,
                "xanchor": "left",
                "yanchor": "top",
                "len": 0.76,
                "pad": {"t": 0, "b": 0},
                "bgcolor": "#E5E7EB",
                "activebgcolor": "#1D4ED8",
                "bordercolor": "#CBD5E1",
                "currentvalue": {"prefix": "Tempo: ", "suffix": "", "font": {"size": 14, "color": DARK_TEXT}},
                "steps": slider_steps,
            }
        ],
    )

    fig.update_xaxes(range=[scene_x_min, scene_x_max], showgrid=False, zeroline=False, title="", showticklabels=False, fixedrange=True, row=1, col=1)
    fig.update_yaxes(range=[scene_y_min, scene_y_top], showgrid=False, zeroline=False, title="Altura (m)", fixedrange=True, row=1, col=1)
    fig.update_xaxes(range=[0, 1], showgrid=False, zeroline=False, title="", showticklabels=False, fixedrange=True, row=1, col=2)
    fig.update_yaxes(range=[0, 1], showgrid=False, zeroline=False, title="", showticklabels=False, fixedrange=True, row=1, col=2)
    fig.update_xaxes(range=[0, time_max * 1.02], title="Tempo (s)", gridcolor=DARK_GRID, zerolinecolor=DARK_GRID, fixedrange=True, row=2, col=1)
    fig.update_yaxes(range=[velocity_min - velocity_margin, velocity_max + velocity_margin], title="Velocidade (m/s)", gridcolor=DARK_GRID, zerolinecolor=DARK_GRID, fixedrange=True, row=2, col=1)
    fig.update_xaxes(range=[0, time_max * 1.02], title="Tempo (s)", gridcolor=DARK_GRID, zerolinecolor=DARK_GRID, fixedrange=True, row=2, col=2)
    fig.update_yaxes(range=[accel_min - accel_margin, accel_max + accel_margin], title="Aceleração vertical (m/s²)", gridcolor=DARK_GRID, zerolinecolor=DARK_GRID, fixedrange=True, row=2, col=2)

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
        falling_object = make_custom_object(name=object_name, mass=1.0, area=1.0, drag_coefficient=0.0)
    else:
        preset = OBJECT_PRESETS[preset_name]
        falling_object = make_custom_object(name=preset.name, mass=1.0, area=1.0, drag_coefficient=0.0)

    st.header("Condições iniciais")

    initial_height = st.number_input("Altura inicial (m)", min_value=0.1, value=100.0, step=10.0)
    initial_vx = st.number_input(
        "Velocidade horizontal inicial (m/s)",
        value=0.0,
        step=1.0,
        help="Use zero para queda vertical pura. Valores diferentes de zero mostram um lançamento horizontal.",
    )
    initial_vy = st.number_input("Velocidade vertical inicial (m/s)", value=0.0, step=1.0, help="Valor positivo para cima e negativo para baixo.")
    gravity = st.number_input("Aceleração da gravidade g (m/s²)", min_value=0.0, value=9.81, step=0.01)

    st.header("Precisão numérica")
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
    data = simulate_fall(falling_object=falling_object, sim_config=sim_config, atmosphere_config=atmosphere_config, wind_config=wind_config)
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
    "Voltei a visualização para o estilo anterior: cena grande com painel lateral. "
    "Os gráficos escuros abaixo da cena agora são construídos simultaneamente com a animação."
)

st.divider()
st.subheader("Animação com gráficos escuros sincronizados")
st.caption(
    "A cena da queda fica no topo. Embaixo, os gráficos escuros de componentes da velocidade e aceleração vertical "
    "são desenhados junto com o movimento."
)

animation_fig = build_animation_with_dark_graphs(full_df=df, object_name=falling_object.name, gravity=gravity)
st.plotly_chart(animation_fig, use_container_width=True, config=ANIMATION_PLOTLY_CONFIG)

st.divider()
left, right = st.columns(2)

with left:
    st.subheader("Componentes da velocidade × tempo — gráfico completo")
    velocity_df = df[["t", "speed", "vx", "vy"]].rename(
        columns={"speed": "|v| velocidade resultante", "vx": "vₓ componente horizontal", "vy": "vᵧ componente vertical"}
    )
    fig_speed = px.line(
        velocity_df,
        x="t",
        y=["|v| velocidade resultante", "vₓ componente horizontal", "vᵧ componente vertical"],
        labels={"t": "Tempo (s)", "value": "Velocidade (m/s)", "variable": "Grandeza"},
        template="plotly_dark",
    )
    st.plotly_chart(fig_speed, use_container_width=True)

with right:
    st.subheader("Aceleração vertical × tempo — gráfico completo")
    fig_ay = px.line(df, x="t", y="ay", labels={"t": "Tempo (s)", "ay": "Aceleração vertical (m/s²)"}, template="plotly_dark")
    st.plotly_chart(fig_ay, use_container_width=True)

with st.expander("Ver tabela da simulação"):
    st.dataframe(df, use_container_width=True)
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("Baixar dados em CSV", data=csv, file_name="dados_queda_livre.csv", mime="text/csv")

with st.expander("Interpretação física"):
    st.markdown(
        """
        - A velocidade resultante é formada pelas componentes horizontal e vertical.
        - A componente horizontal `vₓ` permanece constante quando não há resistência do ar.
        - A componente vertical `vᵧ` muda continuamente por causa da aceleração da gravidade.
        - O gráfico das velocidades mostra `|v|`, `vₓ` e `vᵧ` sendo construídos no mesmo instante da animação.
        - O gráfico da aceleração vertical mostra que, na queda livre ideal, a aceleração permanece constante.
        """
    )
