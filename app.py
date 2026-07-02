"""Interface Streamlit do simulador didático de queda livre."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

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
        f"⚡ <b>Velocidade:</b> {row['speed']:.2f} m/s<br>"
        f"➡ <b>Distância horizontal:</b> {row['x']:.2f} m<br>"
        f"⬇ <b>Aceleração:</b> {gravity:.2f} m/s²<br>"
        f"<br><b>Modelo:</b> queda livre ideal"
    )


def build_visual_animation_figure(
    full_df: pd.DataFrame,
    object_name: str,
    gravity: float,
) -> go.Figure:
    """Cria uma animação visual 2D usando frames nativos do Plotly."""
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
    panel_gap = 0.08 * scene_width
    panel_width = 0.55 * scene_width
    panel_x_min = scene_x_max + panel_gap
    panel_x_max = panel_x_min + panel_width
    panel_x_text = panel_x_min + 0.08 * panel_width
    panel_y_text = 0.58 * scene_y_max

    initial = anim_df.iloc[0]
    initial_text = format_panel_text(initial, object_name, gravity)

    frames: list[go.Frame] = []
    slider_steps = []

    for index, row in anim_df.iterrows():
        trail_df = anim_df.iloc[: index + 1]
        frame_name = str(index)

        frames.append(
            go.Frame(
                name=frame_name,
                data=[
                    go.Scatter(
                        x=trail_df["x"],
                        y=trail_df["y"],
                        mode="lines",
                        line=dict(width=4, color="#0D47A1"),
                        hoverinfo="skip",
                    ),
                    go.Scatter(
                        x=[row["x"]],
                        y=[row["y"]],
                        mode="markers",
                        marker=dict(
                            size=34,
                            color="#FF7043",
                            line=dict(width=3, color="#BF360C"),
                        ),
                        hovertemplate=(
                            "Tempo: %{customdata[0]:.2f} s<br>"
                            "Altura: %{y:.2f} m<br>"
                            "Velocidade: %{customdata[1]:.2f} m/s<extra></extra>"
                        ),
                        customdata=[[row["t"], row["speed"]]],
                    ),
                    go.Scatter(
                        x=[panel_x_text],
                        y=[panel_y_text],
                        mode="text",
                        text=[format_panel_text(row, object_name, gravity)],
                        textposition="middle left",
                        textfont=dict(size=17, color="#1F2937"),
                        hoverinfo="skip",
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

    fig = go.Figure(
        data=[
            go.Scatter(
                x=[initial["x"]],
                y=[initial["y"]],
                mode="lines",
                line=dict(width=4, color="#0D47A1"),
                name="Rastro",
                hoverinfo="skip",
            ),
            go.Scatter(
                x=[initial["x"]],
                y=[initial["y"]],
                mode="markers",
                name="Objeto",
                marker=dict(
                    size=34,
                    color="#FF7043",
                    line=dict(width=3, color="#BF360C"),
                ),
                hovertemplate=(
                    "Tempo: %{customdata[0]:.2f} s<br>"
                    "Altura: %{y:.2f} m<br>"
                    "Velocidade: %{customdata[1]:.2f} m/s<extra></extra>"
                ),
                customdata=[[initial["t"], initial["speed"]]],
            ),
            go.Scatter(
                x=[panel_x_text],
                y=[panel_y_text],
                mode="text",
                name="Painel",
                text=[initial_text],
                textposition="middle left",
                textfont=dict(size=17, color="#1F2937"),
                hoverinfo="skip",
            ),
        ],
        frames=frames,
    )

    fig.update_layout(
        height=560,
        margin=dict(l=20, r=20, t=60, b=40),
        title=dict(
            text="Animação visual da queda livre",
            x=0.02,
            xanchor="left",
        ),
        showlegend=False,
        dragmode=False,
        xaxis=dict(
            range=[scene_x_min, panel_x_max],
            showgrid=False,
            zeroline=False,
            title="",
            showticklabels=False,
            fixedrange=True,
        ),
        yaxis=dict(
            range=[ground_bottom, scene_y_max * 1.08],
            showgrid=False,
            zeroline=False,
            title="Altura (m)",
            fixedrange=True,
        ),
        plot_bgcolor="#EAF7FF",
        paper_bgcolor="white",
        shapes=[
            dict(
                type="rect",
                x0=scene_x_min,
                x1=scene_x_max,
                y0=0,
                y1=scene_y_max * 1.08,
                fillcolor="#EAF7FF",
                line=dict(width=0),
                layer="below",
            ),
            dict(
                type="rect",
                x0=scene_x_min,
                x1=scene_x_max,
                y0=ground_bottom,
                y1=0,
                fillcolor="#7CB342",
                line=dict(width=0),
                layer="below",
            ),
            dict(
                type="line",
                x0=scene_x_min,
                x1=scene_x_max,
                y0=0,
                y1=0,
                line=dict(width=5, color="#33691E"),
                layer="above",
            ),
            dict(
                type="rect",
                x0=panel_x_min,
                x1=panel_x_max,
                y0=ground_bottom,
                y1=scene_y_max * 1.08,
                fillcolor="#FFFFFF",
                line=dict(width=2, color="#CBD5E1"),
                layer="below",
            ),
        ],
        annotations=[
            dict(
                x=(panel_x_min + panel_x_max) / 2,
                y=scene_y_max * 0.95,
                text="<b>Painel da queda</b>",
                showarrow=False,
                font=dict(size=20, color="#111827"),
                align="center",
            ),
            dict(
                x=(scene_x_min + scene_x_max) / 2,
                y=ground_bottom * 0.48,
                text="<b>SOLO</b>",
                showarrow=False,
                font=dict(size=16, color="#1B5E20"),
            ),
        ],
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
                                "frame": {
                                    "duration": ANIMATION_FRAME_DURATION_MS,
                                    "redraw": True,
                                },
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
                "currentvalue": {
                    "prefix": "Tempo: ",
                    "suffix": "",
                    "font": {"size": 14, "color": "#111827"},
                },
                "steps": slider_steps,
            }
        ],
    )

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
metric_cols[2].metric("Altura inicial", f"{initial_height:.2f} m")
metric_cols[3].metric("Distância horizontal", f"{impact['x']:.2f} m")
metric_cols[4].metric("Gravidade", f"{gravity:.2f} m/s²")

st.info(
    "Nesta versão didática, a resistência do ar e o vento foram removidos. "
    "O movimento é calculado considerando apenas a ação da gravidade."
)

st.divider()
st.subheader("Animação visual 2D")
st.caption("Cena simplificada com céu, solo, corpo em queda, rastro e painel lateral.")

animation_fig = build_visual_animation_figure(
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
    st.subheader("Altura × tempo")
    fig_y = px.line(df, x="t", y="y", labels={"t": "Tempo (s)", "y": "Altura (m)"})
    st.plotly_chart(fig_y, use_container_width=True)

    st.subheader("Velocidade × tempo")
    fig_speed = px.line(df, x="t", y="speed", labels={"t": "Tempo (s)", "speed": "Velocidade (m/s)"})
    st.plotly_chart(fig_speed, use_container_width=True)

with right:
    st.subheader("Trajetória")
    fig_traj = px.line(df, x="x", y="y", labels={"x": "Posição horizontal (m)", "y": "Altura (m)"})
    fig_traj.update_yaxes(scaleanchor="x", scaleratio=1)
    st.plotly_chart(fig_traj, use_container_width=True)

    st.subheader("Aceleração vertical × tempo")
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
        - A queda livre ideal considera apenas a força gravitacional.
        - A aceleração vertical é constante e aponta para baixo.
        - Se a velocidade vertical inicial for zero, o corpo parte do repouso e acelera durante a queda.
        - A massa e o formato do objeto não alteram o tempo de queda nesse modelo ideal.
        - A velocidade horizontal inicial, quando diferente de zero, cria uma trajetória em duas dimensões.
        """
    )
