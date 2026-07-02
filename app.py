"""Interface Streamlit do simulador de queda livre."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from atmosphere import AtmosphereConfig, get_air_density
from objects import OBJECT_PRESETS, make_custom_object
from physics import SimulationConfig, estimate_terminal_velocity, simulate_fall
from wind import WindConfig


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


def format_panel_text(row: pd.Series, object_name: str, air_resistance: bool) -> str:
    """Texto do painel lateral da animação."""
    drag_status = "Ativa" if air_resistance else "Desativada"

    return (
        f"<b>{object_name}</b><br><br>"
        f"⏱ <b>Tempo:</b> {row['t']:.2f} s<br>"
        f"↕ <b>Altura:</b> {row['y']:.2f} m<br>"
        f"⚡ <b>Velocidade:</b> {row['speed']:.2f} m/s<br>"
        f"➡ <b>Distância:</b> {row['x']:.2f} m<br>"
        f"ρ <b>Densidade:</b> {row['rho']:.3f} kg/m³<br>"
        f"🌬 <b>Arrasto:</b> {drag_status}"
    )


def build_visual_animation_figure(
    full_df: pd.DataFrame,
    object_name: str,
    air_resistance: bool,
    frame_duration_ms: int,
) -> go.Figure:
    """Cria uma animação visual 2D usando os frames nativos do Plotly."""
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
    panel_width = 0.48 * scene_width
    panel_x_min = scene_x_max + panel_gap
    panel_x_max = panel_x_min + panel_width
    panel_x_text = panel_x_min + 0.08 * panel_width
    panel_y_text = 0.58 * scene_y_max

    initial = anim_df.iloc[0]
    initial_text = format_panel_text(initial, object_name, air_resistance)

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
                        text=[format_panel_text(row, object_name, air_resistance)],
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
            text="Animação visual da queda",
            x=0.02,
            xanchor="left",
        ),
        showlegend=False,
        xaxis=dict(
            range=[scene_x_min, panel_x_max],
            showgrid=False,
            zeroline=False,
            title="",
            showticklabels=False,
        ),
        yaxis=dict(
            range=[ground_bottom, scene_y_max * 1.08],
            showgrid=False,
            zeroline=False,
            title="",
        ),
        plot_bgcolor="#EAF7FF",
        paper_bgcolor="white",
        shapes=[
            # Céu
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
            # Solo
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
            # Linha superior do solo
            dict(
                type="line",
                x0=scene_x_min,
                x1=scene_x_max,
                y0=0,
                y1=0,
                line=dict(width=5, color="#33691E"),
                layer="above",
            ),
            # Painel lateral
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
                "buttons": [
                    {
                        "label": "▶ Play",
                        "method": "animate",
                        "args": [
                            None,
                            {
                                "frame": {"duration": frame_duration_ms, "redraw": True},
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
                "currentvalue": {
                    "prefix": "Tempo: ",
                    "suffix": "",
                    "font": {"size": 14},
                },
                "steps": slider_steps,
            }
        ],
    )

    return fig


st.title("Simulador de Queda Livre")
st.caption("Compare a queda ideal com modelos que incluem resistência do ar, atmosfera e vento.")

with st.sidebar:
    st.header("Objeto")

    preset_name = st.selectbox(
        "Objeto predefinido",
        options=list(OBJECT_PRESETS.keys()) + ["Personalizado"],
        index=0,
    )

    if preset_name == "Personalizado":
        object_name = st.text_input("Nome do objeto", value="Objeto personalizado")
        mass = st.number_input("Massa (kg)", min_value=0.001, value=1.0, step=0.1)
        area = st.number_input("Área frontal (m²)", min_value=0.0001, value=0.01, step=0.001, format="%.4f")
        drag_coefficient = st.number_input("Coeficiente de arrasto Cd", min_value=0.0, value=0.47, step=0.01)
        falling_object = make_custom_object(object_name, mass, area, drag_coefficient)
    else:
        preset = OBJECT_PRESETS[preset_name]
        mass = st.number_input("Massa (kg)", min_value=0.001, value=float(preset.mass), step=0.1)
        area = st.number_input("Área frontal (m²)", min_value=0.0001, value=float(preset.area), step=0.001, format="%.4f")
        drag_coefficient = st.number_input("Coeficiente de arrasto Cd", min_value=0.0, value=float(preset.drag_coefficient), step=0.01)
        falling_object = make_custom_object(preset.name, mass, area, drag_coefficient)

    st.header("Movimento inicial")
    initial_height = st.number_input("Altura inicial (m)", min_value=0.1, value=100.0, step=10.0)
    initial_vx = st.number_input("Velocidade horizontal inicial (m/s)", value=0.0, step=1.0)
    initial_vy = st.number_input(
        "Velocidade vertical inicial (m/s)",
        value=0.0,
        step=1.0,
        help="Valor positivo para cima e negativo para baixo.",
    )
    gravity = st.number_input("Gravidade (m/s²)", min_value=0.0, value=9.81, step=0.01)

    st.header("Resistência do ar")
    use_air_resistance = st.checkbox("Considerar resistência do ar", value=True)

    atmosphere_label = st.selectbox(
        "Modelo atmosférico",
        options=["Densidade constante", "Densidade varia com altitude", "Densidade por pressão e temperatura"],
        index=0,
    )

    atmosphere_model_map = {
        "Densidade constante": "constant",
        "Densidade varia com altitude": "exponential",
        "Densidade por pressão e temperatura": "ideal_gas",
    }

    rho0 = st.number_input("Densidade de referência ρ₀ (kg/m³)", min_value=0.0, value=1.225, step=0.01)
    pressure = st.number_input("Pressão atmosférica (Pa)", min_value=0.0, value=101_325.0, step=500.0)
    temperature_celsius = st.number_input("Temperatura (°C)", value=15.0, step=1.0)
    scale_height = st.number_input("Altura de escala atmosférica H (m)", min_value=1.0, value=8_500.0, step=100.0)

    st.header("Vento")
    wind_label = st.selectbox(
        "Modelo de vento",
        options=["Sem vento", "Vento constante", "Vento aumenta com altitude"],
        index=0,
    )

    wind_model_map = {
        "Sem vento": "none",
        "Vento constante": "constant",
        "Vento aumenta com altitude": "linear_shear",
    }

    wind_x = st.number_input("Vento horizontal base (m/s)", value=0.0, step=1.0)
    wind_y = st.number_input("Vento vertical (m/s)", value=0.0, step=1.0, help="Valor positivo para cima.")
    shear_per_meter = st.number_input("Aumento do vento horizontal por metro (1/s)", value=0.0, step=0.001, format="%.4f")

    st.header("Precisão numérica")
    dt = st.number_input("Passo de tempo dt (s)", min_value=0.001, max_value=1.0, value=0.01, step=0.001, format="%.3f")
    max_time = st.number_input("Tempo máximo de simulação (s)", min_value=1.0, value=300.0, step=10.0)

atmosphere_config = AtmosphereConfig(
    model=atmosphere_model_map[atmosphere_label],
    rho0=rho0,
    pressure=pressure,
    temperature=temperature_celsius + 273.15,
    scale_height=scale_height,
)

wind_config = WindConfig(
    model=wind_model_map[wind_label],
    horizontal_speed=wind_x,
    vertical_speed=wind_y,
    shear_per_meter=shear_per_meter,
)

sim_config = SimulationConfig(
    initial_height=initial_height,
    initial_vx=initial_vx,
    initial_vy=initial_vy,
    gravity=gravity,
    dt=dt,
    max_time=max_time,
    use_air_resistance=use_air_resistance,
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
rho_ground = get_air_density(0.0, atmosphere_config)
terminal_velocity = estimate_terminal_velocity(
    mass=falling_object.mass,
    gravity=gravity,
    rho=rho_ground,
    drag_coefficient=falling_object.drag_coefficient,
    area=falling_object.area,
)

metric_cols = st.columns(5)
metric_cols[0].metric("Tempo de queda", f"{impact['t']:.2f} s")
metric_cols[1].metric("Velocidade final", f"{impact['speed']:.2f} m/s")
metric_cols[2].metric("Distância horizontal", f"{impact['x']:.2f} m")
metric_cols[3].metric("Densidade no solo", f"{rho_ground:.3f} kg/m³")
metric_cols[4].metric(
    "Velocidade terminal aprox.",
    "—" if terminal_velocity is None else f"{terminal_velocity:.2f} m/s",
)

st.divider()
st.subheader("Animação visual 2D")
st.caption("Cena simplificada com céu, solo, corpo em queda, rastro e painel lateral. Use o botão Play do próprio gráfico.")

frame_duration_ms = st.slider(
    "Velocidade da animação nativa do Plotly (ms por quadro)",
    min_value=20,
    max_value=180,
    value=55,
    step=5,
)

animation_fig = build_visual_animation_figure(
    full_df=df,
    object_name=falling_object.name,
    air_resistance=use_air_resistance,
    frame_duration_ms=frame_duration_ms,
)
st.plotly_chart(animation_fig, use_container_width=True)

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
        - Sem resistência do ar, a aceleração vertical permanece aproximadamente igual a `-g`.
        - Com resistência do ar, a força de arrasto cresce com a velocidade relativa entre o objeto e o ar.
        - Quando o peso e o arrasto se equilibram, o objeto se aproxima da velocidade terminal.
        - O vento altera a velocidade relativa, portanto também altera a direção e a intensidade do arrasto.
        - Uma atmosfera mais densa aumenta a resistência do ar; uma atmosfera rarefeita reduz esse efeito.
        """
    )
