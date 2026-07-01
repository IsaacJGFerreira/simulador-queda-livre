"""Interface Streamlit do simulador de queda livre."""

from __future__ import annotations

import time

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


def sample_simulation_frames(df: pd.DataFrame, max_frames: int = 180) -> pd.DataFrame:
    """Reduz a quantidade de pontos da simulação para facilitar a animação."""
    if len(df) <= max_frames:
        return df.reset_index(drop=True)

    indices = np.linspace(0, len(df) - 1, max_frames, dtype=int)
    indices = np.unique(indices)
    return df.iloc[indices].reset_index(drop=True)


def build_scene_figure(
    full_df: pd.DataFrame,
    shown_df: pd.DataFrame,
    object_name: str,
) -> go.Figure:
    """Monta a cena da queda com trajetória, objeto e solo."""
    max_y = max(float(full_df["y"].max()), 1.0)
    min_x = float(full_df["x"].min())
    max_x = float(full_df["x"].max())

    if abs(max_x - min_x) < 1.0:
        center_x = (max_x + min_x) / 2.0
        min_x = center_x - 1.0
        max_x = center_x + 1.0
    else:
        margin = 0.1 * (max_x - min_x)
        min_x -= margin
        max_x += margin

    current = shown_df.iloc[-1]

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=shown_df["x"],
            y=shown_df["y"],
            mode="lines",
            name="Trajetória",
            line=dict(width=3),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=[current["x"]],
            y=[current["y"]],
            mode="markers+text",
            name="Objeto",
            text=[object_name],
            textposition="top center",
            marker=dict(size=18, symbol="circle"),
        )
    )

    fig.add_hline(
        y=0,
        line_width=4,
        annotation_text="Solo",
        annotation_position="bottom right",
    )

    fig.update_layout(
        height=520,
        margin=dict(l=20, r=20, t=20, b=20),
        showlegend=False,
        xaxis_title="Posição horizontal (m)",
        yaxis_title="Altura (m)",
    )

    fig.update_xaxes(range=[min_x, max_x])
    fig.update_yaxes(range=[0, max_y * 1.05])

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
        area = st.number_input(
            "Área frontal (m²)",
            min_value=0.0001,
            value=0.01,
            step=0.001,
            format="%.4f",
        )
        drag_coefficient = st.number_input(
            "Coeficiente de arrasto Cd",
            min_value=0.0,
            value=0.47,
            step=0.01,
        )

        falling_object = make_custom_object(
            object_name,
            mass,
            area,
            drag_coefficient,
        )

    else:
        preset = OBJECT_PRESETS[preset_name]

        mass = st.number_input(
            "Massa (kg)",
            min_value=0.001,
            value=float(preset.mass),
            step=0.1,
        )
        area = st.number_input(
            "Área frontal (m²)",
            min_value=0.0001,
            value=float(preset.area),
            step=0.001,
            format="%.4f",
        )
        drag_coefficient = st.number_input(
            "Coeficiente de arrasto Cd",
            min_value=0.0,
            value=float(preset.drag_coefficient),
            step=0.01,
        )

        falling_object = make_custom_object(
            preset.name,
            mass,
            area,
            drag_coefficient,
        )

    st.header("Movimento inicial")

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
    )

    initial_vy = st.number_input(
        "Velocidade vertical inicial (m/s)",
        value=0.0,
        step=1.0,
        help="Valor positivo para cima e negativo para baixo.",
    )

    gravity = st.number_input(
        "Gravidade (m/s²)",
        min_value=0.0,
        value=9.81,
        step=0.01,
    )

    st.header("Resistência do ar")

    use_air_resistance = st.checkbox(
        "Considerar resistência do ar",
        value=True,
    )

    atmosphere_label = st.selectbox(
        "Modelo atmosférico",
        options=[
            "Densidade constante",
            "Densidade varia com altitude",
            "Densidade por pressão e temperatura",
        ],
        index=0,
    )

    atmosphere_model_map = {
        "Densidade constante": "constant",
        "Densidade varia com altitude": "exponential",
        "Densidade por pressão e temperatura": "ideal_gas",
    }

    rho0 = st.number_input(
        "Densidade de referência ρ₀ (kg/m³)",
        min_value=0.0,
        value=1.225,
        step=0.01,
    )

    pressure = st.number_input(
        "Pressão atmosférica (Pa)",
        min_value=0.0,
        value=101_325.0,
        step=500.0,
    )

    temperature_celsius = st.number_input(
        "Temperatura (°C)",
        value=15.0,
        step=1.0,
    )

    scale_height = st.number_input(
        "Altura de escala atmosférica H (m)",
        min_value=1.0,
        value=8_500.0,
        step=100.0,
    )

    st.header("Vento")

    wind_label = st.selectbox(
        "Modelo de vento",
        options=[
            "Sem vento",
            "Vento constante",
            "Vento aumenta com altitude",
        ],
        index=0,
    )

    wind_model_map = {
        "Sem vento": "none",
        "Vento constante": "constant",
        "Vento aumenta com altitude": "linear_shear",
    }

    wind_x = st.number_input(
        "Vento horizontal base (m/s)",
        value=0.0,
        step=1.0,
    )

    wind_y = st.number_input(
        "Vento vertical (m/s)",
        value=0.0,
        step=1.0,
        help="Valor positivo para cima.",
    )

    shear_per_meter = st.number_input(
        "Aumento do vento horizontal por metro (1/s)",
        value=0.0,
        step=0.001,
        format="%.4f",
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

metric_cols[0].metric(
    "Tempo de queda",
    f"{impact['t']:.2f} s",
)

metric_cols[1].metric(
    "Velocidade final",
    f"{impact['speed']:.2f} m/s",
)

metric_cols[2].metric(
    "Distância horizontal",
    f"{impact['x']:.2f} m",
)

metric_cols[3].metric(
    "Densidade no solo",
    f"{rho_ground:.3f} kg/m³",
)

metric_cols[4].metric(
    "Velocidade terminal aprox.",
    "—" if terminal_velocity is None else f"{terminal_velocity:.2f} m/s",
)


st.divider()

st.subheader("Animação da queda")
st.caption("Visualize o corpo caindo e acompanhe o cronômetro ao lado.")

anim_df = sample_simulation_frames(df, max_frames=180)

animation_cols = st.columns([2.3, 1])

with animation_cols[1]:
    animation_delay_ms = st.slider(
        "Velocidade da animação (ms por quadro)",
        min_value=20,
        max_value=300,
        value=60,
        step=10,
    )

    frame_index = st.slider(
        "Quadro manual",
        min_value=0,
        max_value=len(anim_df) - 1,
        value=0,
        step=1,
    )

    run_animation = st.button("▶️ Rodar animação")

plot_placeholder = animation_cols[0].empty()
timer_placeholder = animation_cols[1].empty()
height_placeholder = animation_cols[1].empty()
speed_placeholder = animation_cols[1].empty()
horizontal_placeholder = animation_cols[1].empty()
density_placeholder = animation_cols[1].empty()


def update_animation(row: pd.Series, shown_df: pd.DataFrame) -> None:
    """Atualiza a cena e o painel lateral da animação."""
    fig_scene = build_scene_figure(
        full_df=anim_df,
        shown_df=shown_df,
        object_name=falling_object.name,
    )

    plot_placeholder.plotly_chart(
        fig_scene,
        use_container_width=True,
    )

    timer_placeholder.metric(
        "Cronômetro",
        f"{row['t']:.2f} s",
    )

    height_placeholder.metric(
        "Altura atual",
        f"{row['y']:.2f} m",
    )

    speed_placeholder.metric(
        "Velocidade atual",
        f"{row['speed']:.2f} m/s",
    )

    horizontal_placeholder.metric(
        "Distância horizontal",
        f"{row['x']:.2f} m",
    )

    density_placeholder.metric(
        "Densidade do ar",
        f"{row['rho']:.3f} kg/m³",
    )


if run_animation:
    for i in range(len(anim_df)):
        shown_df = anim_df.iloc[: i + 1]
        row = shown_df.iloc[-1]

        update_animation(row, shown_df)

        time.sleep(animation_delay_ms / 1000)

else:
    shown_df = anim_df.iloc[: frame_index + 1]
    row = shown_df.iloc[-1]

    update_animation(row, shown_df)


st.divider()

left, right = st.columns(2)

with left:
    st.subheader("Altura × tempo")

    fig_y = px.line(
        df,
        x="t",
        y="y",
        labels={
            "t": "Tempo (s)",
            "y": "Altura (m)",
        },
    )

    st.plotly_chart(fig_y, use_container_width=True)

    st.subheader("Velocidade × tempo")

    fig_speed = px.line(
        df,
        x="t",
        y="speed",
        labels={
            "t": "Tempo (s)",
            "speed": "Velocidade (m/s)",
        },
    )

    st.plotly_chart(fig_speed, use_container_width=True)

with right:
    st.subheader("Trajetória")

    fig_traj = px.line(
        df,
        x="x",
        y="y",
        labels={
            "x": "Posição horizontal (m)",
            "y": "Altura (m)",
        },
    )

    fig_traj.update_yaxes(
        scaleanchor="x",
        scaleratio=1,
    )

    st.plotly_chart(fig_traj, use_container_width=True)

    st.subheader("Aceleração vertical × tempo")

    fig_ay = px.line(
        df,
        x="t",
        y="ay",
        labels={
            "t": "Tempo (s)",
            "ay": "Aceleração vertical (m/s²)",
        },
    )

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
