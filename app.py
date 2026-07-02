"""Interface Streamlit do simulador didático de queda livre."""

from __future__ import annotations

import math

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

# Paleta principal do painel solicitado
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


def add_dashboard_background(fig: go.Figure) -> None:
    """Adiciona fundo principal, barra superior e cards dos gráficos."""
    # Barra superior de resumo
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

    # Cards dos três gráficos
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


def add_summary_annotations(
    fig: go.Figure,
    initial_height: float,
    impact_time: float,
    initial_vx: float,
) -> None:
    """Adiciona os quatro cards superiores de informação."""
    vertical_fall = "Sim" if abs(initial_vx) < 1e-9 else "Não"

    items = [
        {
            "x_icon": 0.055,
            "x_text": 0.085,
            "icon": "↑",
            "title": "Altura inicial",
            "value": f"{format_br(initial_height, 1)} m",
            "color": BLUE,
            "bg": "rgba(38, 135, 255, 0.18)",
        },
        {
            "x_icon": 0.305,
            "x_text": 0.335,
            "icon": "◷",
            "title": "Tempo de simulação",
            "value": f"{format_br(impact_time, 2)} s",
            "color": GREEN,
            "bg": "rgba(74, 222, 128, 0.16)",
        },
        {
            "x_icon": 0.555,
            "x_text": 0.585,
            "icon": "→",
            "title": "Velocidade horizontal (Vx)",
            "value": f"{format_br(initial_vx, 2)} m/s",
            "color": CYAN,
            "bg": "rgba(34, 211, 238, 0.16)",
        },
        {
            "x_icon": 0.805,
            "x_text": 0.835,
            "icon": "↓",
            "title": "Queda livre vertical",
            "value": vertical_fall,
            "color": PINK,
            "bg": "rgba(255, 111, 145, 0.16)",
        },
    ]

    for item in items:
        fig.add_annotation(
            xref="x domain",
            yref="y domain",
            x=item["x_icon"],
            y=0.50,
            text=f"<b>{item['icon']}</b>",
            showarrow=False,
            font=dict(size=20, color=item["color"]),
            bgcolor=item["bg"],
            bordercolor=item["bg"],
            borderpad=8,
        )
        fig.add_annotation(
            xref="x domain",
            yref="y domain",
            x=item["x_text"],
            y=0.61,
            text=item["title"],
            showarrow=False,
            xanchor="left",
            align="left",
            font=dict(size=12, color="#d7e2ee"),
        )
        fig.add_annotation(
            xref="x domain",
            yref="y domain",
            x=item["x_text"],
            y=0.39,
            text=f"<b>{item['value']}</b>",
            showarrow=False,
            xanchor="left",
            align="left",
            font=dict(size=18, color=item["color"]),
        )


def add_chart_header(
    fig: go.Figure,
    xref: str,
    yref: str,
    icon: str,
    title: str,
    icon_color: str,
    icon_bg: str,
    legend_html: str,
) -> None:
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

    add_chart_header(
        fig,
        xref="x2 domain",
        yref="y2 domain",
        icon="↗",
        title="Componentes da velocidade × tempo",
        icon_color=PURPLE,
        icon_bg="rgba(139, 92, 246, 0.18)",
        legend_html=velocity_legend,
    )
    add_chart_header(
        fig,
        xref="x3 domain",
        yref="y3 domain",
        icon="⌁",
        title="Altura × tempo",
        icon_color=CYAN,
        icon_bg="rgba(34, 211, 238, 0.16)",
        legend_html=height_legend,
    )
    add_chart_header(
        fig,
        xref="x4 domain",
        yref="y4 domain",
        icon="→",
        title="Distância horizontal × tempo",
        icon_color=GREEN,
        icon_bg="rgba(74, 222, 128, 0.16)",
        legend_html=distance_legend,
    )


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


def make_value_label_trace(
    row: pd.Series,
) -> go.Scatter:
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


def build_professional_dashboard(
    full_df: pd.DataFrame,
    initial_height: float,
    initial_vx: float,
) -> go.Figure:
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

    # Traço invisível para ativar o eixo da barra superior
    fig.add_trace(
        go.Scatter(x=[0], y=[0], mode="markers", marker=dict(opacity=0), showlegend=False, hoverinfo="skip"),
        row=1,
        col=1,
    )

    # Gráfico 1: velocidades
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

    # Gráfico 2: altura
    fig.add_trace(go.Scatter(x=[initial["t"]], y=[initial["y"]], mode="lines", line=dict(color=CYAN, width=3), line_shape="spline", hoverinfo="skip", showlegend=False), row=2, col=2)
    fig.add_trace(go.Scatter(x=[initial["t"]], y=[initial["y"]], mode="markers", marker=dict(size=9, color=CYAN, line=dict(width=1, color=TEXT_PRIMARY)), hoverinfo="skip", showlegend=False), row=2, col=2)

    # Gráfico 3: distância horizontal
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
                        {
                            "mode": "immediate",
                            "frame": {"duration": 0, "redraw": True},
                            "transition": {"duration": 0},
                        },
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
                "y": -0.09,
                "xanchor": "left",
                "yanchor": "top",
                "len": 0.76,
                "pad": {"t": 0, "b": 0},
                "bgcolor": "#1d3346",
                "activebgcolor": BLUE,
                "bordercolor": BORDER_CARD,
                "currentvalue": {
                    "prefix": "Tempo: ",
                    "suffix": "",
                    "font": {"size": 14, "color": TEXT_PRIMARY},
                },
                "steps": slider_steps,
            }
        ],
    )

    # Eixo invisível da barra superior
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
            Visualize o corpo em queda e acompanhe, em tempo real, os gráficos de velocidade,
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
metric_cols[2].metric("vₓ final", f"{impact['vx']:.2f} m/s")
metric_cols[3].metric("vᵧ final", f"{impact['vy']:.2f} m/s")
metric_cols[4].metric("Distância horizontal", f"{impact['distancia_horizontal']:.2f} m")

st.markdown(
    """
    <div class="section-card">
        <div class="section-title">Painel gráfico sincronizado</div>
        <div class="section-subtitle">
            A barra superior resume a simulação. Os três cards abaixo mostram as curvas sendo construídas durante a execução.
            As cores foram mantidas de forma consistente para cada grandeza física.
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
