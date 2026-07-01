"""Motor físico do simulador de queda livre."""

from __future__ import annotations

import math
from dataclasses import dataclass

from atmosphere import AtmosphereConfig, get_air_density
from objects import FallingObject
from wind import WindConfig, get_wind_velocity


@dataclass(frozen=True)
class SimulationConfig:
    """Configuração geral da simulação."""

    initial_height: float = 100.0
    initial_x: float = 0.0
    initial_vx: float = 0.0
    initial_vy: float = 0.0
    gravity: float = 9.81
    dt: float = 0.01
    max_time: float = 300.0
    use_air_resistance: bool = True


def drag_force_components(
    rho: float,
    drag_coefficient: float,
    area: float,
    relative_vx: float,
    relative_vy: float,
) -> tuple[float, float, float]:
    """Calcula as componentes da força de arrasto.

    A força de arrasto sempre aponta em sentido oposto à velocidade relativa
    entre o objeto e o ar.
    """
    relative_speed = math.hypot(relative_vx, relative_vy)

    if relative_speed == 0 or rho == 0 or area == 0 or drag_coefficient == 0:
        return 0.0, 0.0, relative_speed

    drag_magnitude = 0.5 * rho * drag_coefficient * area * relative_speed**2

    drag_x = -drag_magnitude * relative_vx / relative_speed
    drag_y = -drag_magnitude * relative_vy / relative_speed

    return drag_x, drag_y, relative_speed


def estimate_terminal_velocity(
    mass: float,
    gravity: float,
    rho: float,
    drag_coefficient: float,
    area: float,
) -> float | None:
    """Estima a velocidade terminal vertical para densidade constante."""
    denominator = rho * drag_coefficient * area
    if denominator <= 0:
        return None
    return math.sqrt((2 * mass * gravity) / denominator)


def simulate_fall(
    falling_object: FallingObject,
    sim_config: SimulationConfig,
    atmosphere_config: AtmosphereConfig,
    wind_config: WindConfig,
) -> list[dict[str, float]]:
    """Executa a simulação numérica da queda.

    Convenção:
        - x: posição horizontal, em m.
        - y: altitude, em m.
        - vy positivo para cima.
        - gravidade atua para baixo.
    """
    if sim_config.initial_height < 0:
        raise ValueError("A altura inicial não pode ser negativa.")
    if sim_config.dt <= 0:
        raise ValueError("O passo de tempo deve ser maior que zero.")
    if sim_config.gravity < 0:
        raise ValueError("A gravidade não pode ser negativa.")
    if sim_config.max_time <= 0:
        raise ValueError("O tempo máximo deve ser maior que zero.")

    t = 0.0
    x = sim_config.initial_x
    y = sim_config.initial_height
    vx = sim_config.initial_vx
    vy = sim_config.initial_vy

    data: list[dict[str, float]] = []

    while y > 0 and t <= sim_config.max_time:
        rho = get_air_density(y, atmosphere_config)
        wind_x, wind_y = get_wind_velocity(y, wind_config)

        relative_vx = vx - wind_x
        relative_vy = vy - wind_y

        if sim_config.use_air_resistance:
            drag_x, drag_y, relative_speed = drag_force_components(
                rho=rho,
                drag_coefficient=falling_object.drag_coefficient,
                area=falling_object.area,
                relative_vx=relative_vx,
                relative_vy=relative_vy,
            )
        else:
            drag_x = 0.0
            drag_y = 0.0
            relative_speed = math.hypot(relative_vx, relative_vy)

        weight_y = -falling_object.mass * sim_config.gravity

        force_x = drag_x
        force_y = weight_y + drag_y

        ax = force_x / falling_object.mass
        ay = force_y / falling_object.mass

        speed = math.hypot(vx, vy)

        data.append(
            {
                "t": t,
                "x": x,
                "y": y,
                "vx": vx,
                "vy": vy,
                "speed": speed,
                "ax": ax,
                "ay": ay,
                "rho": rho,
                "wind_x": wind_x,
                "wind_y": wind_y,
                "relative_speed": relative_speed,
                "drag_x": drag_x,
                "drag_y": drag_y,
            }
        )

        old_t = t
        old_x = x
        old_y = y
        old_vx = vx
        old_vy = vy
        old_ax = ax
        old_ay = ay

        vx = vx + ax * sim_config.dt
        vy = vy + ay * sim_config.dt
        x = x + vx * sim_config.dt
        y = y + vy * sim_config.dt
        t = t + sim_config.dt

        if y <= 0:
            # Interpolação linear para estimar melhor o instante de impacto no solo.
            denominator = old_y - y
            fraction = old_y / denominator if denominator != 0 else 1.0
            impact_t = old_t + fraction * sim_config.dt
            impact_x = old_x + fraction * (x - old_x)
            impact_vx = old_vx + fraction * (vx - old_vx)
            impact_vy = old_vy + fraction * (vy - old_vy)
            impact_speed = math.hypot(impact_vx, impact_vy)

            data.append(
                {
                    "t": impact_t,
                    "x": impact_x,
                    "y": 0.0,
                    "vx": impact_vx,
                    "vy": impact_vy,
                    "speed": impact_speed,
                    "ax": old_ax,
                    "ay": old_ay,
                    "rho": get_air_density(0.0, atmosphere_config),
                    "wind_x": wind_x,
                    "wind_y": wind_y,
                    "relative_speed": relative_speed,
                    "drag_x": drag_x,
                    "drag_y": drag_y,
                }
            )

    return data
