"""Modelos simples de vento para a simulação."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class WindConfig:
    """Configuração do vento.

    O eixo x representa movimento horizontal.
    O eixo y representa movimento vertical, positivo para cima.
    """

    model: str = "constant"
    horizontal_speed: float = 0.0
    vertical_speed: float = 0.0
    shear_per_meter: float = 0.0


def get_wind_velocity(altitude: float, config: WindConfig) -> tuple[float, float]:
    """Retorna a velocidade do vento no ponto atual.

    Models:
        none: sem vento.
        constant: vento fixo.
        linear_shear: vento horizontal aumenta linearmente com a altitude.
    """
    model = config.model.lower().strip()
    altitude = max(0.0, altitude)

    if model == "none":
        return 0.0, 0.0

    if model == "constant":
        return config.horizontal_speed, config.vertical_speed

    if model == "linear_shear":
        horizontal = config.horizontal_speed + config.shear_per_meter * altitude
        return horizontal, config.vertical_speed

    raise ValueError(f"Modelo de vento desconhecido: {config.model}")
