"""Modelos atmosféricos usados no simulador."""

from __future__ import annotations

import math
from dataclasses import dataclass


AIR_GAS_CONSTANT = 287.05  # J/(kg*K), constante específica do ar seco


@dataclass(frozen=True)
class AtmosphereConfig:
    """Configuração atmosférica da simulação.

    Attributes:
        model: Modelo de densidade: constant, exponential ou ideal_gas.
        rho0: Densidade de referência em kg/m³.
        pressure: Pressão em Pa.
        temperature: Temperatura em K.
        scale_height: Altura de escala atmosférica em m.
    """

    model: str = "constant"
    rho0: float = 1.225
    pressure: float = 101_325.0
    temperature: float = 288.15
    scale_height: float = 8_500.0


def density_constant(rho0: float = 1.225) -> float:
    """Retorna uma densidade constante."""
    if rho0 < 0:
        raise ValueError("A densidade não pode ser negativa.")
    return rho0


def density_exponential(altitude: float, rho0: float = 1.225, scale_height: float = 8_500.0) -> float:
    """Densidade com decaimento exponencial em função da altitude.

    rho(y) = rho0 * exp(-y / H)
    """
    if rho0 < 0:
        raise ValueError("A densidade de referência não pode ser negativa.")
    if scale_height <= 0:
        raise ValueError("A altura de escala deve ser maior que zero.")

    altitude = max(0.0, altitude)
    return rho0 * math.exp(-altitude / scale_height)


def density_ideal_gas(pressure: float = 101_325.0, temperature: float = 288.15) -> float:
    """Calcula a densidade do ar seco pela equação dos gases ideais.

    rho = P / (R * T)
    """
    if pressure < 0:
        raise ValueError("A pressão não pode ser negativa.")
    if temperature <= 0:
        raise ValueError("A temperatura deve estar em kelvin e ser maior que zero.")

    return pressure / (AIR_GAS_CONSTANT * temperature)


def get_air_density(altitude: float, config: AtmosphereConfig) -> float:
    """Retorna a densidade do ar de acordo com o modelo escolhido."""
    model = config.model.lower().strip()

    if model == "constant":
        return density_constant(config.rho0)

    if model == "exponential":
        return density_exponential(
            altitude=altitude,
            rho0=config.rho0,
            scale_height=config.scale_height,
        )

    if model == "ideal_gas":
        return density_ideal_gas(
            pressure=config.pressure,
            temperature=config.temperature,
        )

    raise ValueError(f"Modelo atmosférico desconhecido: {config.model}")
