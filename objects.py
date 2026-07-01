"""Modelos de objetos para o simulador de queda livre."""

from dataclasses import dataclass


@dataclass(frozen=True)
class FallingObject:
    """Representa um corpo em queda.

    Attributes:
        name: Nome do objeto.
        mass: Massa em kg.
        area: Área frontal em m².
        drag_coefficient: Coeficiente de arrasto adimensional.
    """

    name: str
    mass: float
    area: float
    drag_coefficient: float


OBJECT_PRESETS: dict[str, FallingObject] = {
    "Esfera pequena": FallingObject(
        name="Esfera pequena",
        mass=0.145,
        area=0.0042,
        drag_coefficient=0.47,
    ),
    "Bola de futebol": FallingObject(
        name="Bola de futebol",
        mass=0.43,
        area=0.038,
        drag_coefficient=0.25,
    ),
    "Folha de papel": FallingObject(
        name="Folha de papel",
        mass=0.005,
        area=0.062,
        drag_coefficient=1.28,
    ),
    "Paraquedista fechado": FallingObject(
        name="Paraquedista fechado",
        mass=80.0,
        area=0.70,
        drag_coefficient=1.00,
    ),
    "Paraquedista com paraquedas": FallingObject(
        name="Paraquedista com paraquedas",
        mass=85.0,
        area=25.0,
        drag_coefficient=1.50,
    ),
}


def make_custom_object(
    name: str,
    mass: float,
    area: float,
    drag_coefficient: float,
) -> FallingObject:
    """Cria um objeto personalizado validando os parâmetros principais."""
    if mass <= 0:
        raise ValueError("A massa deve ser maior que zero.")
    if area <= 0:
        raise ValueError("A área frontal deve ser maior que zero.")
    if drag_coefficient < 0:
        raise ValueError("O coeficiente de arrasto não pode ser negativo.")

    return FallingObject(
        name=name,
        mass=mass,
        area=area,
        drag_coefficient=drag_coefficient,
    )
