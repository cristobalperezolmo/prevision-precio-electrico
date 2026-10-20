"""Comprobaciones antes de enviar: lo que la plataforma rechazaría con un 422."""

from datetime import date

import pandas as pd

from prevision.calendario import periodos_utc

PRECIO_MIN, PRECIO_MAX = -1000, 4000


def preparar(entrega: date, precios: pd.Series) -> pd.Series:
    """Devuelve la previsión redondeada a 2 decimales o lanza ValueError si algo no cuadra."""
    esperados = periodos_utc(entrega)
    if len(precios) != len(esperados) or not precios.index.equals(esperados):
        raise ValueError(f"{entrega}: se esperaban {len(esperados)} periodos, hay {len(precios)}")
    if precios.isna().any():
        raise ValueError(f"{entrega}: hay valores vacíos (NaN)")
    precios = precios.round(2)
    fuera = precios[(precios < PRECIO_MIN) | (precios > PRECIO_MAX)]
    if not fuera.empty:
        raise ValueError(f"{entrega}: precios fuera de [{PRECIO_MIN}, {PRECIO_MAX}]: {fuera.to_dict()}")
    return precios
