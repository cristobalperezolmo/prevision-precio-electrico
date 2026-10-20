"""
Corrección del sesgo reciente.

Cada día se guarda en registro/predicciones.csv la previsión horaria SIN corregir. Al predecir E,
ya se conocen los precios reales hasta D = E-1, así que se calcula el error pasado de nuestras
previsiones a cada hora local y se resta una parte:

    corregida(E, h) = previsión(E, h) - K * media_exponencial(errores a la hora h, vida media VIDA_MEDIA días)

En el backtest (ene-2025 a oct-2026) pasa de rMAE 0,550/0,575 a 0,539/0,555 (experimentos/NOTAS.md).
"""

from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from prevision.calendario import MADRID

REGISTRO = Path(__file__).resolve().parent.parent / "registro" / "predicciones.csv"
K = 0.7
VIDA_MEDIA = 14
MIN_DIAS = 7


def leer_registro(ruta: Path = REGISTRO) -> pd.Series:
    if not ruta.exists():
        return pd.Series(dtype=float)
    df = pd.read_csv(ruta, parse_dates=["instante"])
    return pd.Series(df["prevision"].values, index=pd.DatetimeIndex(df["instante"], tz="UTC")
                     if df["instante"].dt.tz is None else pd.DatetimeIndex(df["instante"]))


def guardar_en_registro(prevision: pd.Series, ruta: Path = REGISTRO):
    """Añade (o sustituye) la previsión sin corregir de un día al registro."""
    viejo = leer_registro(ruta)
    nuevo = pd.concat([viejo[~viejo.index.isin(prevision.index)], prevision]).sort_index()
    ruta.parent.mkdir(exist_ok=True)
    pd.DataFrame({"instante": nuevo.index.strftime("%Y-%m-%dT%H:%M:%SZ"),
                  "prevision": nuevo.round(3).values}).to_csv(ruta, index=False)


def correccion(entrega: date, registro: pd.Series, precios: pd.Series) -> pd.Series | None:
    """Corrección (a restar) por hora local 0-23, o None si no hay suficientes días con error conocido."""
    comunes = registro.index.intersection(precios.index)
    if len(comunes) == 0:
        return None
    err = registro.loc[comunes] - precios.loc[comunes]
    local = comunes.tz_convert(MADRID)
    tabla = pd.DataFrame({"dia": local.date, "h": local.hour, "e": err.values}) \
        .pivot_table(index="dia", columns="h", values="e", aggfunc="mean")
    tabla = tabla[[d < entrega for d in tabla.index]]
    if len(tabla) < MIN_DIAS:
        return None
    # La media exponencial cuenta días naturales: se rellenan los días sin registro como huecos
    tabla = tabla.reindex(pd.date_range(min(tabla.index), max(tabla.index)).date)
    media = tabla.ewm(halflife=VIDA_MEDIA, min_periods=MIN_DIAS, ignore_na=True).mean().iloc[-1]
    return (K * media).reindex(range(24)).fillna(0.0)


def aplicar(prevision: pd.Series, entrega: date, registro: pd.Series, precios: pd.Series):
    """Devuelve (previsión corregida, texto informativo)."""
    c = correccion(entrega, registro, precios)
    if c is None:
        return prevision, "sin corrección de sesgo (pocos días en el registro)"
    horas = prevision.index.tz_convert(MADRID).hour
    corregida = prevision - c.reindex(horas).values
    return corregida, f"corrección de sesgo media {np.mean(c.values):+.2f} €/MWh"
