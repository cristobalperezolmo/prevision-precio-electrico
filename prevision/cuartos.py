"""
Previsión a 15 minutos (PT15M) a partir de la previsión horaria.

Cada cuarto de hora = previsión de su hora + forma típica dentro de esa hora, que es la media
de los últimos 7 días de (precio del cuarto - media de su hora). En el backtest (oct-2025 a oct-2026)
esto da rMAE 0,588 frente a 0,625 de repetir el valor horario (experimentos/cuartos.py).
"""

from datetime import date, timedelta

import numpy as np
import pandas as pd

from prevision import omie
from prevision.benchmark import dia_referencia
from prevision.calendario import MADRID, periodos_utc


def _tabla(q: pd.Series) -> pd.DataFrame:
    """[día local x cuarto 0-95]. Las dos 02:xx de octubre se promedian."""
    local = q.index.tz_convert(MADRID)
    df = pd.DataFrame({"dia": local.date, "c": local.hour * 4 + local.minute // 15, "p": q.values})
    return df.pivot_table(index="dia", columns="c", values="p", aggfunc="mean")


def descargar_cuartos(dias: list[date]) -> pd.Series:
    trozos = []
    for d in dias:
        try:
            trozos.append(omie.descargar_dia(d, horario=False))
        except RuntimeError:
            pass
    serie = pd.concat(trozos).sort_index() if trozos else pd.Series(dtype=float)
    # Antes de oct-2025 los ficheros eran horarios: solo sirven los de 15 minutos
    return serie[serie.index.to_series().diff().dt.total_seconds().fillna(900) == 900]


def forma_reciente(q: pd.Series, entrega: date, dias: int = 7) -> pd.Series:
    """Desviación media de cada cuarto respecto a la media de su hora en los días anteriores a la entrega."""
    T = _tabla(q)
    T = T[[d < entrega for d in T.index]].tail(dias)
    if T.empty:
        return pd.Series(0.0, index=range(96))
    desvio = T - T.T.groupby(T.columns // 4).transform("mean").T
    return desvio.mean().reindex(range(96)).fillna(0.0)


def a_cuartos(horaria: pd.Series, entrega: date, q_recientes: pd.Series | None) -> pd.Series:
    """Convierte la previsión horaria (índice UTC) en cuartos de hora del día de entrega."""
    inicios = periodos_utc(entrega, 15)
    base = horaria.reindex(inicios.floor("1h")).values
    forma = forma_reciente(q_recientes, entrega) if q_recientes is not None and len(q_recientes) else None
    if forma is None:
        return pd.Series(base, index=inicios)
    local = inicios.tz_convert(MADRID)
    c = local.hour * 4 + local.minute // 15
    return pd.Series(base + forma.reindex(c).values, index=inicios)


def benchmark_cuartos(entrega: date, q_ref: pd.Series) -> pd.Series:
    """Benchmark de la plataforma a 15 minutos: mismo cuarto de reloj del día de referencia."""
    ref = dia_referencia(entrega)
    T = _tabla(q_ref)
    if ref not in T.index:
        raise RuntimeError(f"Faltan los precios cuartohorarios del día de referencia {ref}")
    fila = T.loc[ref].reindex(range(96)).interpolate(limit_direction="both")  # 02:xx de marzo
    inicios = periodos_utc(entrega, 15)
    local = inicios.tz_convert(MADRID)
    return pd.Series(fila.reindex(local.hour * 4 + local.minute // 15).values, index=inicios)


def cuartos_recientes(entrega: date) -> pd.Series:
    """Precios cuartohorarios de los 14 días anteriores a la entrega (forma y benchmark festivo)."""
    D = entrega - timedelta(days=1)
    return descargar_cuartos([D - timedelta(days=i) for i in range(14)])


