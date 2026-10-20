"""
Calendario del mercado: periodos de cada día de entrega y festivos nacionales.

Todas las fechas internas van en UTC; Europe/Madrid solo se usa para saber
dónde empieza y acaba cada día de entrega y qué hora "de reloj" es cada periodo.
"""

from datetime import date, datetime, timedelta
from functools import lru_cache
from zoneinfo import ZoneInfo

import holidays
import pandas as pd

MADRID = ZoneInfo("Europe/Madrid")
BRUSELAS = ZoneInfo("Europe/Brussels")

# Cutoff de la plataforma: 12:45 hora de Bruselas del día anterior a la entrega
HORA_CUTOFF = (12, 45)


def periodos_utc(entrega: date, minutos: int = 60) -> pd.DatetimeIndex:
    """Inicios (en UTC) de todos los periodos de un día de entrega.

    Salen 24 horas un día normal, 23 el día del cambio de hora de marzo
    y 25 el de octubre, porque se cuenta en tiempo real (UTC) de medianoche a medianoche.
    """
    inicio = pd.Timestamp(entrega).tz_localize(MADRID).tz_convert("UTC")
    fin = pd.Timestamp(entrega + timedelta(days=1)).tz_localize(MADRID).tz_convert("UTC")
    return pd.date_range(inicio, fin, freq=f"{minutos}min", inclusive="left")


def hora_local(instante_utc: pd.Timestamp) -> int:
    """Hora de reloj en Madrid (0-23) de un instante UTC."""
    return instante_utc.tz_convert(MADRID).hour


def cutoff(entrega: date) -> datetime:
    """Momento límite de envío para un día de entrega (con zona horaria)."""
    dia_anterior = entrega - timedelta(days=1)
    return datetime(dia_anterior.year, dia_anterior.month, dia_anterior.day,
                    *HORA_CUTOFF, tzinfo=BRUSELAS)


def proxima_entrega_abierta(ahora: datetime | None = None) -> date:
    """Primer día de entrega cuyo cutoff todavía no ha pasado."""
    ahora = ahora or datetime.now(tz=BRUSELAS)
    entrega = ahora.astimezone(MADRID).date() + timedelta(days=1)
    while cutoff(entrega) <= ahora:
        entrega += timedelta(days=1)
    return entrega


@lru_cache(maxsize=None)
def _festivos(anio: int) -> holidays.HolidayBase:
    # Sin subdivisión: solo los festivos nacionales.
    # Ojo: la librería no incluye los nacionales que caen en domingo (p. ej. 1-nov-2026),
    # pero para el benchmark da igual: un domingo festivo y uno normal usan la misma referencia.
    return holidays.country_holidays("ES", years=anio)


def es_festivo(dia: date) -> bool:
    return dia in _festivos(dia.year)
