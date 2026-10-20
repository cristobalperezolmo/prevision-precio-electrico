"""
Benchmark de Forecast Score (https://forecastscore.eu/docs/scoring):
- martes a viernes: precio del día anterior a la misma hora
- lunes, sábado y domingo: mismo día de la semana anterior
- festivo nacional: último domingo no festivo
- cambio de hora: las dos 02:00 de octubre usan la única 02:00 del día de referencia
"""

from datetime import date, timedelta

import pandas as pd

from prevision.calendario import es_festivo, hora_local, periodos_utc


def dia_referencia(entrega: date) -> date:
    if es_festivo(entrega):
        dia = entrega - timedelta(days=1)
        while dia.weekday() != 6 or es_festivo(dia):
            dia -= timedelta(days=1)
        return dia
    if entrega.weekday() in (1, 2, 3, 4):  # martes a viernes
        return entrega - timedelta(days=1)
    return entrega - timedelta(days=7)


def copiar_por_hora_local(entrega: date, precios_ref: pd.Series) -> pd.Series:
    """Asigna a cada periodo del día de entrega el precio de la misma hora de reloj del día de referencia.

    - Si la referencia tiene dos 02:00 (octubre), se usa la primera.
    - Si a la referencia le falta una hora (marzo), se rellena con la media de la hora anterior
      y la siguiente (la plataforma no puntúa el benchmark ahí, pero nosotros tenemos que enviar algo).
    """
    por_hora = {}
    for instante, precio in precios_ref.items():
        por_hora.setdefault(hora_local(instante), precio)

    valores = {}
    for instante in periodos_utc(entrega):
        h = hora_local(instante)
        if h in por_hora:
            valores[instante] = por_hora[h]
        else:
            valores[instante] = (por_hora[h - 1] + por_hora[h + 1]) / 2
    return pd.Series(valores)
