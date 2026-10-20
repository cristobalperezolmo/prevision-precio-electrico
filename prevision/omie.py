"""
Precios day-ahead de España desde los ficheros públicos de OMIE (marginalpdbc).

No necesita clave. Cada fichero trae un día de entrega con líneas
"año;mes;día;periodo;precio_PT;precio_ES;". Desde el 1-oct-2025 los periodos son
cuartohorarios (96 / 92 / 100 al día); antes eran horarios (24 / 23 / 25).

El precio horario se calcula como la media de los cuatro cuartos de hora.
PENDIENTE: confirmar que coincide con lo que muestra https://forecastscore.eu/market-data.
"""

from datetime import date

import pandas as pd
import requests

from prevision.calendario import periodos_utc

URL = ("https://www.omie.es/es/file-download?parents%5B0%5D=marginalpdbc"
       "&filename=marginalpdbc_{fecha:%Y%m%d}.{version}")


def leer_fichero(texto: str, entrega: date) -> pd.Series:
    """Convierte el texto de un fichero marginalpdbc en precios horarios indexados en UTC."""
    precios = {}
    for linea in texto.splitlines():
        partes = linea.strip().split(";")
        if len(partes) < 6 or not partes[0].isdigit():
            continue  # cabecera "MARGINALPDBC;" y la línea final "*"
        anio, mes, dia, periodo = (int(x) for x in partes[:4])
        if date(anio, mes, dia) != entrega:
            raise ValueError(f"El fichero de OMIE es del {anio}-{mes}-{dia}, no del {entrega}")
        precios[periodo] = float(partes[5])  # columna 6 = España

    # Deducir la resolución por el número de periodos
    n = len(precios)
    if n == len(periodos_utc(entrega, 15)):
        minutos = 15
    elif n == len(periodos_utc(entrega, 60)):
        minutos = 60
    else:
        raise ValueError(f"OMIE {entrega}: {n} periodos, no cuadra con el calendario")

    inicios = periodos_utc(entrega, minutos)
    serie = pd.Series([precios[p] for p in range(1, n + 1)], index=inicios)
    # Las horas de Madrid son horas enteras en UTC, así que agrupar por hora UTC es correcto
    return serie.resample("1h").mean()


def descargar_dia(entrega: date, reintentos: int = 3) -> pd.Series:
    """Descarga los precios horarios de España de un día de entrega.

    Normalmente el fichero es la versión .1; algunos días solo existe una versión
    corregida (.2, .3...), y entonces OMIE devuelve 404 o una página HTML para la .1.
    """
    ultimo_error = None
    for _ in range(reintentos):
        for version in (1, 2, 3, 4):
            try:
                r = requests.get(URL.format(fecha=entrega, version=version), timeout=30)
                r.raise_for_status()
                if not r.text.startswith("MARGINALPDBC"):
                    raise ValueError(f"versión .{version} no disponible")
                return leer_fichero(r.text, entrega)
            except requests.HTTPError as e:
                ultimo_error = e
                if e.response is not None and e.response.status_code == 404:
                    continue  # probar la siguiente versión
                break
            except (requests.RequestException, ValueError) as e:
                ultimo_error = e
                if "no disponible" in str(e):
                    continue
                break
    raise RuntimeError(f"No se pudieron descargar los precios de OMIE del {entrega}: {ultimo_error}")
