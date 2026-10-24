"""
Futuros de electricidad de OMIP (España, carga base): precios de referencia de cada sesión.

Cada día hábil OMIP publica, al cierre de la sesión, el precio de referencia de los contratos diarios
(FTB D, uno por día de entrega), de fin de semana (FTB WE) y semanales (FTB Wk). Es la expectativa del
mercado sobre el precio medio de ese periodo.

OJO con las fugas: el último día de negociación de un contrato diario es el día anterior a la entrega, y
su referencia de ese día es el precio del mercado diario ya casado (comprobado: FTB D Fr23Oct-26 en la
sesión del 22-oct = 110,09 = media real del 23). Para predecir E el día D solo se pueden usar
sesiones de D-1 o anteriores (cerradas la tarde anterior).
"""

import re
import html
import time
from datetime import date, timedelta

import pandas as pd
import requests

URL = "https://www.omip.pt/en/dados-mercado"
MESES = {m: i for i, m in enumerate(["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct",
                                     "Nov", "Dec"], 1)}


def _numero(texto: str):
    try:
        return float(texto.replace(",", ""))
    except ValueError:
        return None


def leer_sesion(texto: str, sesion: date) -> pd.DataFrame:
    """Filas (sesion, contrato, tipo, inicio, ref, ref_anterior) de la página de una sesión."""
    filas = []
    for r in re.findall(r"<tr[^>]*>(.*?)</tr>", texto, re.S):
        celdas = [" ".join(html.unescape(re.sub(r"<[^>]+>", " ", c)).split())
                  for c in re.findall(r"<t[dh][^>]*>(.*?)</t[dh]>", r, re.S)]
        if len(celdas) < 16 or "FTB" not in celdas[0]:
            continue
        nombre = re.search(r"(FTB .*)$", celdas[0]).group(1)
        m = re.match(r"FTB (D|WE) (?:[A-Z][a-z])?(\d{2})([A-Z][a-z]{2})-(\d{2})$", nombre)
        w = re.match(r"FTB Wk(\d{2})-(\d{2})$", nombre)
        if m:
            tipo, inicio = m.group(1), date(2000 + int(m.group(4)), MESES[m.group(3)], int(m.group(2)))
        elif w:
            tipo, inicio = "Wk", date.fromisocalendar(2000 + int(w.group(2)), int(w.group(1)), 1)
        else:
            continue
        filas.append({"sesion": sesion, "contrato": nombre, "tipo": tipo, "inicio": inicio,
                      "ref": _numero(celdas[14]), "ref_anterior": _numero(celdas[15])})
    return pd.DataFrame(filas)


def descargar_sesion(sesion: date, intentos: int = 4, espera_bloqueo: float = 300) -> pd.DataFrame:
    for i in range(intentos):
        try:
            r = requests.get(URL, timeout=60, headers={"User-Agent": "Mozilla/5.0"},
                             params={"date": sesion.isoformat(), "product": "EL", "zone": "ES", "instrument": "FTB"})
            if r.status_code == 200:
                t = leer_sesion(r.text, sesion)
                if len(t) or i == intentos - 1:
                    return t
                # Página sin tabla: OMIP limita si se le piden muchas seguidas; se espera más
                time.sleep(espera_bloqueo)
                continue
        except requests.RequestException:
            pass
        time.sleep(10 * (i + 1))
    raise RuntimeError(f"OMIP no responde para la sesión {sesion}")


def descargar(desde: date, hasta: date, pausa: float = 1.0) -> pd.DataFrame:
    trozos, d = [], desde
    while d <= hasta:
        if d.weekday() < 5:
            t = descargar_sesion(d)
            if len(t):
                trozos.append(t)
            time.sleep(pausa)
        d += timedelta(days=1)
    return pd.concat(trozos, ignore_index=True) if trozos else pd.DataFrame()


def actualizar(carpeta) -> pd.DataFrame:
    """Histórico en data/omip.parquet: se vuelven a pedir las últimas 2 semanas."""
    ruta = carpeta / "omip.parquet"
    viejo = pd.read_parquet(ruta) if ruta.exists() else pd.DataFrame()
    inicio = viejo["sesion"].max() - timedelta(days=14) if len(viejo) else date(2022, 12, 1)
    nuevo = descargar(inicio, date.today())
    if len(viejo):
        viejo = viejo[viejo["sesion"] < inicio]
    tabla = pd.concat([viejo, nuevo], ignore_index=True).drop_duplicates(["sesion", "contrato"], keep="last")
    tabla.to_parquet(ruta)
    print(f"OMIP: {len(tabla)} filas, última sesión {tabla['sesion'].max()}")
    return tabla
