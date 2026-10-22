"""
Cliente mínimo de la API de e·sios (Red Eléctrica). Token en ESIOS_API_KEY.

Previsiones "diarias D+1" de REE para la península: se publican hacia las 09:00 UTC del día anterior
(comprobado el 22-oct-2026: 08:53-08:59 UTC), antes de nuestro cutoff (10:45 UTC en verano).
"""

import os
from datetime import date, timedelta

import pandas as pd
import requests

URL = "https://api.esios.ree.es/indicators/{id}"
INDICADORES = {1777: "esios_eolica", 1779: "esios_fv", 1775: "esios_demanda"}


def _token() -> str:
    tok = os.getenv("ESIOS_API_KEY", "").strip()
    if not tok:
        from dotenv import dotenv_values
        tok = (dotenv_values(".env").get("ESIOS_API_KEY") or "").strip()
    if not tok:
        raise RuntimeError("Falta ESIOS_API_KEY")
    return tok


def indicador(id_: int, desde: date, hasta: date) -> pd.Series:
    """Valores horarios (índice UTC) de un indicador para la península."""
    h = {"x-api-key": _token(), "Accept": "application/json; application/vnd.esios-api-v1+json"}
    try:
        r = requests.get(URL.format(id=id_), headers=h, timeout=90,
                         params={"start_date": f"{desde}T00:00", "end_date": f"{hasta}T23:59"})
    except requests.RequestException as e:
        raise RuntimeError(f"e·sios no responde ({type(e).__name__})") from None
    if r.status_code != 200:
        raise RuntimeError(f"e·sios respondió {r.status_code}")
    v = pd.DataFrame(r.json()["indicator"]["values"])
    if v.empty:
        return pd.Series(dtype=float)
    if "geo_id" in v:
        v = v[v["geo_id"] == 8741]  # península
    s = pd.Series(v["value"].values, index=pd.to_datetime(v["datetime_utc"], utc=True), dtype=float).sort_index()
    return s[~s.index.duplicated(keep="last")].resample("1h").mean()


def previsiones(desde: date, hasta: date) -> pd.DataFrame:
    """Eólica, fotovoltaica y demanda previstas D+1 (MW), por trozos de ~3 meses."""
    columnas = {}
    for id_, nombre in INDICADORES.items():
        trozos, a = [], desde
        while a <= hasta:
            b = min(a + timedelta(days=90), hasta)
            trozos.append(indicador(id_, a, b))
            a = b + timedelta(days=1)
        s = pd.concat(trozos).sort_index()
        columnas[nombre] = s[~s.index.duplicated(keep="last")]
    return pd.DataFrame(columnas)


def actualizar(carpeta) -> pd.DataFrame:
    """Histórico en data/esios.parquet: se vuelven a pedir los últimos 40 días (y lo nuevo)."""
    ruta = carpeta / "esios.parquet"
    viejo = pd.read_parquet(ruta) if ruta.exists() else pd.DataFrame()
    inicio = viejo.index.max().date() - timedelta(days=40) if len(viejo) else date(2022, 12, 31)
    nuevo = pd.concat([viejo, previsiones(inicio, date.today() + timedelta(days=1))]).sort_index()
    nuevo = nuevo[~nuevo.index.duplicated(keep="last")]
    nuevo.to_parquet(ruta)
    print(f"e·sios: {nuevo.shape}, último {nuevo.dropna().index.max()}")
    return nuevo
