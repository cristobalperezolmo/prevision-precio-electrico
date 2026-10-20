"""
Fuentes adicionales (en pruebas):

- mibgas.parquet:  índice MIBGAS-ES (gas español) por día de entrega. El de D se fija con la
                   negociación de D-1, así que a las 10 UTC de D ya se conoce.
- co2.parquet:     cierre diario del ETC de derechos de emisión (SparkChange, CO2.L). Probado y
                   descartado (no aporta); se deja la función por si se quiere volver a probar.
- balance.parquet: saldo de intercambios internacionales y bombeo diarios (REData), en MWh.
- meteo2_*.parquet: más puntos meteorológicos para índices eólico/solar ponderados.

Uso: python -m prevision.datos_extra
"""

import io
from datetime import date, timedelta

import pandas as pd
import requests

from prevision.historico import (CARPETA, DESDE, MODELOS_METEO, URL_METEO, URL_REDATA,
                                 _get_con_reintentos)

# --- MIBGAS ---
URL_MIBGAS = "https://www.mibgas.es/en/file-access/MIBGAS_Data_{anio}.xlsx?path=AGNO_{anio}/XLS"


def descargar_mibgas(anios) -> pd.Series:
    trozos = []
    for anio in anios:
        r = requests.get(URL_MIBGAS.format(anio=anio), timeout=120)
        r.raise_for_status()
        df = pd.read_excel(io.BytesIO(r.content), sheet_name="MIBGAS Indexes")
        col = [c for c in df.columns if c.startswith("MIBGAS-ES Index")][0]
        s = pd.Series(df[col].values, index=pd.to_datetime(df["Delivery day"]).dt.date, name="mibgas")
        trozos.append(s.dropna())
    s = pd.concat(trozos).sort_index()
    return s[~s.index.duplicated(keep="last")]


def descargar_co2(desde: date) -> pd.Series:
    import yfinance as yf
    datos = yf.download("CO2.L", start=desde.isoformat(), progress=False, auto_adjust=False)
    s = datos["Close"].squeeze().dropna()
    s.index = pd.to_datetime(s.index).date
    s.name = "co2"
    return s


# --- REData: balance (intercambios y bombeo) ---
BALANCE = {"Saldo I. internacionales": "saldo_internacional", "Consumo bombeo": "consumo_bombeo",
           "Turbinación bombeo": "turbinacion_bombeo", "Generación renovable": "renovable"}


def descargar_balance(desde: date, hasta: date) -> pd.DataFrame:
    columnas = {}
    a = desde
    while a <= hasta:
        b = min(date(a.year, 12, 31), hasta)
        r = requests.get(URL_REDATA.format(endpoint="balance/balance-electrico"), timeout=60, params={
            "start_date": f"{a}T00:00", "end_date": f"{b}T23:59", "time_trunc": "day",
            "geo_trunc": "electric_system", "geo_limit": "peninsular", "geo_ids": 8741})
        r.raise_for_status()
        for grupo in r.json()["included"]:
            for c in grupo["attributes"].get("content", []):
                nombre = BALANCE.get(c["attributes"]["title"])
                if nombre:
                    for v in c["attributes"]["values"]:
                        columnas.setdefault(nombre, {})[date.fromisoformat(v["datetime"][:10])] = v["value"]
        a = b + timedelta(days=1)
    return pd.DataFrame(columnas).sort_index()


# --- Más puntos meteorológicos ---
# Zonas con mucha potencia eólica / solar instalada (aprox.), con un peso relativo.
PUNTOS_EOLICOS = {
    "soria": (41.76, -2.46, 1.0), "palencia": (42.4, -4.5, 1.0), "leon": (42.6, -5.6, 0.7),
    "navarra": (42.6, -1.7, 0.8), "teruel": (40.7, -0.9, 1.0), "huesca": (41.9, -0.3, 0.6),
    "cuenca": (39.9, -2.0, 0.8), "albacete2": (38.8, -1.4, 1.0), "coruna": (43.2, -8.6, 1.0),
    "tarifa": (36.1, -5.7, 0.6), "malaga": (36.9, -4.6, 0.5), "tarragona": (41.1, 0.6, 0.5),
}
PUNTOS_SOLARES = {
    "badajoz2": (38.7, -6.4, 1.0), "caceres": (39.5, -6.2, 0.8), "ciudadreal2": (38.7, -3.8, 1.0),
    "toledo": (39.8, -4.3, 0.7), "cordoba": (37.8, -4.9, 0.8), "jaen": (37.9, -3.6, 0.6),
    "granada": (37.3, -3.3, 0.5), "murcia2": (38.0, -1.3, 0.8), "alicante": (38.4, -0.8, 0.4),
    "zaragoza2": (41.5, -0.9, 0.8), "valladolid": (41.6, -4.7, 0.5), "huelva": (37.5, -6.9, 0.5),
}
VARS2 = ["wind_speed_100m", "shortwave_radiation"]


def descargar_meteo2(desde: date, hasta: date, modelo: str, rapido: bool = False) -> pd.DataFrame:
    puntos = {**{k: v for k, v in PUNTOS_EOLICOS.items()}, **PUNTOS_SOLARES}
    nombres = list(puntos)
    r = _get_con_reintentos(URL_METEO, params={
        "latitude": ",".join(str(puntos[n][0]) for n in nombres),
        "longitude": ",".join(str(puntos[n][1]) for n in nombres),
        "hourly": ",".join(f"{v}_previous_day2" for v in VARS2),
        "start_date": desde.isoformat(), "end_date": hasta.isoformat(),
        "timezone": "GMT", "models": modelo,
    }, **({"intentos": 3, "espera": 20, "timeout": 40} if rapido else {}))
    columnas = {}
    for nombre, bloque in zip(nombres, r.json()):
        horas = pd.to_datetime(bloque["hourly"]["time"]).tz_localize("UTC")
        var = "wind_speed_100m" if nombre in PUNTOS_EOLICOS else "shortwave_radiation"
        columnas[f"{var}__{nombre}"] = pd.Series(bloque["hourly"][f"{var}_previous_day2"], index=horas, dtype=float)
    return pd.DataFrame(columnas)


def actualizar(desde: date = date(2024, 3, 1)):
    hoy = date.today()
    s = descargar_mibgas(range(DESDE.year, hoy.year + 1))
    s.to_frame().to_parquet(CARPETA / "mibgas.parquet")
    print(f"MIBGAS: {len(s)} días, último {s.index.max()} = {s.iloc[-1]:.2f}")
    b = descargar_balance(DESDE, hoy)
    b.to_parquet(CARPETA / "balance.parquet")
    print(f"Balance: {b.shape}")
    for modelo in MODELOS_METEO:
        ruta = CARPETA / f"meteo2_{modelo}.parquet"
        viejo = pd.read_parquet(ruta) if ruta.exists() else pd.DataFrame()
        inicio = max(desde, viejo.dropna(how="all").index.max().date() - timedelta(days=3)) if len(viejo) else desde
        trozos, a, fin = [], inicio, hoy + timedelta(days=2)
        while a <= fin:
            b2 = min(a + timedelta(days=90), fin)
            trozos.append(descargar_meteo2(a, b2, modelo))
            a = b2 + timedelta(days=1)
        m = pd.concat([viejo] + trozos).sort_index()
        m = m[~m.index.duplicated(keep="last")]
        m.to_parquet(ruta)
        print(f"Meteo2 {modelo}: {m.shape}, huecos {m.isna().mean().mean():.1%}")


if __name__ == "__main__":
    actualizar()
