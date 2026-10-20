"""
Descarga y guarda en data/ (parquet) el histórico que usa el modelo:

- precios.parquet: precio day-ahead horario de España (OMIE), índice UTC
- meteo*.parquet:  previsiones meteorológicas tal y como se conocían ~2 días antes (Open-Meteo, Previous Runs API),
                   un fichero por modelo meteorológico (ECMWF, GFS, ICON)
- gas.parquet:     cierre diario del gas TTF (futuro mes siguiente, Yahoo Finance)
- redata.parquet:  generación real por tecnología y demanda real, diarias (REData de Red Eléctrica, sin clave)

Uso: python -m prevision.historico [--desde 2023-01-01]
Si el parquet ya existe, solo descarga lo que falta.
"""

import argparse
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import requests

from prevision import omie

CARPETA = Path(__file__).resolve().parent.parent / "data"
DESDE = date(2023, 1, 1)

# --- Meteorología ---
# Puntos representativos: zonas eólicas, solares y grandes centros de consumo.
PUNTOS = {
    "zaragoza":   (41.65, -0.88),   # eólica valle del Ebro
    "burgos":     (42.34, -3.70),   # eólica Castilla y León
    "lugo":       (43.01, -7.56),   # eólica Galicia
    "albacete":   (38.99, -1.86),   # eólica y solar La Mancha
    "cadiz":      (36.01, -5.60),   # eólica Estrecho
    "badajoz":    (38.88, -6.97),   # solar Extremadura
    "sevilla":    (37.39, -5.98),   # solar y demanda
    "ciudadreal": (38.99, -3.93),   # solar
    "murcia":     (37.98, -1.13),   # solar
    "madrid":     (40.42, -3.70),   # demanda
    "barcelona":  (41.39, 2.17),    # demanda
    "valencia":   (39.47, -0.38),   # demanda
}
VARIABLES = ["wind_speed_100m", "shortwave_radiation", "temperature_2m", "cloud_cover"]
# Antelación: previous_day2 = lo que el modelo meteorológico preveía 48 h antes.
# A las ~10 UTC del día D eso está disponible para TODAS las horas de D+1, así que no hay fuga.
ANTELACION = "previous_day2"
MODELO_METEO = "ecmwf_ifs025"
# Modelo meteorológico -> fichero. Promediar varios modelos reduce el error de la previsión del tiempo.
MODELOS_METEO = {"ecmwf_ifs025": "meteo.parquet", "gfs_seamless": "meteo_gfs.parquet",
                 "icon_seamless": "meteo_icon.parquet"}
URL_METEO = "https://previous-runs-api.open-meteo.com/v1/forecast"


def _get_con_reintentos(url, params, intentos=5, espera=65):
    """GET que espera y reintenta si la API limita las peticiones (429) o falla el servidor."""
    for i in range(intentos):
        try:
            r = requests.get(url, params=params, timeout=120)
            if r.status_code != 429 and r.status_code < 500:
                r.raise_for_status()
                return r
            motivo = r.status_code
        except (requests.ConnectionError, requests.Timeout) as e:
            motivo = type(e).__name__
        if i < intentos - 1:
            print(f"AVISO: {url.split('/')[2]} respondió {motivo}; reintento en {espera} s")
            time.sleep(espera)
    raise RuntimeError(f"{url} sigue fallando tras {intentos} intentos ({motivo})")


def descargar_meteo(desde: date, hasta: date, modelo: str = MODELO_METEO) -> pd.DataFrame:
    """Previsiones horarias (índice UTC), una columna por variable y punto: 'wind_speed_100m__zaragoza'."""
    nombres = list(PUNTOS)
    r = _get_con_reintentos(URL_METEO, params={
        "latitude": ",".join(str(PUNTOS[n][0]) for n in nombres),
        "longitude": ",".join(str(PUNTOS[n][1]) for n in nombres),
        "hourly": ",".join(f"{v}_{ANTELACION}" for v in VARIABLES),
        "start_date": desde.isoformat(), "end_date": hasta.isoformat(),
        "timezone": "GMT", "models": modelo,
    })
    columnas = {}
    for nombre, bloque in zip(nombres, r.json()):
        horas = pd.to_datetime(bloque["hourly"]["time"]).tz_localize("UTC")
        for v in VARIABLES:
            columnas[f"{v}__{nombre}"] = pd.Series(bloque["hourly"][f"{v}_{ANTELACION}"], index=horas, dtype=float)
    return pd.DataFrame(columnas)


# --- REData (Red Eléctrica) ---
URL_REDATA = "https://apidatos.ree.es/es/datos/{endpoint}"
TECNOLOGIAS = {"Nuclear": "nuclear", "Hidráulica": "hidraulica", "Eólica": "eolica",
               "Solar fotovoltaica": "solar", "Ciclo combinado": "ciclo", "Cogeneración": "cogeneracion",
               "Generación total": "generacion_total", "Demanda": "demanda"}


def descargar_redata(desde: date, hasta: date) -> pd.DataFrame:
    """Generación por tecnología y demanda real del sistema peninsular, en MWh por día."""
    columnas = {}
    a = desde
    while a <= hasta:
        b = min(date(a.year, 12, 31), hasta)  # la API admite como mucho un año por petición
        for endpoint in ("generacion/estructura-generacion", "demanda/evolucion"):
            r = requests.get(URL_REDATA.format(endpoint=endpoint), timeout=60, params={
                "start_date": f"{a}T00:00", "end_date": f"{b}T23:59", "time_trunc": "day",
                "geo_trunc": "electric_system", "geo_limit": "peninsular", "geo_ids": 8741})
            r.raise_for_status()
            for bloque in r.json()["included"]:
                nombre = TECNOLOGIAS.get(bloque["attributes"]["title"])
                if nombre:
                    for v in bloque["attributes"]["values"]:
                        columnas.setdefault(nombre, {})[date.fromisoformat(v["datetime"][:10])] = v["value"]
        a = b + timedelta(days=1)
    return pd.DataFrame(columnas).sort_index()


def descargar_gas(desde: date) -> pd.Series:
    import yfinance as yf
    datos = yf.download("TTF=F", start=desde.isoformat(), progress=False, auto_adjust=False)
    serie = datos["Close"].squeeze().dropna()
    serie.index = pd.to_datetime(serie.index).date
    serie.name = "ttf"
    return serie


def descargar_precios(dias: list[date]) -> pd.Series:
    with ThreadPoolExecutor(max_workers=8) as pool:
        trozos = list(pool.map(_precio_o_nada, dias))
    trozos = [t for t in trozos if t is not None]
    return pd.concat(trozos).sort_index().rename("precio") if trozos else pd.Series(dtype=float)


def _precio_o_nada(dia):
    try:
        return omie.descargar_dia(dia)
    except RuntimeError as e:
        print(f"AVISO: {e}")
        return None


def actualizar(desde: date = DESDE, hasta: date | None = None):
    """Actualiza los parquet hasta 'hasta' (por defecto: mañana, si OMIE ya lo publicó)."""
    CARPETA.mkdir(exist_ok=True)
    hasta = hasta or date.today() + timedelta(days=1)

    # Precios
    ruta = CARPETA / "precios.parquet"
    viejo = pd.read_parquet(ruta)["precio"] if ruta.exists() else pd.Series(dtype=float)
    hechos = set(viejo.index.tz_convert("Europe/Madrid").date) if len(viejo) else set()
    faltan = [desde + timedelta(days=i) for i in range((hasta - desde).days + 1)]
    faltan = [d for d in faltan if d not in hechos]
    if faltan:
        print(f"Precios: descargando {len(faltan)} días de OMIE...")
        nuevo = descargar_precios(faltan)
        precios = pd.concat([viejo, nuevo]).sort_index()
        precios = precios[~precios.index.duplicated(keep="last")]
        precios.to_frame("precio").to_parquet(ruta)
        print(f"Precios: {len(precios)} horas, de {precios.index.min()} a {precios.index.max()}")

    # Meteorología: un fichero por modelo, pedido por trozos de ~3 meses
    for modelo, fichero in MODELOS_METEO.items():
        ruta = CARPETA / fichero
        viejo = pd.read_parquet(ruta) if ruta.exists() else pd.DataFrame()
        # se vuelve a pedir desde unos días antes del final: las horas que eran futuras
        # al descargarlas se sustituyen por la previsión archivada definitiva
        inicio = max(desde, viejo.dropna(how="all").index.max().date() - timedelta(days=3)) if len(viejo) else desde
        fin_meteo = hasta + timedelta(days=1)
        trozos = []
        a = inicio
        while a <= fin_meteo:
            b = min(a + timedelta(days=90), fin_meteo)
            trozos.append(descargar_meteo(a, b, modelo))
            a = b + timedelta(days=1)
        meteo = pd.concat([viejo] + trozos).sort_index()
        meteo = meteo[~meteo.index.duplicated(keep="last")]
        meteo.to_parquet(ruta)
        print(f"Meteo {modelo}: {meteo.shape}, huecos: {meteo.isna().mean().mean():.1%}")

    # REData: se vuelve a pedir el último mes (los datos recientes se revisan)
    ruta = CARPETA / "redata.parquet"
    viejo = pd.read_parquet(ruta) if ruta.exists() else pd.DataFrame()
    inicio = max(desde, viejo.index.max() - timedelta(days=30)) if len(viejo) else desde
    redata = pd.concat([viejo, descargar_redata(inicio, min(hasta, date.today()))]).sort_index()
    redata = redata[~redata.index.duplicated(keep="last")]
    redata.to_parquet(ruta)
    print(f"REData: {redata.shape}, último día {redata.index.max()}")

    # Gas (pocas filas: se descarga entero cada vez)
    gas = descargar_gas(desde)
    gas.to_frame().to_parquet(CARPETA / "gas.parquet")
    print(f"Gas: {len(gas)} días, último {gas.index.max()} = {gas.iloc[-1]:.2f}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--desde", type=date.fromisoformat, default=DESDE)
    actualizar(p.parse_args().desde)
