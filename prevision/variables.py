"""
Variables (features) del modelo. Una fila por periodo horario de cada día de entrega E.

Regla de oro (sin fugas): al predecir E a las ~10 UTC del día D = E-1 solo se conocen
- precios day-ahead hasta D incluido,
- previsiones meteorológicas emitidas como tarde ~48 h antes de cada hora (previous_day2),
- el cierre del gas del último día hábil ANTERIOR a D.
"""

from datetime import date, timedelta

import numpy as np
import pandas as pd

from prevision.benchmark import dia_referencia
from prevision.calendario import MADRID, es_festivo, periodos_utc
from prevision.historico import CARPETA, PUNTOS

EOLICOS = ["zaragoza", "burgos", "lugo", "albacete", "cadiz"]
SOLARES = ["badajoz", "sevilla", "ciudadreal", "murcia", "albacete"]
CONSUMO = ["madrid", "barcelona", "sevilla", "valencia"]


def cargar() -> dict:
    datos = {
        "precios": pd.read_parquet(CARPETA / "precios.parquet")["precio"],
        "meteo": pd.read_parquet(CARPETA / "meteo.parquet"),
        "gas": pd.read_parquet(CARPETA / "gas.parquet")["ttf"],
    }
    # Opcionales (v1.1): otros modelos meteorológicos y REData
    from prevision.historico import MODELOS_METEO
    datos["meteo_modelos"] = {m: pd.read_parquet(CARPETA / f) for m, f in MODELOS_METEO.items()
                              if (CARPETA / f).exists()}
    if (CARPETA / "redata.parquet").exists():
        datos["redata"] = pd.read_parquet(CARPETA / "redata.parquet")
    return datos


def tabla_diaria(precios: pd.Series) -> pd.DataFrame:
    """Precios como matriz [día local x hora local 0-23]. En octubre se promedian las dos 02:00."""
    local = precios.index.tz_convert(MADRID)
    df = pd.DataFrame({"dia": local.date, "hora": local.hour, "precio": precios.values})
    return df.pivot_table(index="dia", columns="hora", values="precio", aggfunc="mean")


def meteo_agregada(meteo: pd.DataFrame) -> pd.DataFrame:
    """Resume los puntos en índices nacionales (viento, sol, temperatura) y conserva algunos puntos."""
    def media(var, puntos):
        return meteo[[f"{var}__{p}" for p in puntos]].mean(axis=1)

    viento = media("wind_speed_100m", EOLICOS)
    agregada = pd.DataFrame({
        "viento": viento,
        # la potencia eólica crece ~ v^3 hasta saturar: un proxy simple
        "viento3": meteo[[f"wind_speed_100m__{p}" for p in EOLICOS]].clip(upper=45).pow(3).mean(axis=1),
        "sol": media("shortwave_radiation", SOLARES),
        "nubes": media("cloud_cover", SOLARES),
        "temp": media("temperature_2m", CONSUMO),
    })
    for p in EOLICOS:
        agregada[f"viento_{p}"] = meteo[f"wind_speed_100m__{p}"]
    # Grados de calefacción / refrigeración (la demanda sube con el frío y con el calor)
    agregada["calor"] = (agregada["temp"] - 22).clip(lower=0)
    agregada["frio"] = (15 - agregada["temp"]).clip(lower=0)
    return agregada


def construir(entregas: list[date], datos: dict) -> pd.DataFrame:
    """Tabla de variables para una lista de días de entrega (índice: inicio del periodo en UTC)."""
    P = tabla_diaria(datos["precios"])
    met = meteo_agregada(datos["meteo"])
    met_diaria = met.groupby(met.index.tz_convert(MADRID).date).mean()
    gas = datos["gas"].sort_index()

    filas = []
    for E in entregas:
        D = E - timedelta(days=1)
        if D not in P.index:
            continue
        precio_D = P.loc[D]
        # cierre de gas del último día hábil antes de D (el de D aún no ha cerrado a las 10 UTC)
        gas_previo = gas[gas.index < D]
        ttf = gas_previo.iloc[-1] if len(gas_previo) else np.nan
        ttf_7 = gas_previo.iloc[-6] if len(gas_previo) >= 6 else np.nan
        ref = dia_referencia(E)
        ultimos7 = P.loc[[d for d in (D - timedelta(days=i) for i in range(7)) if d in P.index]]
        met_E = met_diaria.loc[E] if E in met_diaria.index else None
        met_D = met_diaria.loc[D] if D in met_diaria.index else None

        for t in periodos_utc(E):
            h = t.tz_convert(MADRID).hour
            fila = {
                "instante": t, "entrega": E, "hora": h,
                "dia_semana": E.weekday(), "mes": E.month,
                "doy_sin": np.sin(2 * np.pi * E.timetuple().tm_yday / 365.25),
                "doy_cos": np.cos(2 * np.pi * E.timetuple().tm_yday / 365.25),
                "festivo": int(es_festivo(E)),
                "festivo_D": int(es_festivo(D)),
                # Precios conocidos
                "p_D_h": precio_D.get(h, np.nan),
                "p_D1_h": P.loc[D - timedelta(days=1)].get(h, np.nan) if D - timedelta(days=1) in P.index else np.nan,
                "p_E7_h": P.loc[E - timedelta(days=7)].get(h, np.nan) if E - timedelta(days=7) in P.index else np.nan,
                "p_ref_h": P.loc[ref].get(h, np.nan) if ref in P.index else np.nan,
                "p_D_media": precio_D.mean(), "p_D_max": precio_D.max(), "p_D_min": precio_D.min(),
                "p_D_solar": precio_D.loc[[x for x in range(11, 17) if x in precio_D.index]].mean(),
                "p_D_noche": precio_D.loc[[x for x in (20, 21, 22) if x in precio_D.index]].mean(),
                "p_7_media_h": ultimos7[h].mean() if h in ultimos7 else np.nan,
                "p_7_std_h": ultimos7[h].std() if h in ultimos7 else np.nan,
                "p_7_media": ultimos7.values.mean(),
                "ttf": ttf, "ttf_cambio": ttf - ttf_7,
            }
            # Meteorología prevista para la hora t, para la misma hora de D y medias diarias
            for col in met.columns:
                fila[f"m_{col}"] = met[col].get(t, np.nan)
            for col in ("viento", "viento3", "sol", "temp"):
                fila[f"m_{col}_D"] = met[col].get(t - pd.Timedelta(days=1), np.nan)
                fila[f"d_{col}"] = fila[f"m_{col}"] - fila[f"m_{col}_D"]
                if met_E is not None:
                    fila[f"md_{col}_E"] = met_E[col]
                if met_E is not None and met_D is not None:
                    fila[f"dd_{col}"] = met_E[col] - met_D[col]
            filas.append(fila)

    X = pd.DataFrame(filas).set_index("instante")
    X["objetivo"] = datos["precios"].reindex(X.index)
    return X


def anadir_extras(X: pd.DataFrame, datos: dict) -> pd.DataFrame:
    """Variables adicionales (v1.1): consenso de varios modelos meteorológicos, generación real
    de REData y algunos precios más. Si falta alguna fuente, sus columnas quedan vacías (NaN)."""
    X = X.copy()
    entrega = pd.Series(X["entrega"].values, index=X.index)
    P = tabla_diaria(datos["precios"])

    # 1) Consenso meteorológico: media y discrepancia entre ECMWF, GFS e ICON
    modelos = [meteo_agregada(m) for m in datos.get("meteo_modelos", {}).values()]
    if modelos:
        for var in ("viento", "viento3", "sol", "nubes", "temp"):
            tabla = pd.concat([m[var] for m in modelos], axis=1)
            media, disp = tabla.mean(axis=1), tabla.std(axis=1)
            X[f"mm_{var}"] = media.reindex(X.index).values
            X[f"mm_{var}_disp"] = disp.reindex(X.index).values
            X[f"mm_{var}_D"] = media.reindex(X.index - pd.Timedelta(days=1)).values
            X[f"mmd_{var}"] = X[f"mm_{var}"] - X[f"mm_{var}_D"]
            diaria = media.groupby(media.index.tz_convert(MADRID).date).mean()
            X[f"mm_{var}_E"] = entrega.map(diaria).values
            X[f"mmdd_{var}"] = X[f"mm_{var}_E"] - (entrega - timedelta(days=1)).map(diaria).values

    # 2) REData: el último día completo a las 10 UTC de D es D-1 = E-2. En GW medios.
    if "redata" in datos:
        r = datos["redata"] / 24000
        for col in r.columns:
            X[f"r_{col}"] = (entrega - timedelta(days=2)).map(r[col]).values
        for col in ("nuclear", "hidraulica"):
            if col in r:
                X[f"r_{col}_cambio7"] = X[f"r_{col}"] - (entrega - timedelta(days=9)).map(r[col]).values
        if "hidraulica" in r:
            X["r_hidraulica_media7"] = (entrega - timedelta(days=2)).map(r["hidraulica"].rolling(7).mean()).values

    # 3) Más precios: horas a ~0 € en D (canibalización solar), horas vecinas, media de 28 días
    horas_cero = (P <= 1).sum(axis=1)
    X["p_D_horas_cero"] = (entrega - timedelta(days=1)).map(horas_cero).values
    media28 = P.rolling(28, min_periods=14).mean()
    D = entrega - timedelta(days=1)
    for desfase, nombre in ((-1, "p_D_hmenos1"), (1, "p_D_hmas1")):
        X[nombre] = [P.at[d, h + desfase] if d in P.index and (h + desfase) in P.columns else np.nan
                     for d, h in zip(D, X["hora"])]
    X["p_28_media_h"] = [media28.at[d, h] if d in media28.index and h in media28.columns else np.nan
                         for d, h in zip(D, X["hora"])]
    return X


COLUMNAS_NO_VARIABLE = ["entrega", "objetivo", "escala"]


def columnas_modelo(X: pd.DataFrame) -> list[str]:
    return [c for c in X.columns if c not in COLUMNAS_NO_VARIABLE]
