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


def anadir_extras2(X: pd.DataFrame, datos: dict) -> pd.DataFrame:
    """Variables en pruebas: gas español (MIBGAS), CO2, intercambios y bombeo (REData) e índices
    eólico/solar con más puntos ponderados. Si falta una fuente, sus columnas quedan vacías."""
    from prevision.datos_extra import PUNTOS_EOLICOS, PUNTOS_SOLARES
    X = X.copy()
    entrega = pd.Series(X["entrega"].values, index=X.index)
    D = entrega - timedelta(days=1)

    if "mibgas" in datos:  # índice del día D (fijado en la negociación de D-1)
        g = datos["mibgas"]
        X["mibgas_D"] = D.map(g).values
        X["mibgas_cambio7"] = X["mibgas_D"] - (D - timedelta(days=7)).map(g).values
    if "co2" in datos:  # último cierre antes de D
        c = datos["co2"].sort_index()
        dias = pd.Index(c.index)
        pos = dias.searchsorted(list(D), side="left") - 1
        X["co2"] = [c.iloc[p] if p >= 0 else np.nan for p in pos]
    if "balance" in datos:  # último día completo: E-2. En GW medios
        b = datos["balance"] / 24000
        for col in b.columns:
            X[f"b_{col}"] = (entrega - timedelta(days=2)).map(b[col]).values

    modelos = datos.get("meteo2_modelos", {})
    if modelos:
        pe = {k: v[2] for k, v in PUNTOS_EOLICOS.items()}
        ps = {k: v[2] for k, v in PUNTOS_SOLARES.items()}
        eol, sol = [], []
        for m in modelos.values():
            v = m[[f"wind_speed_100m__{k}" for k in pe]].clip(upper=45)
            # curva de potencia aproximada: 0 por debajo de 11 km/h, satura a 45 km/h
            cf = ((v - 11).clip(lower=0) / (45 - 11)).pow(2).clip(upper=1)
            eol.append((cf * list(pe.values())).sum(axis=1) / sum(pe.values()))
            r = m[[f"shortwave_radiation__{k}" for k in ps]]
            sol.append((r * list(ps.values())).sum(axis=1) / sum(ps.values()))
        for nombre, lista in (("idx_eolico", eol), ("idx_solar", sol)):
            serie = pd.concat(lista, axis=1).mean(axis=1)
            X[nombre] = serie.reindex(X.index).values
            X[f"{nombre}_D"] = serie.reindex(X.index - pd.Timedelta(days=1)).values
            X[f"{nombre}_dif"] = X[nombre] - X[f"{nombre}_D"]
            diaria = serie.groupby(serie.index.tz_convert(MADRID).date).mean()
            X[f"{nombre}_E"] = entrega.map(diaria).values
            X[f"{nombre}_dd"] = X[f"{nombre}_E"] - D.map(diaria).values
    return X


def cargar_extras2(datos: dict) -> dict:
    """Añade al diccionario de datos las fuentes en pruebas que existan en data/."""
    from prevision.historico import MODELOS_METEO
    for nombre, col in (("mibgas", "mibgas"), ("co2", "co2")):
        if (CARPETA / f"{nombre}.parquet").exists():
            datos[nombre] = pd.read_parquet(CARPETA / f"{nombre}.parquet")[col]
    if (CARPETA / "balance.parquet").exists():
        datos["balance"] = pd.read_parquet(CARPETA / "balance.parquet")
    datos["meteo2_modelos"] = {m: pd.read_parquet(CARPETA / f"meteo2_{m}.parquet") for m in MODELOS_METEO
                               if (CARPETA / f"meteo2_{m}.parquet").exists()}
    if (CARPETA / "ec.parquet").exists():
        datos["ec"] = pd.read_parquet(CARPETA / "ec.parquet")
    return datos


def anadir_perfil_D(X: pd.DataFrame, datos: dict) -> pd.DataFrame:
    """Los 24 precios de D (y la media horaria de D-6) como columnas, como en LEAR."""
    X = X.copy()
    P = tabla_diaria(datos["precios"]).reindex(columns=range(24))
    D = pd.to_datetime(pd.Series(X["entrega"].values)) - pd.Timedelta(days=1)
    perfil = P.reindex(D.dt.date.values)
    for h in range(24):
        X[f"pD_{h}"] = perfil[h].values
    return X


def anadir_puntos(X: pd.DataFrame, datos: dict) -> pd.DataFrame:
    """Factor de carga eólico y radiación de cada uno de los 24 puntos (media de los modelos
    meteorológicos), a la hora t y su media diaria de E: el modelo aprende el peso de cada zona."""
    from prevision.datos_extra import PUNTOS_EOLICOS, PUNTOS_SOLARES
    modelos = list(datos.get("meteo2_modelos", {}).values())
    if not modelos:
        return X
    X = X.copy()
    entrega = pd.Series(X["entrega"].values, index=X.index)
    media = sum(m.reindex(modelos[0].index) for m in modelos) / len(modelos)
    columnas = {}
    for k in PUNTOS_EOLICOS:
        v = media[f"wind_speed_100m__{k}"].clip(upper=45)
        columnas[f"pt_eol_{k}"] = ((v - 11).clip(lower=0) / 34).pow(2).clip(upper=1)
    for k in PUNTOS_SOLARES:
        columnas[f"pt_sol_{k}"] = media[f"shortwave_radiation__{k}"]
    tabla = pd.DataFrame(columnas)
    for c in tabla.columns:
        X[c] = tabla[c].reindex(X.index).values
        if c.startswith("pt_eol_"):
            diaria = tabla[c].groupby(tabla.index.tz_convert(MADRID).date).mean()
            X[f"{c}_E"] = entrega.map(diaria).values
    return X


def anadir_ec(X: pd.DataFrame, datos: dict) -> pd.DataFrame:
    """Previsiones oficiales day-ahead de España (Energy-Charts / ENTSO-E), solo las disponibles a tiempo:
    - demanda prevista para E (se publica antes de las 10:00 de D, 2 h antes del cierre de la subasta)
    - eólica, solar y demanda previstas para D (publicadas el día anterior): con qué renovables se formó
      el precio de D. La eólica/solar prevista para E sale hacia las 18:00 de D: NO se usa (sería fuga)."""
    if "ec" not in datos:
        return X
    X = X.copy()
    ec = datos["ec"] / 1000  # GW
    entrega = pd.Series(X["entrega"].values, index=X.index)
    D = entrega - timedelta(days=1)
    antes = X.index - pd.Timedelta(days=1)
    X["ec_demanda_E"] = ec["ec_demanda"].reindex(X.index).values
    X["ec_demanda_D"] = ec["ec_demanda"].reindex(antes).values
    X["ec_demanda_dif"] = X["ec_demanda_E"] - X["ec_demanda_D"]
    X["ec_eolica_D"] = ec["ec_eolica"].reindex(antes).values
    X["ec_solar_D"] = ec["ec_solar"].reindex(antes).values
    X["ec_residual_D"] = X["ec_demanda_D"] - X["ec_eolica_D"] - X["ec_solar_D"]
    diaria = ec.groupby(ec.index.tz_convert(MADRID).date).mean()
    X["ec_demanda_E_dia"] = entrega.map(diaria["ec_demanda"]).values
    for col in ("ec_eolica", "ec_solar", "ec_demanda"):
        X[f"{col}_D_dia"] = D.map(diaria[col]).values
    return X


def anadir_entsoe(X: pd.DataFrame, datos: dict) -> pd.DataFrame:
    """Variables de ENTSO-E (token): precios de Francia y Portugal de D, capacidades de interconexión
    (prevista para E y la de D) y embalses (dato con ≥ 7 días de antigüedad, por el retraso de publicación)."""
    X = X.copy()
    entrega = pd.Series(X["entrega"].values, index=X.index)
    antes = X.index - pd.Timedelta(days=1)
    if "entsoe_horario" in datos:
        h = datos["entsoe_horario"]
        for zona in ("fr", "pt"):
            col = f"precio_{zona}"
            if col in h:
                X[f"{zona}_D_h"] = h[col].reindex(antes).values
                diaria = h[col].groupby(h.index.tz_convert(MADRID).date).mean()
                X[f"{zona}_D_media"] = (entrega - timedelta(days=1)).map(diaria).values
                X[f"{zona}_dif_D_h"] = X["p_D_h"] - X[f"{zona}_D_h"]
        for col in ("cap_fr_es", "cap_es_fr", "cap_pt_es", "cap_es_pt"):
            if col in h:
                X[f"{col}_E"] = h[col].reindex(X.index).values
                X[f"{col}_D"] = h[col].reindex(antes).values
    if "entsoe_embalses" in datos:
        emb = datos["entsoe_embalses"].sort_index()
        fechas = emb.index.tz_convert(MADRID).date if emb.index.tz is not None else emb.index.date
        tabla = pd.Series(emb.values / 1e6, index=pd.Index(fechas))  # TWh
        limite = (entrega - timedelta(days=8)).values
        pos = pd.Index(tabla.index).searchsorted(limite, side="right") - 1
        X["embalses"] = [tabla.iloc[p] if p >= 0 else np.nan for p in pos]
    return X


GRUPOS_CENTRAL = {"B14": "nuclear", "B04": "gas", "B05": "carbon", "B10": "hidraulica", "B11": "hidraulica",
                  "B12": "hidraulica"}


def anadir_paradas(X: pd.DataFrame, datos: dict, corte_utc: str = "09:30") -> pd.DataFrame:
    """MW indisponibles por tipo de central para cada hora de E y de D, contando solo los avisos de ENTSO-E
    (paradas programadas y averías) publicados antes de las `corte_utc` del día D. Si una unidad tiene varios
    avisos solapados se toma la mayor indisponibilidad (no se suman)."""
    if "paradas" not in datos:
        return X
    p = datos["paradas"].copy()
    p = p[~p["estado"].isin(["A09", "A13"])]  # avisos cancelados o retirados
    p["no_disp"] = (p["nominal"] - p["disponible"]).clip(lower=0)
    p["grupo"] = p["psr"].map(GRUPOS_CENTRAL).fillna("otras")
    p = p[p["no_disp"] > 0]
    creado, ini, fin = p["creado"].values, p["inicio"].values, p["fin"].values
    X = X.copy()
    grupos = ["nuclear", "gas", "carbon", "hidraulica", "otras"]
    res = {g: np.full(len(X), np.nan) for g in grupos}
    res_D = {g: np.full(len(X), np.nan) for g in grupos}
    for E, filas in pd.Series(np.arange(len(X)), index=X["entrega"].values).groupby(level=0):
        corte = np.datetime64(pd.Timestamp(f"{E - timedelta(days=1)} {corte_utc}", tz="UTC").tz_convert(None))
        visibles = p[creado < corte]
        if visibles.empty:
            continue
        horas = X.index[filas.values]
        for destino, desfase in ((res, pd.Timedelta(0)), (res_D, pd.Timedelta(days=1))):
            for k, t in zip(filas.values, horas - desfase):
                t64 = np.datetime64(t.tz_convert(None))
                activos = visibles[(visibles["inicio"].values <= t64) & (visibles["fin"].values > t64)]
                por_unidad = activos.groupby(["grupo", "unidad"])["no_disp"].max().groupby(level=0).sum()
                for g in grupos:
                    destino[g][k] = por_unidad.get(g, 0.0) / 1000  # GW
    for g in grupos:
        X[f"par_{g}_E"] = res[g]
        X[f"par_{g}_D"] = res_D[g]
        X[f"par_{g}_dif"] = X[f"par_{g}_E"] - X[f"par_{g}_D"]
    return X


def anadir_esios(X: pd.DataFrame, datos: dict) -> pd.DataFrame:
    """Previsiones D+1 de Red Eléctrica (e·sios), publicadas hacia las 09:00 UTC de D: eólica, fotovoltaica
    y demanda para cada hora de E, la demanda residual (demanda − eólica − FV), lo mismo para D y el cambio."""
    if "esios" not in datos:
        return X
    X = X.copy()
    e = datos["esios"] / 1000  # GW
    e = e.assign(esios_residual=e["esios_demanda"] - e["esios_eolica"] - e["esios_fv"])
    entrega = pd.Series(X["entrega"].values, index=X.index)
    antes = X.index - pd.Timedelta(days=1)
    diaria = e.groupby(e.index.tz_convert(MADRID).date).mean()
    for col in e.columns:
        X[f"{col}_E"] = e[col].reindex(X.index).values
        X[f"{col}_D"] = e[col].reindex(antes).values
        X[f"{col}_dif"] = X[f"{col}_E"] - X[f"{col}_D"]
        X[f"{col}_E_dia"] = entrega.map(diaria[col]).values
        X[f"{col}_dd"] = X[f"{col}_E_dia"] - (entrega - timedelta(days=1)).map(diaria[col]).values
    return X


def _futuro_por_dia(omip: pd.DataFrame, minimo_dias: int = 2) -> pd.DataFrame:
    """Para cada día de entrega, la referencia de OMIP de la última sesión cerrada al menos `minimo_dias`
    antes (con 2: la tarde de D-1 para predecir E el día D). Contrato diario; si no hay, el de fin de semana."""
    t = omip[omip["tipo"].isin(["D", "WE"]) & omip["ref"].notna()].copy()
    t["sesion"] = pd.to_datetime(t["sesion"])
    t["inicio"] = pd.to_datetime(t["inicio"])
    filas = []
    for tipo, desfase in (("D", [0]), ("WE", [0, 1])):
        s = t[t["tipo"] == tipo]
        for d in desfase:  # el de fin de semana vale para el sábado y el domingo
            filas.append(s.assign(entrega=s["inicio"] + pd.Timedelta(days=d), prioridad=0 if tipo == "D" else 1))
    t = pd.concat(filas)
    t = t[(t["entrega"] - t["sesion"]).dt.days >= minimo_dias]
    t = t.sort_values(["entrega", "sesion", "prioridad"], ascending=[True, False, True])
    t = t.drop_duplicates("entrega")
    return pd.DataFrame({"fut": t["ref"].values, "fut_ant": t["ref_anterior"].values,
                         "fut_edad": (t["entrega"] - t["sesion"]).dt.days.values},
                        index=t["entrega"].dt.date.values)


def anadir_futuros(X: pd.DataFrame, datos: dict) -> pd.DataFrame:
    """Futuros de OMIP: precio medio que espera el mercado para E (sesión de D-1 o anterior), su cambio
    respecto a la sesión previa, la diferencia con la media de D y lo que se equivocó el mercado con D."""
    if "omip" not in datos:
        return X
    X = X.copy()
    f = _futuro_por_dia(datos["omip"])
    entrega = pd.Series(X["entrega"].values, index=X.index)
    D = entrega - timedelta(days=1)
    X["fut_E"] = entrega.map(f["fut"]).values
    X["fut_edad"] = entrega.map(f["fut_edad"]).values
    X["fut_cambio"] = X["fut_E"] - entrega.map(f["fut_ant"]).values
    real = tabla_diaria(datos["precios"]).mean(axis=1)
    X["fut_menos_D"] = X["fut_E"] - D.map(real).values
    X["fut_error_D"] = D.map(f["fut"]).values - D.map(real).values
    if "p_D_h" in X:
        X["fut_hora"] = X["fut_E"] + (X["p_D_h"] - D.map(real).values)  # forma de D sobre el nivel del futuro
    return X


def anadir_curva(X: pd.DataFrame, datos: dict, dias: int = 14, k: int = 12, sufijo: str = "",
                 por_tramo: bool = False) -> pd.DataFrame:
    """Curva de oferta reciente: para cada hora de E, mediana del precio real en las k horas de los
    `dias` días anteriores (hasta D incluido) cuya demanda residual prevista por REE se parecía más
    a la prevista para esa hora de E. Capta que el precio no es lineal en la residual (por encima de
    ~14 GW lo marca el gas) sin anclarse al precio de D."""
    if "esios" not in datos:
        return X
    X = X.copy()
    e = datos["esios"] / 1000
    residual = (e["esios_demanda"] - e["esios_eolica"] - e["esios_fv"]).dropna()
    precio = datos["precios"].reindex(residual.index)
    pares = pd.DataFrame({"r": residual, "p": precio}).dropna()
    dia_par = pares.index.tz_convert(MADRID).date
    tramos = np.array([0] * 7 + [1] * 3 + [2] * 8 + [3] * 6)
    tramo_par = tramos[pares.index.tz_convert(MADRID).hour]
    tramo_X = tramos[X.index.tz_convert(MADRID).hour]
    salida = {f"curva{sufijo}_k": np.full(len(X), np.nan), f"curva{sufijo}_q25": np.full(len(X), np.nan),
              f"curva{sufijo}_q75": np.full(len(X), np.nan)}
    filas = pd.Series(np.arange(len(X)), index=X["entrega"].values)
    r_E = residual.reindex(X.index).values
    for E, idx in filas.groupby(level=0):
        sel = (dia_par < E) & (dia_par >= E - timedelta(days=dias))
        if sel.sum() < 48:
            continue
        r_p, p_p, t_p = pares["r"].values[sel], pares["p"].values[sel], tramo_par[sel]
        for i in idx.values:
            if np.isnan(r_E[i]):
                continue
            if por_tramo:
                mismo = t_p == tramo_X[i]
                cerca = np.argsort(np.abs(r_p[mismo] - r_E[i]))[:k]
                v = p_p[mismo][cerca]
            else:
                cerca = np.argsort(np.abs(r_p - r_E[i]))[:k]
                v = p_p[cerca]
            salida[f"curva{sufijo}_k"][i] = np.median(v)
            salida[f"curva{sufijo}_q25"][i], salida[f"curva{sufijo}_q75"][i] = np.percentile(v, [25, 75])
    for c, v in salida.items():
        X[c] = v
    X[f"curva{sufijo}_menos_D"] = X[f"curva{sufijo}_k"] - X["p_D_h"]
    return X
