"""
Backtest walk-forward: se reentrena al principio de cada mes con todo lo anterior
y se predice ese mes día a día, igual que se haría en producción.

Uso: python -m prevision.backtest [--desde-test 2025-01-01]
"""

import argparse
from datetime import date

import lightgbm as lgb
import numpy as np
import pandas as pd

from prevision.historico import CARPETA
from prevision.variables import MADRID, columnas_modelo, tabla_diaria

PARAMS = dict(n_estimators=800, learning_rate=0.04, n_jobs=4, num_leaves=31, min_child_samples=30,
              subsample=0.8, subsample_freq=1, colsample_bytree=0.7, reg_lambda=1.0,
              objective="l1", verbose=-1)


def anadir_escala(X: pd.DataFrame, precios: pd.Series) -> pd.DataFrame:
    """Columna 'escala': cuánto se mueve típicamente el precio a esa hora (media de |p(d,h) - p(d-1,h)|
    en los 365 días anteriores a D). Imita la normalización de la plataforma, que da más peso
    a los errores en las horas tranquilas. No es una variable del modelo: solo pesa y normaliza."""
    P = tabla_diaria(precios)
    movimiento = P.diff().abs().rolling(365, min_periods=60).mean()
    D = pd.to_datetime(pd.Series(X["entrega"].values)) - pd.Timedelta(days=1)
    X = X.copy()
    X["escala"] = [movimiento.at[d.date(), h] if d.date() in movimiento.index and h in movimiento.columns
                   else np.nan for d, h in zip(D, X["hora"])]
    X["escala"] = X["escala"].clip(lower=1.0)
    return X


def metricas(real, pred, bench, escala=None) -> dict:
    ok = real.notna() & pred.notna() & bench.notna()
    e = (pred - real)[ok]
    mae_bench = (bench - real)[ok].abs().mean()
    mae = e.abs().mean()
    m = {"MAE": mae, "RMSE": np.sqrt((e ** 2).mean()), "sesgo": e.mean(),
         "MedAE": e.abs().median(), "P90": e.abs().quantile(0.9),
         "rMAE": mae / mae_bench, "skill": 1 - mae / mae_bench, "MAE_bench": mae_bench}
    if escala is not None:
        # Errores normalizados, como hace la plataforma (y su versión relativa al benchmark)
        en = (e / escala[ok]).abs()
        eb = ((bench - real)[ok] / escala[ok]).abs()
        m.update({"nMAE": en.mean(), "nP90": en.quantile(0.9), "nsesgo": (e / escala[ok]).mean(),
                  "rnMAE": en.mean() / eb.mean()})
    return m


def pesos_entrenamiento(X, ponderar=None, vida_media=None) -> np.ndarray | None:
    """ponderar='escala': cada hora pesa 1/escala (alinea el entrenamiento con la métrica de la plataforma).
    vida_media=N: los datos de hace N días pesan la mitad que los de ayer."""
    w = np.ones(len(X))
    if ponderar == "escala":
        w = w / X["escala"].fillna(X["escala"].median()).values
    if vida_media:
        edad = (pd.to_datetime(X["entrega"]).max() - pd.to_datetime(X["entrega"])).dt.days.values
        w = w * 0.5 ** (edad / vida_media)
    return w / w.mean()


def referencia(X: pd.DataFrame, objetivo: str) -> np.ndarray:
    """Valor del que se predice la diferencia: precio de D (diferencia), del día de referencia del
    benchmark (diferencia_ref) o nada (directo)."""
    if objetivo == "diferencia":
        return X["p_D_h"].fillna(X["p_D_media"]).values
    if objetivo == "diferencia_ref":
        return X["p_ref_h"].fillna(X["p_D_h"]).fillna(X["p_D_media"]).values
    if objetivo == "diferencia_curva":  # respecto a la curva de oferta reciente (variables.anadir_curva)
        return X["curva_k"].fillna(X["p_D_h"]).fillna(X["p_D_media"]).values
    if objetivo == "diferencia_mixta":  # respecto a la media de D y la curva
        d = X["p_D_h"].fillna(X["p_D_media"])
        return (0.5 * d + 0.5 * X["curva_k"].fillna(d)).values
    return np.zeros(len(X))


def entrenar(X: pd.DataFrame, params=None, objetivo="diferencia", ponderar=None, vida_media=None,
             columnas=None) -> lgb.LGBMRegressor:
    """objetivo='diferencia': el modelo aprende cuánto cambia el precio respecto a D (misma hora)."""
    params = {**PARAMS, **(params or {})}
    X = X[X["objetivo"].notna()]
    y = X["objetivo"] - referencia(X, objetivo)
    columnas = columnas or columnas_modelo(X)
    modelo = lgb.LGBMRegressor(**params)
    modelo.fit(X[columnas], y, sample_weight=pesos_entrenamiento(X, ponderar, vida_media))
    modelo.objetivo_ = objetivo
    modelo.columnas_ = columnas
    return modelo


def predecir(modelo, X: pd.DataFrame) -> pd.Series:
    pred = modelo.predict(X[modelo.columnas_]) + referencia(X, modelo.objetivo_)
    return pd.Series(pred, index=X.index)


TRAMOS = {"madrugada": range(0, 8), "solar": range(8, 18), "tarde_noche": range(18, 24)}
ESTACIONES = {"invierno": (12, 1, 2), "primavera": (3, 4, 5), "verano": (6, 7, 8), "otono": (9, 10, 11)}


def segmento(X: pd.DataFrame, segmentar: str | None) -> pd.Series:
    """Etiqueta de cada fila para entrenar un modelo distinto por segmento."""
    partes = []
    if segmentar in ("tramos", "tramos_estacion"):
        partes.append(X["hora"].map({h: t for t, horas in TRAMOS.items() for h in horas}))
    if segmentar in ("estacion", "tramos_estacion"):
        partes.append(X["mes"].map({m: e for e, meses in ESTACIONES.items() for m in meses}))
    if not partes:
        return pd.Series("todo", index=X.index)
    return partes[0] if len(partes) == 1 else partes[0] + "_" + partes[1]


def walk_forward(X: pd.DataFrame, desde_test: date, hasta_test: date | None = None,
                 desde_train: date | None = None, semillas=(0,), segmentar=None, ventana_dias=None,
                 frecuencia="M", **kw) -> pd.Series:
    """kw se pasa a entrenar(): params, objetivo, ponderar, vida_media, columnas.
    segmentar: None (un modelo), 'tramos' (por tramo del día), 'estacion' o 'tramos_estacion'.
    ventana_dias: entrenar solo con los últimos N días. frecuencia: 'M' (mensual) o 'W' (semanal)."""
    meses = pd.period_range(desde_test, hasta_test or X["entrega"].max(), freq=frecuencia)
    preds = []
    for mes in meses:
        inicio, fin = mes.start_time.date(), mes.end_time.date()
        inicio = max(inicio, desde_test)
        train = X[(X["entrega"] < inicio) & ((X["entrega"] >= desde_train) if desde_train else True)]
        if ventana_dias:
            train = train[train["entrega"] >= inicio - pd.Timedelta(days=ventana_dias)]
        test = X[(X["entrega"] >= inicio) & (X["entrega"] <= fin)]
        if test.empty:
            continue
        params = kw.pop("params", None) or {}
        seg_train, seg_test = segmento(train, segmentar), segmento(test, segmentar)
        for g in seg_test.unique():
            tr, te = train[seg_train == g], test[seg_test == g]
            # promedio de varias semillas: reduce la varianza del modelo
            p = np.mean([predecir(entrenar(tr, {**params, "random_state": s}, **kw), te)
                         for s in semillas], axis=0)
            preds.append(pd.Series(p, index=te.index))
        kw["params"] = params
    return pd.concat(preds).sort_index()


def informe(X: pd.DataFrame, pred: pd.Series, titulo="", imprimir=True):
    T = X.loc[pred.index]
    escala = T["escala"] if "escala" in T else None
    m = metricas(T["objetivo"], pred, T["p_ref_h"], escala)
    if imprimir:
        print(f"\n== {titulo} ==  " + "  ".join(f"{k}={v:.3f}" for k, v in m.items()))
    por_mes = pd.DataFrame({
        mes: metricas(g["objetivo"], pred.loc[g.index], g["p_ref_h"])
        for mes, g in T.groupby(pd.to_datetime(T["entrega"]).dt.to_period("M"))
    }).T[["MAE", "MAE_bench", "rMAE", "sesgo"]]
    return m, por_mes


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--desde-test", type=date.fromisoformat, default=date(2025, 1, 1))
    args = p.parse_args()
    X = pd.read_parquet(CARPETA / "variables.parquet")
    pred = walk_forward(X, args.desde_test)
    m, por_mes = informe(X, pred, "LightGBM")
    print(por_mes.round(2).to_string())
