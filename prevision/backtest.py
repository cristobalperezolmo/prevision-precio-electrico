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
from prevision.variables import columnas_modelo

PARAMS = dict(n_estimators=800, learning_rate=0.04, n_jobs=4, num_leaves=31, min_child_samples=30,
              subsample=0.8, subsample_freq=1, colsample_bytree=0.7, reg_lambda=1.0,
              objective="l1", verbose=-1)


def metricas(real: pd.Series, pred: pd.Series, bench: pd.Series) -> dict:
    ok = real.notna() & pred.notna() & bench.notna()
    e = (pred - real)[ok]
    mae_bench = (bench - real)[ok].abs().mean()
    mae = e.abs().mean()
    return {"MAE": mae, "RMSE": np.sqrt((e ** 2).mean()), "sesgo": e.mean(),
            "MedAE": e.abs().median(), "P90": e.abs().quantile(0.9),
            "rMAE": mae / mae_bench, "skill": 1 - mae / mae_bench, "MAE_bench": mae_bench}


def entrenar(X: pd.DataFrame, params=None, objetivo="diferencia") -> lgb.LGBMRegressor:
    """objetivo='diferencia': el modelo aprende cuánto cambia el precio respecto a D (misma hora)."""
    params = {**PARAMS, **(params or {})}
    X = X[X["objetivo"].notna()]
    y = X["objetivo"] - X["p_D_h"].fillna(X["p_D_media"]) if objetivo == "diferencia" else X["objetivo"]
    modelo = lgb.LGBMRegressor(**params)
    modelo.fit(X[columnas_modelo(X)], y)
    modelo.objetivo_ = objetivo
    return modelo


def predecir(modelo, X: pd.DataFrame) -> pd.Series:
    pred = modelo.predict(X[columnas_modelo(X)])
    if modelo.objetivo_ == "diferencia":
        pred = pred + X["p_D_h"].fillna(X["p_D_media"]).values
    return pd.Series(pred, index=X.index)


def walk_forward(X: pd.DataFrame, desde_test: date, desde_train: date | None = None,
                 params=None, objetivo="diferencia", semillas=(0,)) -> pd.Series:
    meses = pd.period_range(desde_test, X["entrega"].max(), freq="M")
    preds = []
    for mes in meses:
        inicio, fin = mes.start_time.date(), mes.end_time.date()
        train = X[(X["entrega"] < inicio) & ((X["entrega"] >= desde_train) if desde_train else True)]
        test = X[(X["entrega"] >= inicio) & (X["entrega"] <= fin)]
        if test.empty:
            continue
        # promedio de varias semillas: reduce la varianza del modelo
        p = np.mean([predecir(entrenar(train, {**(params or {}), "random_state": s}, objetivo), test)
                     for s in semillas], axis=0)
        preds.append(pd.Series(p, index=test.index))
    return pd.concat(preds)


def informe(X: pd.DataFrame, pred: pd.Series, titulo=""):
    T = X.loc[pred.index]
    m = metricas(T["objetivo"], pred, T["p_ref_h"])
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
