"""
Modelo v1 en producción: entrenar con todo el histórico, guardar en modelos/ y predecir un día.

Entrenar (mensual):  python -m prevision.modelo entrenar
Predecir (prueba):   python -m prevision.modelo predecir --fecha 2026-10-22
"""

import argparse
import json
from datetime import date, timedelta
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from prevision import historico, variables
from prevision.backtest import PARAMS
from prevision.calendario import periodos_utc

CARPETA_MODELOS = Path(__file__).resolve().parent.parent / "modelos"
VERSION = "v1"
# Configuración elegida en el backtest (ver README)
CONFIG = {"objetivo": "diferencia", "desde_train": "2023-01-08", "semillas": [0, 1, 2, 3, 4],
          "params": {"objective": "huber", "alpha": 15}}


def entrenar_y_guardar():
    historico.actualizar()
    datos = variables.cargar()
    inicio = date.fromisoformat(CONFIG["desde_train"])
    ultimo = max(datos["precios"].index.tz_convert("Europe/Madrid").date)
    dias = [inicio + timedelta(days=i) for i in range((ultimo - inicio).days + 1)]
    X = variables.construir(dias, datos)
    X = X[X["objetivo"].notna()]
    columnas = variables.columnas_modelo(X)
    y = X["objetivo"] - X["p_D_h"].fillna(X["p_D_media"])

    carpeta = CARPETA_MODELOS / VERSION
    carpeta.mkdir(parents=True, exist_ok=True)
    for s in CONFIG["semillas"]:
        m = lgb.LGBMRegressor(**{**PARAMS, **CONFIG["params"], "random_state": s})
        m.fit(X[columnas], y)
        m.booster_.save_model(carpeta / f"semilla_{s}.txt")
    meta = {**CONFIG, "columnas": columnas, "entrenado_hasta": str(ultimo), "filas": len(X)}
    (carpeta / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    print(f"Modelo {VERSION} entrenado con {len(X)} horas hasta {ultimo}")


def datos_recientes(entrega: date) -> dict:
    """Solo lo necesario para predecir un día: ~3 semanas de precios, meteo de D y E, y gas."""
    D = entrega - timedelta(days=1)
    dias = [D - timedelta(days=i) for i in range(21)]
    precios = historico.descargar_precios(dias)
    if D not in set(precios.index.tz_convert("Europe/Madrid").date):
        raise RuntimeError(f"Faltan los precios de {D}")
    meteo = historico.descargar_meteo(D - timedelta(days=1), entrega + timedelta(days=1))
    gas = historico.descargar_gas(D - timedelta(days=30))
    return {"precios": precios, "meteo": meteo, "gas": gas}


def predecir(entrega: date, datos: dict | None = None) -> pd.Series:
    carpeta = CARPETA_MODELOS / VERSION
    meta = json.loads((carpeta / "meta.json").read_text())
    datos = datos or datos_recientes(entrega)
    X = variables.construir([entrega], datos)
    if X.empty:
        raise RuntimeError(f"No se pudieron construir las variables de {entrega}")
    # La meteo es imprescindible: si viene vacía, mejor caer al benchmark
    met = [c for c in X.columns if c.startswith("m_")]
    if X[met].isna().mean().mean() > 0.2:
        raise RuntimeError("Faltan demasiados datos meteorológicos")
    entradas = X[meta["columnas"]]
    pred = np.mean([lgb.Booster(model_file=str(carpeta / f"semilla_{s}.txt")).predict(entradas)
                    for s in meta["semillas"]], axis=0)
    pred = pred + X["p_D_h"].fillna(X["p_D_media"]).values
    serie = pd.Series(pred, index=X.index)
    return serie.reindex(periodos_utc(entrega))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("accion", choices=["entrenar", "predecir"])
    p.add_argument("--fecha", type=date.fromisoformat)
    a = p.parse_args()
    if a.accion == "entrenar":
        entrenar_y_guardar()
    else:
        print(predecir(a.fecha).round(2).to_string())
