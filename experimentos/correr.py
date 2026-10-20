"""
Lanzador genérico de experimentos LightGBM walk-forward (2025 = ajuste, 2026 = confirmación).

    python -m experimentos.correr <tabla_variables> <ronda> [n_procesos]

donde <ronda> es el nombre de un diccionario de variantes en experimentos/rondas.py.
Guarda cada predicción en data/pred_<nombre>.parquet y un resumen en data/ronda_<ronda>.csv.
"""

import sys
from datetime import date
from multiprocessing import Pool

import pandas as pd

from experimentos import rondas
from prevision.backtest import informe, walk_forward
from prevision.historico import CARPETA
from prevision.variables import columnas_modelo

BASE = {"objective": "huber", "alpha": 15, "n_jobs": 2}
PERIODOS = {"2025": ("2025-01-01", "2025-12-31"), "2026": ("2026-01-01", "2026-12-31")}


def resumen(X, pred) -> dict:
    filas = {}
    for periodo, (a, b) in PERIODOS.items():
        m, _ = informe(X, pred.loc[a:b].dropna(), imprimir=False)
        filas.update({f"{periodo}_{k}": v for k, v in m.items() if k in ("rMAE", "rnMAE", "P90", "sesgo")})
    return filas


def correr(args):
    tabla, nombre, kw = args
    X = pd.read_parquet(CARPETA / tabla)
    kw = dict(kw)
    excluir = kw.pop("excluir", ())
    solo = kw.pop("solo", None)
    columnas = [c for c in columnas_modelo(X) if not c.startswith(tuple(excluir))] if excluir else None
    if solo is not None:
        columnas = [c for c in columnas_modelo(X) if c.startswith(tuple(solo)) or c in solo]
    if columnas:
        kw["columnas"] = columnas
    kw["params"] = {**BASE, **kw.get("params", {})}
    pred = walk_forward(X, date(2025, 1, 1), **kw)
    pred.to_frame("pred").to_parquet(CARPETA / f"pred_{nombre}.parquet")
    filas = resumen(X, pred)
    print(f"{nombre}: " + "  ".join(f"{k}={v:.3f}" for k, v in filas.items()), flush=True)
    return nombre, filas


if __name__ == "__main__":
    tabla, ronda = sys.argv[1], sys.argv[2]
    n = int(sys.argv[3]) if len(sys.argv) > 3 else 4
    variantes = getattr(rondas, ronda)
    with Pool(n) as pool:
        res = dict(pool.map(correr, [(tabla, k, v) for k, v in variantes.items()]))
    tabla_res = pd.DataFrame(res).T
    print(tabla_res.round(3).to_string())
    tabla_res.to_csv(CARPETA / f"ronda_{ronda}.csv")
