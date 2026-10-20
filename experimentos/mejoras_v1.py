"""
Experimentos para mejorar el v1 sin datos de ENTSO-E.

Para no engañarnos: las variantes se comparan en 2025 ("ajuste") y la ganadora se confirma
en 2026 ("confirmación"), un periodo que no se usa para elegir.

Uso: python -m experimentos.mejoras_v1
"""

from datetime import date, timedelta
from multiprocessing import Pool

import pandas as pd

from prevision import variables as V
from prevision.backtest import anadir_escala, informe, walk_forward
from prevision.historico import CARPETA

BASE_HUBER = {"objective": "huber", "alpha": 15, "n_jobs": 2}


def preparar():
    datos = V.cargar()
    inicio = date(2023, 1, 8)
    fin = max(datos["precios"].index.tz_convert("Europe/Madrid").date)
    X = V.construir([inicio + timedelta(days=i) for i in range((fin - inicio).days + 1)], datos)
    X = V.anadir_extras(X, datos)
    X = anadir_escala(X, datos["precios"])
    X.to_parquet(CARPETA / "variables_ext.parquet")
    return X


def columnas_base(X):
    extras = ("mm_", "mmd_", "mmdd_", "r_", "p_D_horas_cero", "p_D_hmenos1", "p_D_hmas1", "p_28_media_h")
    return [c for c in V.columnas_modelo(X) if not c.startswith(extras)]


def variantes(X):
    base = columnas_base(X)
    return {
        "v1_actual": dict(columnas=base),
        "v1+pesos": dict(columnas=base, ponderar="escala"),
        "extras": dict(),
        "extras+pesos": dict(ponderar="escala"),
        "extras+pesos+reciente365": dict(ponderar="escala", vida_media=365),
        "extras+pesos+reciente180": dict(ponderar="escala", vida_media=180),
        "extras+pesos+hojas63": dict(ponderar="escala", params={"num_leaves": 63, "min_child_samples": 50,
                                                                "colsample_bytree": 0.5}),
        "extras+pesos+hojas15": dict(ponderar="escala", params={"num_leaves": 15, "n_estimators": 1500,
                                                                "learning_rate": 0.03}),
        # Modelos separados por tramo del día y/o estación del año
        "extras+pesos+tramos": dict(ponderar="escala", segmentar="tramos"),
        "extras+pesos+estacion": dict(ponderar="escala", segmentar="estacion"),
        "extras+pesos+tramos_estacion": dict(ponderar="escala", segmentar="tramos_estacion"),
    }


def correr(args):
    nombre, kw = args
    X = pd.read_parquet(CARPETA / "variables_ext.parquet")
    kw = {**kw, "params": {**BASE_HUBER, **kw.get("params", {})}}
    pred = walk_forward(X, date(2025, 1, 1), **kw)
    pred.to_frame("pred").to_parquet(CARPETA / f"pred_{nombre}.parquet")
    filas = {}
    for periodo, (a, b) in {"ajuste_2025": ("2025-01-01", "2025-12-31"),
                            "confirmacion_2026": ("2026-01-01", "2026-12-31")}.items():
        p = pred.loc[a:b]
        m, _ = informe(X, p, imprimir=False)
        filas.update({f"{periodo}_{k}": v for k, v in m.items() if k in ("rMAE", "rnMAE", "nMAE", "P90", "sesgo")})
    print(f"{nombre}: " + "  ".join(f"{k}={v:.3f}" for k, v in filas.items()), flush=True)
    return nombre, filas


if __name__ == "__main__":
    X = preparar()
    print(f"Variables: {X.shape}", flush=True)
    with Pool(4) as pool:
        resultados = dict(pool.map(correr, variantes(X).items()))
    tabla = pd.DataFrame(resultados).T
    print(tabla.round(3).to_string())
    tabla.to_csv(CARPETA / "experimentos_mejoras_v1.csv")
