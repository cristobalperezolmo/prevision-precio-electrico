"""
Red neuronal tipo DNN de epftoolbox (perceptrón multicapa con 24 salidas, una por hora),
con las mismas entradas diarias que LEAR. Walk-forward con recalibración mensual.

Uso: python -m experimentos.dnn
"""

from datetime import date

import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor

from prevision import lear
from prevision.backtest import informe
from prevision.historico import CARPETA


def ajustar_y_predecir(F_train, Y_train, F_test, semillas=(0, 1, 2)):
    ok = F_train.notna().all(axis=1) & Y_train.notna().all(axis=1)
    F_train, Y_train = F_train[ok], Y_train[ok]
    columnas = [c for c in F_train.columns if F_train[c].std() > 0]
    ex = lear.Escalador().ajustar(F_train[columnas].values)
    ey = lear.Escalador().ajustar(Y_train.values)
    A = ex.transformar(F_train[columnas].values)
    B = ex.transformar(F_test[columnas].fillna(pd.Series(ex.med, index=columnas)).values)
    Z = ey.transformar(Y_train.values)
    salidas = []
    for s in semillas:
        m = MLPRegressor(hidden_layer_sizes=(256, 128), alpha=1e-3, learning_rate_init=1e-3,
                         max_iter=400, early_stopping=True, validation_fraction=0.15,
                         n_iter_no_change=25, random_state=s)
        m.fit(A, Z)
        salidas.append(m.predict(B))
    return ey.invertir(np.mean(salidas, axis=0))


if __name__ == "__main__":
    X = pd.read_parquet(CARPETA / "variables_ext.parquet")
    precios = pd.read_parquet(CARPETA / "precios.parquet")["precio"]
    F, Y = lear.matriz_diaria(X, precios)
    dias_test = [d for d in F.index if d >= pd.Timestamp("2025-01-01").date()]
    meses = sorted({(d.year, d.month) for d in dias_test})
    preds = []
    for a, m in meses:
        bloque = [d for d in dias_test if (d.year, d.month) == (a, m)]
        train = F.index[F.index < bloque[0]]
        p = ajustar_y_predecir(F.loc[train], Y.loc[train], F.loc[bloque])
        preds.append(pd.DataFrame(p, index=bloque, columns=range(24)))
        print(f"{a}-{m:02d}", flush=True)
    s = lear.a_horario(pd.concat(preds), X).dropna()
    s.to_frame("pred").to_parquet(CARPETA / "pred_dnn.parquet")
    for per, (i, f) in {"2025": ("2025-01-01", "2025-12-31"), "2026": ("2026-01-01", "2026-12-31")}.items():
        r, _ = informe(X, s.loc[i:f], imprimir=False)
        print(f"DNN {per}: rMAE={r['rMAE']:.3f} rnMAE={r['rnMAE']:.3f} P90={r['P90']:.1f} sesgo={r['sesgo']:+.2f}")
