"""
Evalúa medias de predicciones guardadas (data/pred_*.parquet).

    python -m experimentos.combinar "nombre1,nombre2,...[:peso]" ...

Cada argumento es un grupo de predicciones que se promedian por igual; con ':peso' el grupo
entra con ese peso en la combinación final (el resto se reparte entre los grupos sin peso).
"""
import sys

import pandas as pd

from experimentos.correr import resumen
from prevision.historico import CARPETA


def cargar(nombre):
    return pd.read_parquet(CARPETA / f"pred_{nombre}.parquet")["pred"]


def combinar(grupos):
    series, pesos = [], []
    for g in grupos:
        nombres, _, peso = g.partition(":")
        series.append(pd.concat([cargar(n) for n in nombres.split(",")], axis=1).mean(axis=1))
        pesos.append(float(peso) if peso else None)
    libre = (1 - sum(p for p in pesos if p)) / max(1, sum(p is None for p in pesos))
    pesos = [p if p is not None else libre for p in pesos]
    base = series[0]
    return sum(w * s.reindex(base.index).fillna(base) for w, s in zip(pesos, series))


if __name__ == "__main__":
    X = pd.read_parquet(CARPETA / "variables_ext.parquet")
    for arg in sys.argv[1:]:
        r = resumen(X, combinar(arg.split("|")))
        print(f"{arg[:90]:90s} 2025={r['2025_rMAE']:.3f} 2026={r['2026_rMAE']:.3f} "
              f"(rn {r['2025_rnMAE']:.3f}/{r['2026_rnMAE']:.3f}, P90 {r['2025_P90']:.1f}/{r['2026_P90']:.1f})")
