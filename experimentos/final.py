"""
Evaluación del sistema completo: media de variantes LightGBM + peso de LEAR + corrección de sesgo,
con la tabla mes a mes y la versión a 15 minutos.

    python -m experimentos.final "<lgbm1,lgbm2,...>" <lear> <peso_lear> [<nombre_salida>]
"""

import sys

import pandas as pd

from experimentos.combinar import combinar
from experimentos.correr import resumen
from prevision import sesgo
from prevision.backtest import metricas
from prevision.historico import CARPETA


def corregir(base: pd.Series, X: pd.DataFrame, precios: pd.Series) -> pd.Series:
    ent = X.loc[base.index, "entrega"]
    trozos = []
    for E in sorted(set(ent)):
        p = base[ent == E]
        trozos.append(sesgo.aplicar(p, E, base[ent < E], precios[precios.index < p.index.min()])[0])
    return pd.concat(trozos)


if __name__ == "__main__":
    lgbm, lear, peso = sys.argv[1], sys.argv[2], float(sys.argv[3])
    salida = sys.argv[4] if len(sys.argv) > 4 else "final"
    X = pd.read_parquet(CARPETA / "variables_ext.parquet")
    precios = pd.read_parquet(CARPETA / "precios.parquet")["precio"]
    cruda = combinar([lgbm, f"{lear}:{peso}"])
    final = corregir(cruda, X, precios)
    final.to_frame("pred").to_parquet(CARPETA / f"pred_{salida}.parquet")
    for nombre, s in (("sin corrección", cruda), ("con corrección", final)):
        r = resumen(X, s)
        print(f"{nombre}: 2025 rMAE={r['2025_rMAE']:.3f} rnMAE={r['2025_rnMAE']:.3f} P90={r['2025_P90']:.1f} "
              f"sesgo={r['2025_sesgo']:+.2f} | 2026 rMAE={r['2026_rMAE']:.3f} rnMAE={r['2026_rnMAE']:.3f} "
              f"P90={r['2026_P90']:.1f} sesgo={r['2026_sesgo']:+.2f}")
    T = X.loc[final.index]
    mes = pd.to_datetime(T["entrega"]).dt.to_period("M")
    tabla = pd.DataFrame({m: metricas(g["objetivo"], final.loc[g.index], g["p_ref_h"]) for m, g in T.groupby(mes)}).T
    print(tabla[["MAE", "MAE_bench", "rMAE", "sesgo", "P90"]].round(2).to_string())
    total = metricas(T["objetivo"], final, T["p_ref_h"])
    print(f"TOTAL ene-2025 a oct-2026: MAE={total['MAE']:.2f} (benchmark {total['MAE_bench']:.2f}) rMAE={total['rMAE']:.3f}")

    # A 15 minutos (desde oct-2025): forma media de 7 días
    from experimentos.cuartos import benchmark_cuartos, cuartos_desde_horario
    Q = pd.read_parquet(CARPETA / "precios_15min.parquet")["precio"]
    Q = Q[(Q.index >= pd.Timestamp("2025-10-08", tz="UTC")) & (Q.index <= final.index.max() + pd.Timedelta(minutes=45))]
    p15, b15 = cuartos_desde_horario(final, Q, "forma_7"), benchmark_cuartos(Q)
    ok = p15.notna() & b15.notna()
    print(f"15 min (oct-2025 a oct-2026): MAE={(p15 - Q)[ok].abs().mean():.2f} "
          f"benchmark={(b15 - Q)[ok].abs().mean():.2f} rMAE={(p15 - Q)[ok].abs().mean() / (b15 - Q)[ok].abs().mean():.3f}")
