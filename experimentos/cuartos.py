"""
¿Cómo pasar la previsión horaria a 15 minutos? Se compara con los precios cuartohorarios reales
(desde oct-2025) y con el benchmark de la plataforma aplicado a cada cuarto de hora.
"""
import numpy as np
import pandas as pd

from prevision.benchmark import dia_referencia
from prevision.calendario import MADRID


def tabla_cuartos(q: pd.Series) -> pd.DataFrame:
    local = q.index.tz_convert(MADRID)
    df = pd.DataFrame({"dia": local.date, "c": local.hour * 4 + local.minute // 15, "p": q.values})
    return df.pivot_table(index="dia", columns="c", values="p", aggfunc="mean")


def cuartos_desde_horario(H: pd.Series, Q: pd.Series, metodo: str) -> pd.Series:
    """H: previsión horaria (índice UTC). Q: precios cuartohorarios reales (solo días anteriores
    a cada entrega se usan para la forma). Devuelve la previsión en los instantes de Q."""
    T = tabla_cuartos(Q)
    horaria = T.T.groupby(T.columns // 4).transform("mean").T  # media horaria de cada cuarto
    desvio = T - horaria  # forma dentro de la hora
    idx = Q.index
    local = idx.tz_convert(MADRID)
    hora_utc = idx.floor("1h")
    base = H.reindex(hora_utc).values
    if metodo == "plano":
        return pd.Series(base, index=idx)
    if metodo.startswith("interp"):
        # interpolación lineal entre los centros de cada hora
        centros = H.copy(); centros.index = centros.index + pd.Timedelta(minutes=30)
        objetivo = idx + pd.Timedelta(minutes=7.5)
        todo = pd.concat([centros, pd.Series(np.nan, index=objetivo)]).sort_index()
        todo = todo[~todo.index.duplicated()].interpolate(method="time", limit_direction="both")
        base = todo.reindex(objetivo).values
        if metodo == "interp":
            return pd.Series(base, index=idx)
    entrega = pd.Series(local.date, index=idx)
    c = local.hour * 4 + local.minute // 15
    if metodo.endswith("forma_D"):
        D = entrega - pd.Timedelta(days=1)
        forma = [desvio.at[d, k] if d in desvio.index else 0.0 for d, k in zip(D, c)]
    elif metodo.endswith("forma_7"):
        media7 = desvio.rolling(7, min_periods=3).mean().shift(1)  # solo días anteriores
        forma = [media7.at[d, k] if d in media7.index else 0.0 for d, k in zip(entrega, c)]
    return pd.Series(base + np.nan_to_num(np.array(forma, dtype=float)), index=idx)


def benchmark_cuartos(Q: pd.Series) -> pd.Series:
    T = tabla_cuartos(Q)
    local = Q.index.tz_convert(MADRID)
    vals = []
    for d, k in zip(local.date, local.hour * 4 + local.minute // 15):
        ref = dia_referencia(d)
        vals.append(T.at[ref, k] if ref in T.index and k in T.columns else np.nan)
    return pd.Series(vals, index=Q.index)


if __name__ == "__main__":
    from prevision.historico import CARPETA
    Q = pd.read_parquet(CARPETA / "precios_15min.parquet")["precio"]
    preds = [pd.read_parquet(CARPETA / f"pred_{n}.parquet")["pred"] for n in
             ["extras", "extras+pesos+reciente365", "extras+pesos+hojas63", "extras+pesos+tramos"]]
    H = pd.concat(preds, axis=1).mean(axis=1)
    Q = Q[(Q.index >= pd.Timestamp("2025-10-08", tz="UTC")) & (Q.index <= H.index.max() + pd.Timedelta(minutes=45))]
    bench = benchmark_cuartos(Q)
    real_h = Q.resample("1h").mean()
    print(f"Horario (referencia): MAE={(H.reindex(real_h.index) - real_h).abs().mean():.2f}")
    for metodo in ["plano", "forma_D", "forma_7", "interp", "interp+forma_7"]:
        p = cuartos_desde_horario(H, Q, metodo)
        ok = p.notna() & bench.notna()
        mae, mae_b = (p - Q)[ok].abs().mean(), (bench - Q)[ok].abs().mean()
        print(f"15 min {metodo:16s}: MAE={mae:.2f}  MAE_bench15={mae_b:.2f}  rMAE={mae / mae_b:.3f}")
