"""Variantes de cada ronda de experimentos (ver experimentos/correr.py)."""

# Las 4 variantes del v1 en producción
V1 = {
    "extras": dict(),
    "reciente365": dict(ponderar="escala", vida_media=365),
    "hojas63": dict(ponderar="escala", params={"num_leaves": 63, "min_child_samples": 50, "colsample_bytree": 0.5}),
    "tramos": dict(ponderar="escala", segmentar="tramos"),
}

# Ronda 2: las mismas 4 variantes con las fuentes nuevas (MIBGAS, CO2, balance, índices ponderados)
r2 = {f"e2_{k}": v for k, v in V1.items()}
