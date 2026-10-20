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

# Ronda 3 (tabla ext3 = ext2 + perfil de D): objetivo, ventana, frecuencia, perfil, tasa de aprendizaje
_sin_perfil = ("pD_",)
r3 = {
    "r3_perfil": dict(),
    "r3_ref": dict(excluir=_sin_perfil, objetivo="diferencia_ref"),
    "r3_v730": dict(excluir=_sin_perfil, ventana_dias=730),
    "r3_lento": dict(excluir=_sin_perfil, params={"n_estimators": 2000, "learning_rate": 0.015}),
    "r3_semanal": dict(excluir=_sin_perfil, frecuencia="W"),
    "r3_directo": dict(excluir=_sin_perfil, objetivo="directo"),
}

# Ronda 4 (tabla ext2): qué fuentes nuevas aportan (quitando cada grupo) y ventana de 730 días en el resto
_v = V1
r4 = {
    "r4_sin_co2": dict(excluir=("co2",)),
    "r4_sin_mibgas": dict(excluir=("mibgas",)),
    "r4_sin_balance": dict(excluir=("b_",)),
    "r4_sin_idx": dict(excluir=("idx_",)),
    "r4_hojas63_v730": dict(_v["hojas63"], ventana_dias=730),
    "r4_tramos_v730": dict(_v["tramos"], ventana_dias=730),
    "r4_reciente365_v730": dict(_v["reciente365"], ventana_dias=730),
}
