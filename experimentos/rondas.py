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

# Ronda 5 (tabla ext4 = ext2 sin CO2 + factor de carga/radiación de cada punto): las 4 variantes
r5 = {f"e4_{k}": v for k, v in V1.items()}

# Ronda 6 (tabla ext2, sin CO2): las 4 variantes con reentrenamiento SEMANAL
r6 = {f"sem_{k}": dict(v, frecuencia="W", excluir=("co2",)) for k, v in V1.items()}

# Ronda 7 (tabla ext4): las 4 variantes con reentrenamiento SEMANAL
r7 = {f"sem4_{k}": dict(v, frecuencia="W") for k, v in V1.items()}

# Ronda 8 (tabla ext5 = ext4 + previsiones ENTSO-E sin fuga: demanda de E, eólica/solar/demanda de D), mensual
r8 = {f"e5_{k}": v for k, v in V1.items()}

# Ronda 9 (ext5): ¿de dónde sale la mejora de ENTSO-E? Sin la demanda prevista para E (horario dudoso)
_sin_E = ("ec_demanda_E", "ec_demanda_dif")
r9 = {
    "e5_extras_sinE": dict(excluir=_sin_E),
    "e5_hojas63_sinE": dict(V1["hojas63"], excluir=_sin_E),
    "e5_reciente365_sinE": dict(V1["reciente365"], excluir=_sin_E),
}

# Ronda 10 (ext5): las 4 variantes con reentrenamiento SEMANAL
r10 = {f"sem5_{k}": dict(v, frecuencia="W") for k, v in V1.items()}

# Ronda 11 (ext5, semanal): 3 semillas por variante
r11 = {f"sem5s3_{k}": dict(v, frecuencia="W", semillas=(0, 1, 2)) for k, v in V1.items()}

# Ronda 12 (ext6 = ext5 + ENTSO-E: precios FR/PT de D, capacidades, embalses), mensual (comparar con r8)
r12 = {f"e6_{k}": v for k, v in V1.items()}

# Ronda 13 (ext7 = ext5 + paradas de centrales ENTSO-E publicadas antes de las 09:30 UTC de D), mensual
r13 = {f"e7_{k}": v for k, v in V1.items()}

# Ronda 14 (ext8 = ext5 + previsiones D+1 de Red Eléctrica, e·sios), mensual (comparar con r8: 0,526 / 0,555)
r14 = {f"e8_{k}": v for k, v in V1.items()}

# Ronda 15 (ext9 = ext8 + futuros de OMIP de la sesión de D-1 o anterior), mensual (comparar con r14: 0,461 / 0,497)
r15 = {f"e9_{k}": v for k, v in V1.items()}

# Ronda 16 (ext8): variantes nuevas para diversificar la media (comparar cada una y la media de 5-8)
r16 = {
    "e8_lento": dict(ponderar="escala", params={"n_estimators": 2000, "learning_rate": 0.015}),
    "e8_l1": dict(ponderar="escala", params={"objective": "regression_l1"}),
    "e8_hojas127": dict(ponderar="escala", params={"num_leaves": 127, "min_child_samples": 80, "colsample_bytree": 0.4}),
    "e8_v730": dict(ponderar="escala", ventana_dias=730),
}

# Ronda 17 (ext10 = ext9 + curva de oferta reciente: precio en horas con residual REE parecida), mensual
r17 = {f"e10_{k}": v for k, v in V1.items()}

# Ronda 18 (ext10): objetivo = diferencia con la curva de oferta reciente, o con la media de D y la curva
r18 = {**{f"e10c_{k}": dict(v, objetivo="diferencia_curva") for k, v in V1.items()},
       **{f"e10m_{k}": dict(v, objetivo="diferencia_mixta") for k, v in V1.items()}}

# Ronda 19 (ext11 = ext10 + curvas de 7 días/k=6, 28 días/k=24 y 14 días/k=4), mensual
r19 = {f"e11_{k}": v for k, v in V1.items()}

# Ronda 20 (ext12 = ext11 + curvas por tramo del día, 14 días/k=6 y 28 días/k=10), mensual
r20 = {f"e12_{k}": v for k, v in V1.items()}

# Ronda 21 (ext11): objetivo = diferencia con la curva, para combinar con r19
r21 = {f"e11c_{k}": dict(v, objetivo="diferencia_curva") for k, v in V1.items()}
