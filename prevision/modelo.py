"""
Modelo v1 en producción: entrenar con todo el histórico, guardar en modelos/ y predecir un día.

El v1 es la media de 4 variantes de LightGBM (elegidas en experimentos/mejoras_v1.py):
  - extras:      todas las variables
  - reciente365: los datos de hace un año pesan la mitad
  - hojas63:     árboles más grandes
  - tramos:      un modelo por tramo del día (madrugada, solar, tarde-noche)
Todas predicen la diferencia con el precio de D a la misma hora.

Entrenar (mensual):  python -m prevision.modelo entrenar
Predecir (prueba):   python -m prevision.modelo predecir --fecha 2026-10-22
"""

import argparse
import gzip
import json
import sys
from datetime import date, timedelta
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from prevision import historico, variables
from prevision.backtest import anadir_escala, entrenar, segmento
from prevision.calendario import MADRID, periodos_utc

CARPETA_MODELOS = Path(__file__).resolve().parent.parent / "modelos"
VERSION = "v1"
DESDE_TRAIN = date(2023, 1, 8)
HUBER = {"objective": "huber", "alpha": 15}
# Configuración elegida en el backtest (ver README)
VARIANTES = {
    "extras": dict(),
    "reciente365": dict(ponderar="escala", vida_media=365),
    "hojas63": dict(ponderar="escala", params={"num_leaves": 63, "min_child_samples": 50,
                                               "colsample_bytree": 0.5}),
    "tramos": dict(ponderar="escala", segmentar="tramos"),
}


def _guardar(booster: lgb.Booster, ruta: Path):
    ruta.write_bytes(gzip.compress(booster.model_to_string().encode()))


def _cargar(ruta: Path) -> lgb.Booster:
    return lgb.Booster(model_str=gzip.decompress(ruta.read_bytes()).decode())


def entrenar_y_guardar():
    historico.actualizar()
    datos = variables.cargar()
    ultimo = max(datos["precios"].index.tz_convert(MADRID).date)
    dias = [DESDE_TRAIN + timedelta(days=i) for i in range((ultimo - DESDE_TRAIN).days + 1)]
    X = variables.anadir_extras(variables.construir(dias, datos), datos)
    X = anadir_escala(X, datos["precios"])
    X = X[X["objetivo"].notna()]

    carpeta = CARPETA_MODELOS / VERSION
    carpeta.mkdir(parents=True, exist_ok=True)
    for viejo in carpeta.glob("*.txt*"):
        viejo.unlink()
    meta = {"desde_train": str(DESDE_TRAIN), "entrenado_hasta": str(ultimo), "filas": len(X),
            "columnas": variables.columnas_modelo(X), "variantes": {}}
    for nombre, kw in VARIANTES.items():
        kw = dict(kw)
        segmentar = kw.pop("segmentar", None)
        params = {**HUBER, "random_state": 0, **kw.pop("params", {})}
        segmentos = segmento(X, segmentar)
        ficheros = {}
        for g in sorted(segmentos.unique()):
            m = entrenar(X[segmentos == g], params, ponderar=kw.get("ponderar"), vida_media=kw.get("vida_media"))
            fichero = f"{nombre}__{g}.txt.gz"
            _guardar(m.booster_, carpeta / fichero)
            ficheros[g] = fichero
        meta["variantes"][nombre] = {"segmentar": segmentar, "ficheros": ficheros}
        print(f"  {nombre}: {len(ficheros)} modelo(s)")
    (carpeta / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    print(f"Modelo {VERSION} entrenado con {len(X)} horas hasta {ultimo}")


def datos_recientes(entrega: date) -> tuple[dict, list[str]]:
    """Solo lo necesario para predecir un día. Los precios son imprescindibles;
    si falla otra fuente se sigue sin ella (sus variables quedan vacías) y se avisa."""
    D = entrega - timedelta(days=1)
    avisos = []
    precios = historico.descargar_precios([D - timedelta(days=i) for i in range(35)])
    if D not in set(precios.index.tz_convert(MADRID).date):
        raise RuntimeError(f"Faltan los precios de {D}")
    datos = {"precios": precios, "meteo_modelos": {}}

    for modelo in historico.MODELOS_METEO:
        try:
            datos["meteo_modelos"][modelo] = historico.descargar_meteo(D - timedelta(days=1),
                                                                       entrega + timedelta(days=1), modelo)
        except Exception as e:  # noqa: BLE001
            avisos.append(f"Meteo {modelo} no disponible: {e}")
    if not datos["meteo_modelos"]:
        raise RuntimeError("Ningún modelo meteorológico disponible")
    # Las variables base usan ECMWF; si no está, se sustituye por otro modelo
    if historico.MODELO_METEO in datos["meteo_modelos"]:
        datos["meteo"] = datos["meteo_modelos"][historico.MODELO_METEO]
    else:
        datos["meteo"] = next(iter(datos["meteo_modelos"].values()))
        avisos.append(f"Variables base calculadas con otro modelo meteorológico en lugar de {historico.MODELO_METEO}")

    try:
        datos["gas"] = historico.descargar_gas(D - timedelta(days=30))
    except Exception as e:  # noqa: BLE001
        avisos.append(f"Gas no disponible: {e}")
        datos["gas"] = pd.Series(dtype=float)
    try:
        datos["redata"] = historico.descargar_redata(D - timedelta(days=15), D)
    except Exception as e:  # noqa: BLE001
        avisos.append(f"REData no disponible: {e}")
    return datos, avisos


def predecir(entrega: date, datos: dict | None = None) -> tuple[pd.Series, list[str]]:
    """Devuelve (precios previstos indexados por inicio del periodo en UTC, avisos)."""
    carpeta = CARPETA_MODELOS / VERSION
    meta = json.loads((carpeta / "meta.json").read_text())
    avisos = []
    if datos is None:
        datos, avisos = datos_recientes(entrega)
    X = variables.anadir_extras(variables.construir([entrega], datos), datos)
    if X.empty:
        raise RuntimeError(f"No se pudieron construir las variables de {entrega}")
    # La meteo es imprescindible: si viene vacía, mejor caer al benchmark
    met = [c for c in X.columns if c.startswith("m_")]
    if X[met].isna().mean().mean() > 0.2:
        raise RuntimeError("Faltan demasiados datos meteorológicos")

    entradas = X.reindex(columns=meta["columnas"])
    predicciones = []
    for nombre, v in meta["variantes"].items():
        segmentos = segmento(X, v["segmentar"])
        p = pd.Series(np.nan, index=X.index)
        for g, fichero in v["ficheros"].items():
            filas = segmentos == g
            if filas.any():
                p[filas] = _cargar(carpeta / fichero).predict(entradas[filas])
        predicciones.append(p)
    pred = pd.concat(predicciones, axis=1).mean(axis=1) + X["p_D_h"].fillna(X["p_D_media"])
    for a in avisos:
        print(f"AVISO: {a}", file=sys.stderr)
    return pred.reindex(periodos_utc(entrega)), avisos


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("accion", choices=["entrenar", "predecir"])
    p.add_argument("--fecha", type=date.fromisoformat)
    a = p.parse_args()
    if a.accion == "entrenar":
        entrenar_y_guardar()
    else:
        print(predecir(a.fecha)[0].round(2).to_string())
