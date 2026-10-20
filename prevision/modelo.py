"""
Modelo v1 en producción: entrenar con todo el histórico, guardar en modelos/ y predecir un día.

El v1 (versión 1.2) combina:
  - 85 %: media de 4 variantes de LightGBM (elegidas en experimentos/, ver NOTAS.md)
      extras:      todas las variables
      reciente365: los datos de hace un año pesan la mitad
      hojas63:     árboles más grandes
      tramos:      un modelo por tramo del día (madrugada, solar, tarde-noche)
    Todas predicen la diferencia con el precio de D a la misma hora.
  - 15 %: LEAR (un Lasso por hora, modelo lineal de referencia en la literatura)
  - y se le resta el 70 % del error reciente a cada hora (prevision/sesgo.py).

Entrenar (mensual):  python -m prevision.modelo entrenar
Predecir (prueba):   python -m prevision.modelo predecir --fecha 2026-10-22
"""

import argparse
import gzip
import json
import sys
from datetime import date, timedelta
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd

from prevision import datos_extra, historico, lear, sesgo, variables
from prevision.backtest import anadir_escala, entrenar, segmento
from prevision.calendario import MADRID, periodos_utc

CARPETA_MODELOS = Path(__file__).resolve().parent.parent / "modelos"
VERSION = "v1"
DESDE_TRAIN = date(2023, 1, 8)
HUBER = {"objective": "huber", "alpha": 15}
PESO_LEAR = 0.15
# Variables que se descartan (no aportan en el backtest: experimentos/NOTAS.md, ronda 4)
EXCLUIR = ("co2",)
# Variables que dependen de la demanda prevista de ENTSO-E para E (puede no estar publicada aún)
SOLO_CON_EC = ("ec_demanda_E", "ec_demanda_dif")
# Configuración elegida en el backtest (ver experimentos/NOTAS.md)
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


def construir_variables(dias: list[date], datos: dict) -> pd.DataFrame:
    X = variables.construir(dias, datos)
    X = variables.anadir_extras(X, datos)
    X = variables.anadir_extras2(X, datos)
    X = variables.anadir_puntos(X, datos)
    return variables.anadir_ec(X, datos)


def entrenar_y_guardar():
    historico.actualizar()
    datos_extra.actualizar()
    datos = variables.cargar_extras2(variables.cargar())
    ultimo = max(datos["precios"].index.tz_convert(MADRID).date)
    dias = [DESDE_TRAIN + timedelta(days=i) for i in range((ultimo - DESDE_TRAIN).days + 1)]
    X = anadir_escala(construir_variables(dias, datos), datos["precios"])
    X = X[X["objetivo"].notna()]

    carpeta = CARPETA_MODELOS / VERSION
    carpeta.mkdir(parents=True, exist_ok=True)
    for viejo in list(carpeta.glob("*.txt*")) + list(carpeta.glob("*.joblib")):
        viejo.unlink()
    meta = {"desde_train": str(DESDE_TRAIN), "entrenado_hasta": str(ultimo), "filas": len(X),
            "peso_lear": PESO_LEAR, "juegos": {}}
    todas = [c for c in variables.columnas_modelo(X) if not c.startswith(EXCLUIR)]
    # Dos juegos: con la demanda prevista de ENTSO-E para E (se publica la mañana de D) y sin ella,
    # para los envíos en los que todavía no está publicada
    for juego, columnas in (("con_ec", todas), ("sin_ec", [c for c in todas if not c.startswith(SOLO_CON_EC)])):
        meta["juegos"][juego] = {"columnas": columnas, "variantes": {}}
        for nombre, kw in VARIANTES.items():
            kw = dict(kw)
            segmentar = kw.pop("segmentar", None)
            ventana = kw.pop("ventana_dias", None)
            params = {**HUBER, "random_state": 0, **kw.pop("params", {})}
            Xv = X[X["entrega"] >= ultimo - timedelta(days=ventana)] if ventana else X
            segmentos = segmento(Xv, segmentar)
            ficheros = {}
            for g in sorted(segmentos.unique()):
                m = entrenar(Xv[segmentos == g], params, ponderar=kw.get("ponderar"),
                             vida_media=kw.get("vida_media"), columnas=columnas)
                fichero = f"{juego}__{nombre}__{g}.txt.gz"
                _guardar(m.booster_, carpeta / fichero)
                ficheros[g] = fichero
            meta["juegos"][juego]["variantes"][nombre] = {"segmentar": segmentar, "ficheros": ficheros}
        print(f"  {juego}: {len(VARIANTES)} variantes, {len(columnas)} variables")

    F, Y = lear.matriz_diaria(X, datos["precios"])
    joblib.dump(lear.LEAR().fit(F, Y), carpeta / "lear.joblib", compress=3)
    meta["lear_columnas"] = list(F.columns)
    print("  lear: 24 modelos Lasso")
    (carpeta / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    print(f"Modelo {VERSION} entrenado con {len(X)} horas hasta {ultimo}")


def _intentar(avisos, nombre, funcion, *args, **kw):
    try:
        return funcion(*args, **kw)
    except Exception as e:  # noqa: BLE001 - una fuente secundaria caída no debe parar el envío
        avisos.append(f"{nombre} no disponible: {e}")
        return None


def datos_recientes(entrega: date) -> tuple[dict, list[str]]:
    """Solo lo necesario para predecir un día. Los precios son imprescindibles;
    si falla otra fuente se sigue sin ella (sus variables quedan vacías) y se avisa."""
    D = entrega - timedelta(days=1)
    avisos = []
    # 60 días de precios: variables (28 días) y corrección de sesgo (errores recientes)
    precios = historico.descargar_precios([D - timedelta(days=i) for i in range(60)])
    if D not in set(precios.index.tz_convert(MADRID).date):
        raise RuntimeError(f"Faltan los precios de {D}")
    datos = {"precios": precios, "meteo_modelos": {}, "meteo2_modelos": {}}

    desde, hasta = D - timedelta(days=1), entrega + timedelta(days=1)
    for modelo in historico.MODELOS_METEO:
        m = _intentar(avisos, f"Meteo {modelo}", historico.descargar_meteo, desde, hasta, modelo, rapido=True)
        if m is not None:
            datos["meteo_modelos"][modelo] = m
        m2 = _intentar(avisos, f"Meteo2 {modelo}", datos_extra.descargar_meteo2, desde, hasta, modelo, rapido=True)
        if m2 is not None:
            datos["meteo2_modelos"][modelo] = m2
    if not datos["meteo_modelos"]:
        raise RuntimeError("Ningún modelo meteorológico disponible")
    # Las variables base usan ECMWF; si no está, se sustituye por otro modelo
    if historico.MODELO_METEO in datos["meteo_modelos"]:
        datos["meteo"] = datos["meteo_modelos"][historico.MODELO_METEO]
    else:
        datos["meteo"] = next(iter(datos["meteo_modelos"].values()))
        avisos.append(f"Variables base calculadas con otro modelo meteorológico en lugar de {historico.MODELO_METEO}")

    gas = _intentar(avisos, "Gas TTF", historico.descargar_gas, D - timedelta(days=30))
    datos["gas"] = gas if gas is not None else pd.Series(dtype=float)
    for clave, nombre, funcion, args in (
            ("redata", "REData", historico.descargar_redata, (D - timedelta(days=15), D)),
            ("balance", "Balance REData", datos_extra.descargar_balance, (D - timedelta(days=15), D)),
            ("mibgas", "MIBGAS", datos_extra.descargar_mibgas, (sorted({(D - timedelta(days=8)).year, D.year}),)),
            ("ec", "Energy-Charts", lambda a, b: datos_extra.descargar_ec(a, b, rapido=True), (D - timedelta(days=2), entrega))):
        valor = _intentar(avisos, nombre, funcion, *args)
        if valor is not None:
            datos[clave] = valor
    return datos, avisos


def predecir(entrega: date, datos: dict | None = None, registro: pd.Series | None = None):
    """Devuelve (previsión corregida, avisos, previsión sin corregir) indexadas por inicio del periodo en UTC."""
    carpeta = CARPETA_MODELOS / VERSION
    meta = json.loads((carpeta / "meta.json").read_text())
    avisos = []
    if datos is None:
        datos, avisos = datos_recientes(entrega)
    X = construir_variables([entrega], datos)
    if X.empty:
        raise RuntimeError(f"No se pudieron construir las variables de {entrega}")
    # La meteo es imprescindible: si viene vacía, mejor caer al benchmark
    met = [c for c in X.columns if c.startswith("m_")]
    if X[met].isna().mean().mean() > 0.2:
        raise RuntimeError("Faltan demasiados datos meteorológicos")

    # 1) LightGBM: media de las variantes del juego que se pueda usar
    con_ec = "ec_demanda_E" in X and X["ec_demanda_E"].notna().all()
    juego = meta["juegos"]["con_ec" if con_ec else "sin_ec"]
    # Que aún no esté publicada es normal en los envíos tempranos: se informa, pero no se avisa por email
    print(f"Juego de modelos: {'con_ec' if con_ec else 'sin_ec (demanda prevista de ENTSO-E para E aún no publicada)'}")
    entradas = X.reindex(columns=juego["columnas"])
    predicciones = []
    for nombre, v in juego["variantes"].items():
        segmentos = segmento(X, v["segmentar"])
        p = pd.Series(np.nan, index=X.index)
        for g, fichero in v["ficheros"].items():
            filas = segmentos == g
            if filas.any():
                p[filas] = _cargar(carpeta / fichero).predict(entradas[filas])
        predicciones.append(p)
    pred = pd.concat(predicciones, axis=1).mean(axis=1) + X["p_D_h"].fillna(X["p_D_media"])

    # 2) LEAR (si falla, se sigue solo con LightGBM)
    if meta.get("peso_lear") and (carpeta / "lear.joblib").exists():
        try:
            F, _ = lear.matriz_diaria(X, datos["precios"])
            p_lear = lear.a_horario(pd.DataFrame(joblib.load(carpeta / "lear.joblib")
                                                 .predict(F.reindex(columns=meta["lear_columnas"])),
                                                 index=F.index, columns=range(24)), X)
            if p_lear.notna().all():
                pred = (1 - meta["peso_lear"]) * pred + meta["peso_lear"] * p_lear
            else:
                avisos.append("LEAR dio valores vacíos; se usa solo LightGBM")
        except Exception as e:  # noqa: BLE001
            avisos.append(f"LEAR falló ({e}); se usa solo LightGBM")
    cruda = pred.reindex(periodos_utc(entrega))

    # 3) Corrección del sesgo reciente con el registro de previsiones pasadas
    registro = sesgo.leer_registro() if registro is None else registro
    corregida, info = sesgo.aplicar(cruda, entrega, registro, datos["precios"])
    print(f"Sesgo: {info}")
    for a in avisos:
        print(f"AVISO: {a}", file=sys.stderr)
    return corregida, avisos, cruda


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("accion", choices=["entrenar", "predecir"])
    p.add_argument("--fecha", type=date.fromisoformat)
    a = p.parse_args()
    if a.accion == "entrenar":
        entrenar_y_guardar()
    else:
        print(predecir(a.fecha)[0].round(2).to_string())
