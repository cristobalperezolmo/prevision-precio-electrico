"""
Archivo "tal como estaba": guarda los datos que usó cada envío, en el momento del envío.

El histórico de previsiones meteorológicas que podemos descargar (Open-Meteo Previous Runs) solo garantiza
previsiones de ~48 h antes. En producción, a la hora del envío ya hay previsiones más recientes (mejores).
Guardando cada día lo que había de verdad, dentro de unos meses podremos reentrenar con datos más frescos
y medidos sin trampa.

Se guarda un fichero por día de entrega en archivo/<YYYY-MM-DD>.parquet (formato largo:
fuente, variable, instante, valor), con la hora del envío. Si hay varios envíos para el mismo día,
el último sustituye al anterior: queda justo lo que se puntuó. El workflow lo sube a la rama `archivo`.
"""

from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from prevision.calendario import periodos_utc

CARPETA = Path(__file__).resolve().parent.parent / "archivo"


def _largo(df: pd.DataFrame, fuente: str, desde, hasta) -> pd.DataFrame:
    df = df[(df.index >= desde) & (df.index < hasta)]
    largo = df.reset_index(names="instante").melt(id_vars="instante", var_name="variable", value_name="valor")
    largo.insert(0, "fuente", fuente)
    return largo.dropna(subset=["valor"])


def guardar(entrega: date, datos: dict, momento: datetime | None = None) -> Path:
    """Guarda las previsiones (meteo de cada modelo, ENTSO-E y e·sios) para las horas de D y de E."""
    momento = momento or datetime.now(timezone.utc)
    desde = periodos_utc(entrega - timedelta(days=1))[0]
    hasta = periodos_utc(entrega)[-1] + pd.Timedelta(hours=1)
    trozos = []
    for clave in ("meteo_modelos", "meteo2_modelos"):
        for modelo, df in datos.get(clave, {}).items():
            trozos.append(_largo(df, f"{clave.split('_')[0]}:{modelo}", desde, hasta))
    if "ec" in datos:
        trozos.append(_largo(datos["ec"], "energy-charts", desde, hasta))
    if "esios" in datos:
        trozos.append(_largo(datos["esios"], "esios", desde, hasta))
    for nombre in ("gas", "mibgas"):
        serie = datos.get(nombre)
        if serie is not None and len(serie):
            trozos.append(pd.DataFrame({"fuente": [nombre], "variable": ["ultimo"],
                                        "instante": [pd.Timestamp(serie.index[-1]).tz_localize("UTC")
                                                     if pd.Timestamp(serie.index[-1]).tzinfo is None
                                                     else pd.Timestamp(serie.index[-1])],
                                        "valor": [float(serie.iloc[-1])]}))
    tabla = pd.concat(trozos, ignore_index=True)
    tabla["valor"] = tabla["valor"].astype("float32")
    tabla["guardado_utc"] = pd.Timestamp(momento).tz_convert("UTC") if pd.Timestamp(momento).tzinfo \
        else pd.Timestamp(momento).tz_localize("UTC")
    CARPETA.mkdir(exist_ok=True)
    ruta = CARPETA / f"{entrega.isoformat()}.parquet"
    tabla.to_parquet(ruta, index=False, compression="zstd")
    return ruta
