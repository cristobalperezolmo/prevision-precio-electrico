"""Pruebas del v0, con atención especial a los días de cambio de hora."""

from datetime import date, datetime

import pandas as pd
import pytest

from prevision import benchmark, omie, validacion
from prevision.calendario import BRUSELAS, hora_local, periodos_utc, proxima_entrega_abierta

OCTUBRE = date(2026, 10, 25)  # 25 horas
MARZO = date(2027, 3, 28)     # 23 horas


def test_numero_de_periodos():
    assert len(periodos_utc(date(2026, 10, 22))) == 24
    assert len(periodos_utc(OCTUBRE)) == 25
    assert len(periodos_utc(MARZO)) == 23
    assert len(periodos_utc(OCTUBRE, 15)) == 100
    assert len(periodos_utc(MARZO, 15)) == 92


def test_dia_referencia():
    assert benchmark.dia_referencia(date(2026, 10, 22)) == date(2026, 10, 21)  # jueves -> miércoles
    assert benchmark.dia_referencia(date(2026, 10, 26)) == date(2026, 10, 19)  # lunes -> lunes anterior
    assert benchmark.dia_referencia(date(2026, 10, 24)) == date(2026, 10, 17)  # sábado
    assert benchmark.dia_referencia(date(2026, 12, 8)) == date(2026, 12, 6)    # festivo martes -> domingo
    assert benchmark.dia_referencia(date(2026, 12, 25)) == date(2026, 12, 20)  # Navidad (viernes)


def test_proxima_entrega():
    assert proxima_entrega_abierta(datetime(2026, 10, 20, 12, 44, tzinfo=BRUSELAS)) == date(2026, 10, 21)
    assert proxima_entrega_abierta(datetime(2026, 10, 20, 12, 45, tzinfo=BRUSELAS)) == date(2026, 10, 22)


def _fichero_falso(entrega: date, n: int) -> str:
    lineas = ["MARGINALPDBC;"]
    for p in range(1, n + 1):
        lineas.append(f"{entrega.year};{entrega.month:02d};{entrega.day:02d};{p};0;{p}.5;")
    return "\n".join(lineas + ["*"])


def test_leer_omie_cuartohorario_octubre():
    serie = omie.leer_fichero(_fichero_falso(OCTUBRE, 100), OCTUBRE)
    assert len(serie) == 25
    assert serie.iloc[0] == pytest.approx((1.5 + 2.5 + 3.5 + 4.5) / 4)
    # las dos 02:00 de reloj
    assert [hora_local(t) for t in serie.index[2:4]] == [2, 2]


def test_leer_omie_rechaza_numero_raro():
    with pytest.raises(ValueError):
        omie.leer_fichero(_fichero_falso(OCTUBRE, 96), OCTUBRE)


def test_benchmark_octubre_repite_la_0200():
    ref = date(2026, 10, 18)
    precios_ref = pd.Series(range(24), index=periodos_utc(ref), dtype=float)  # precio = hora
    prev = benchmark.copiar_por_hora_local(OCTUBRE, precios_ref)
    assert list(prev.values) == [0, 1, 2, 2] + list(range(3, 24))


def test_benchmark_marzo_y_referencia_de_23_horas():
    # Día de 23 horas copiando de uno normal: se salta la 02:00
    precios_ref = pd.Series(range(24), index=periodos_utc(date(2027, 3, 21)), dtype=float)
    prev = benchmark.copiar_por_hora_local(MARZO, precios_ref)
    assert list(prev.values) == [0, 1] + list(range(3, 24))
    # Día normal copiando del de 23 horas: la 02:00 se rellena con la media de 01 y 03
    precios_marzo = pd.Series(prev.values, index=periodos_utc(MARZO))
    siguiente = benchmark.copiar_por_hora_local(date(2027, 4, 4), precios_marzo)
    assert siguiente.iloc[2] == 2.0 and len(siguiente) == 24


def test_validacion():
    entrega = date(2026, 10, 22)
    buenos = pd.Series(50.123, index=periodos_utc(entrega))
    assert validacion.preparar(entrega, buenos).iloc[0] == 50.12
    with pytest.raises(ValueError):
        validacion.preparar(entrega, buenos.iloc[:23])
    with pytest.raises(ValueError):
        validacion.preparar(entrega, buenos.replace(50.123, 5000))
    with pytest.raises(ValueError):
        validacion.preparar(entrega, buenos.replace(50.123, float("nan")))


def _datos_sinteticos(desde: date, dias: int) -> dict:
    """Precios, meteo y gas inventados para probar la construcción de variables sin internet."""
    from datetime import timedelta
    import numpy as np
    from prevision.historico import PUNTOS, VARIABLES
    horas = pd.date_range(periodos_utc(desde)[0], periodos_utc(desde + timedelta(days=dias))[-1], freq="1h")
    precios = pd.Series(50 + 30 * np.sin(np.arange(len(horas)) / 24 * 2 * np.pi), index=horas)
    meteo = pd.DataFrame({f"{v}__{p}": 10.0 for v in VARIABLES for p in PUNTOS}, index=horas)
    gas = pd.Series(30.0, index=[desde + timedelta(days=i) for i in range(dias + 1)])
    from prevision.historico import MODELOS_METEO, TECNOLOGIAS
    redata = pd.DataFrame({c: 100000.0 for c in TECNOLOGIAS.values()},
                          index=[desde + timedelta(days=i) for i in range(dias + 1)])
    return {"precios": precios, "meteo": meteo, "gas": gas, "redata": redata,
            "meteo_modelos": {m: meteo for m in MODELOS_METEO}}


def test_variables_dias_de_cambio_de_hora():
    from prevision.variables import columnas_modelo, construir
    datos = _datos_sinteticos(date(2026, 9, 20), 40)
    X = construir([OCTUBRE, date(2026, 10, 22)], datos)
    assert list(X.groupby("entrega").size()) == [24, 25]
    assert X.loc[X["entrega"] == OCTUBRE, "hora"].tolist()[2:4] == [2, 2]
    assert not X[columnas_modelo(X)].isna().all().any()
    datos = _datos_sinteticos(date(2027, 3, 1), 40)
    assert len(construir([MARZO], datos)) == 23


def test_variables_extra_y_modelo_en_dias_de_cambio_de_hora():
    """El modelo guardado en modelos/ predice un valor válido para cada periodo, también con 25 y 23 horas."""
    from prevision import modelo
    for entrega, desde in ((OCTUBRE, date(2026, 9, 20)), (MARZO, date(2027, 2, 20))):
        prevision, avisos = modelo.predecir(entrega, _datos_sinteticos(desde, 45))
        precios = validacion.preparar(entrega, prevision)
        assert len(precios) == len(periodos_utc(entrega)) and avisos == []
