"""Pruebas del v0, con atención especial a los días de cambio de hora."""

from datetime import date, datetime

import numpy as np
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
    from prevision.datos_extra import BALANCE, PUNTOS_EOLICOS, PUNTOS_SOLARES
    dias_idx = [desde + timedelta(days=i) for i in range(dias + 1)]
    meteo2 = pd.DataFrame({**{f"wind_speed_100m__{k}": 20.0 for k in PUNTOS_EOLICOS},
                           **{f"shortwave_radiation__{k}": 200.0 for k in PUNTOS_SOLARES}}, index=horas)
    return {"precios": precios, "meteo": meteo, "gas": gas, "redata": redata,
            "meteo_modelos": {m: meteo for m in MODELOS_METEO},
            "meteo2_modelos": {m: meteo2 for m in MODELOS_METEO},
            "mibgas": pd.Series(30.0, index=dias_idx), "co2": pd.Series(70.0, index=dias_idx),
            "balance": pd.DataFrame({c: 1000.0 for c in BALANCE.values()}, index=dias_idx),
            "ec": pd.DataFrame({"ec_solar": 5000.0, "ec_eolica": 7000.0, "ec_demanda": 27000.0}, index=horas),
            "esios": pd.DataFrame({"esios_eolica": 7000.0, "esios_fv": 5000.0, "esios_demanda": 27000.0},
                                  index=horas)}


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
        prevision, avisos, cruda = modelo.predecir(entrega, _datos_sinteticos(desde, 45),
                                                   registro=pd.Series(dtype=float))
        precios = validacion.preparar(entrega, prevision)
        assert len(precios) == len(periodos_utc(entrega)) and avisos == []
        assert prevision.equals(cruda)  # sin registro no hay corrección de sesgo


def test_cuartos_de_hora_dias_de_cambio_de_hora():
    """La previsión a 15 min tiene 96/100/92 periodos y respeta la media de cada hora."""
    from prevision import cuartos
    for entrega, n in ((date(2026, 10, 22), 96), (OCTUBRE, 100), (MARZO, 92)):
        horaria = pd.Series(50.0 + np.arange(len(periodos_utc(entrega))), index=periodos_utc(entrega))
        # forma inventada: cada hora sube 4 €/MWh de un cuarto al siguiente
        dias_prev = [entrega - pd.Timedelta(days=i) for i in range(1, 8)]
        q = pd.concat([pd.Series(np.tile([-6.0, -2.0, 2.0, 6.0], len(periodos_utc(d, 15)) // 4),
                                 index=periodos_utc(d, 15)) for d in dias_prev])
        p = validacion.preparar(entrega, cuartos.a_cuartos(horaria, entrega, q), 15)
        assert len(p) == n
        assert p.resample("1h").mean().round(6).equals(horaria.round(6))
        assert list(p.iloc[:4]) == [44.0, 48.0, 52.0, 56.0]
        sin_forma = cuartos.a_cuartos(horaria, entrega, None)
        assert (sin_forma.resample("1h").mean() == horaria).all()


def test_correccion_de_sesgo():
    """Si el modelo se pasó 10 €/MWh todos los días a las 08:00, la corrección resta el 70 % de eso."""
    from prevision import sesgo
    dias = [date(2026, 9, 1) + pd.Timedelta(days=i) for i in range(30)]
    reales = pd.concat([pd.Series(50.0, index=periodos_utc(d)) for d in dias])
    registro = reales.copy()
    registro[registro.index.tz_convert("Europe/Madrid").hour == 8] += 10
    entrega = date(2026, 10, 1)
    prevision = pd.Series(60.0, index=periodos_utc(entrega))
    corregida, _ = sesgo.aplicar(prevision, entrega, registro, reales)
    horas = corregida.index.tz_convert("Europe/Madrid").hour
    assert corregida[horas == 8].iloc[0] == pytest.approx(60 - 0.7 * 10)
    assert (corregida[horas != 8] == 60).all()
    # con pocos días en el registro no se corrige
    assert sesgo.aplicar(prevision, entrega, registro.iloc[:48], reales)[0].equals(prevision)


def test_modelo_sin_demanda_prevista_usa_el_otro_juego():
    """Si la demanda prevista de ENTSO-E para E no está publicada, se usa el juego de modelos sin ella."""
    from prevision import modelo
    datos = _datos_sinteticos(date(2026, 9, 20), 45)
    datos["ec"] = datos["ec"][datos["ec"].index < periodos_utc(date(2026, 10, 22))[0]]
    prevision, avisos, _ = modelo.predecir(date(2026, 10, 22), datos, registro=pd.Series(dtype=float))
    assert len(validacion.preparar(date(2026, 10, 22), prevision)) == 24
    assert avisos == []  # no es un fallo: solo se usa el otro juego
    assert modelo.ULTIMA_SOMBRA is None


def test_modelo_con_y_sin_esios():
    """Con las previsiones de e·sios para E se usa su juego y se calcula la sombra (v1.2);
    si aún no están publicadas, se usa el v1.2 sin avisar."""
    from prevision import modelo
    entrega = date(2026, 10, 22)
    datos = _datos_sinteticos(date(2026, 9, 20), 45)
    prevision, avisos, cruda = modelo.predecir(entrega, datos, registro=pd.Series(dtype=float))
    assert avisos == [] and modelo.ULTIMA_SOMBRA is not None
    assert modelo.ULTIMA_SOMBRA.index.equals(cruda.index) and modelo.ULTIMA_SOMBRA.notna().all()
    datos["esios"] = datos["esios"][datos["esios"].index < periodos_utc(entrega)[0]]
    prevision2, avisos, _ = modelo.predecir(entrega, datos, registro=pd.Series(dtype=float))
    assert avisos == [] and modelo.ULTIMA_SOMBRA is None
    assert len(validacion.preparar(entrega, prevision2)) == 24


def test_archivo_de_datos(tmp_path, monkeypatch):
    """Se guardan en formato largo las previsiones de D y E, con la hora del guardado."""
    from prevision import archivo
    monkeypatch.setattr(archivo, "CARPETA", tmp_path)
    datos = _datos_sinteticos(date(2026, 9, 20), 45)
    ruta = archivo.guardar(date(2026, 10, 25), datos)
    t = pd.read_parquet(ruta)
    assert set(t["fuente"]) >= {"meteo:ecmwf_ifs025", "meteo2:icon_seamless", "energy-charts", "esios", "gas", "mibgas"}
    horas = t.loc[t["fuente"] == "energy-charts", "instante"].nunique()
    assert horas == 24 + 25  # D (24 h) y E (25 h, cambio de hora)
    assert t["guardado_utc"].notna().all()
