# Previsión del precio eléctrico day-ahead en España

Modelo que predice el precio horario del mercado diario español (OMIE / ENTSO-E) para el día
siguiente y lo envía cada día a [Forecast Score](https://forecastscore.eu), que lo puntúa en un ranking público.

> Estado: **v1.2** — media de 4 LightGBM + LEAR, reentrenado cada semana, con corrección del sesgo reciente.
> El v0 (réplica del benchmark) no se envía: solo es el respaldo si el v1 falla.

## Resultados del backtest (walk-forward, ene-2025 a oct-2026)

Prediciendo cada día solo con la información disponible antes del cutoff. Las decisiones se tomaron con
2025 y se confirmaron con 2026. Detalle de todos los experimentos en [experimentos/NOTAS.md](experimentos/NOTAS.md).

| Modelo | MAE total (€/MWh) | rMAE 2025 | rMAE 2026 |
|---|---|---|---|
| Benchmark de la plataforma (v0) | 22,1 | 1,00 | 1,00 |
| v1 inicial (un LightGBM, reentreno mensual) | 13,6 | 0,598 | 0,639 |
| v1.1 (media de 4 LightGBM + GFS/ICON/REData, en producción) | 12,8 | 0,563 | 0,600 |
| **v1.2** (fuentes nuevas, ENTSO-E, LEAR, semanal, sesgo) | **11,4** | **0,511** | **0,521** |

El v1.2 bate al benchmark todos los meses (rMAE mensual entre 0,38 y 0,75). A 15 minutos: rMAE 0,527.

**Qué aportó (de más a menos)**: la demanda prevista oficial (ENTSO-E vía Energy-Charts), el reentrenamiento
semanal, los índices de viento/sol en 24 puntos con 3 modelos meteorológicos, la corrección del sesgo reciente,
LEAR, el gas español (MIBGAS) y promediar variantes. **Qué no**: modelos por estación, CO2, ventanas cortas,
redes neuronales (mejora mínima), perfil completo de 24 h de D (inestable).

## Cómo funciona el v1.2

1. **Datos** (todo gratuito y sin clave): precios de OMIE, previsiones meteorológicas emitidas ~48 h antes
   (Open-Meteo Previous Runs, ECMWF/GFS/ICON, 36 puntos), generación real de D-1 (REData), gas TTF y MIBGAS,
   y previsiones oficiales de ENTSO-E vía Energy-Charts (demanda prevista para E; eólica/solar previstas para D).
2. **Variables** sin fugas de datos (`prevision/variables.py`): precios de D por hora y resúmenes, calendario,
   meteo de E y su cambio respecto a D, factor de carga eólico y radiación por punto, etc.
3. **Modelo**: 85 % media de 4 LightGBM (predicen la diferencia con el precio de D) + 15 % LEAR (un Lasso por hora).
   Hay dos juegos de LightGBM: con la demanda prevista de ENTSO-E para E y sin ella (si aún no está publicada).
4. **Corrección de sesgo**: se resta el 70 % de la media exponencial (vida media 14 días) del error pasado a
   cada hora, calculado con el registro de previsiones enviadas (`registro/predicciones.csv`).
5. **Respaldos**: si cae una fuente secundaria, se predice sin ella y se avisa; si cae toda la meteo o el modelo,
   se envía el benchmark. El envío diario corre 5 veces al día.
6. **Reentrenamiento semanal** (lunes): el modelo se publica en la rama `modelos` (un único commit que se sobrescribe).

## Uso

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env              # y rellenar las claves
python enviar_prevision.py --dry-run   # calcula y valida sin enviar
python enviar_prevision.py             # envía el próximo día de entrega abierto
python -m pytest -q
python -m prevision.historico          # descarga el histórico a data/
python -m prevision.backtest           # backtest walk-forward
python -m prevision.modelo entrenar    # entrena y guarda modelos/v1
```

## Estructura

```
prevision/calendario.py     periodos del día (23/24/25), cutoff, festivos
prevision/omie.py           precios day-ahead de OMIE (cuartohorarios -> media horaria)
prevision/benchmark.py      benchmark oficial de la plataforma
prevision/validacion.py     comprobaciones antes de enviar
prevision/forecastscore.py  cliente de la API de envío
prevision/historico.py      descarga de histórico (OMIE, Open-Meteo, TTF)
prevision/variables.py      construcción de variables sin fugas de datos
prevision/backtest.py       backtest walk-forward y métricas
prevision/modelo.py         entrenamiento y predicción del v1
modelos/v1/                 modelo de respaldo (el semanal está en la rama `modelos`)
registro/                   previsiones enviadas sin corregir (para la corrección de sesgo)
prevision/lear.py           modelo LEAR
prevision/sesgo.py          corrección del sesgo reciente
prevision/datos_extra.py    MIBGAS, balance REData, más puntos meteo, Energy-Charts
prevision/cuartos.py        previsión a 15 minutos
experimentos/               comparación de variantes
enviar_prevision.py         script diario (con respaldo si falla la descarga)
.github/workflows/          envío automático con GitHub Actions
```
