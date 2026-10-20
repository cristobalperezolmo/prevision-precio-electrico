# Previsión del precio eléctrico day-ahead en España

Modelo que predice el precio horario del mercado diario español (OMIE / ENTSO-E) para el día
siguiente y lo envía cada día a [Forecast Score](https://forecastscore.eu), que lo puntúa en un ranking público.

> Estado: **v1** (LightGBM) en producción. El v0 (réplica del benchmark) ya no se envía: solo se usa como respaldo si el v1 falla.

## Resultados del backtest (walk-forward, ene-2025 a oct-2026)

Reentrenando cada mes con todo lo anterior y prediciendo con la información disponible antes del cutoff.
Las variantes se eligieron con 2025 y se confirmaron con 2026 (`experimentos/mejoras_v1.py`).

| Modelo | rMAE 2025 (ajuste) | rMAE 2026 (confirmación) |
|---|---|---|
| Benchmark (v0) | 1,00 | 1,00 |
| v1 inicial (un LightGBM) | 0,598 | 0,639 |
| **v1 actual (media de 4 LightGBM + fuentes nuevas)** | **0,563** | **0,600** |

Lo que funcionó: consenso de 3 modelos meteorológicos (ECMWF, GFS, ICON), generación real de REData
(nuclear, hidráulica…) y promediar variantes. Lo que no: un modelo por estación (pocos datos por estación),
pesos como la métrica de la plataforma, árboles más pequeños.

**Variables**: precios de D por hora (y de D-1, E-7, horas vecinas, medias de 7 y 28 días, horas a 0 €),
calendario y festivos, gas TTF del día hábil anterior, generación real por tecnología de D-1 (REData),
y previsiones meteorológicas (viento a 100 m, radiación, nubes y temperatura en 12 puntos) emitidas ~48 h
antes (Open-Meteo Previous Runs). El modelo predice la *diferencia* respecto al precio de D.

**Respaldos**: si cae GFS, ICON, REData o el gas, se predice sin esa fuente y se avisa por email;
si cae toda la meteorología o el modelo, se envía el benchmark.

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
modelos/v1/                 modelo entrenado (4 variantes, comprimidas)
experimentos/               comparación de variantes
enviar_prevision.py         script diario (con respaldo si falla la descarga)
.github/workflows/          envío automático con GitHub Actions
```
