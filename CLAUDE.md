# Proyecto: modelo de previsión del precio eléctrico day-ahead (España) para Forecast Score

## Objetivo
Construir un modelo que prediga el precio horario del mercado diario (day-ahead) de España para el día siguiente,
y enviarlo automáticamente cada día a la plataforma Forecast Score (https://forecastscore.eu), que lo puntúa
contra los precios oficiales de ENTSO-E y lo clasifica en un ranking público.

Prioridades, en este orden:
1. **Fiabilidad**: enviar una previsión válida TODOS los días. Un día sin envío puntúa 0 en el ranking semanal/mensual.
2. **Batir al benchmark** de la plataforma (rMAE < 1).
3. Mejorar la precisión (MAE, error en los peores periodos, sesgo).

## Reglas de la plataforma (no negociables)
- Mercado: `ES`. Resolución: `PT60M` (horaria) para empezar. Zona horaria del mercado: `Europe/Madrid`.
- **Cutoff: 12:45 Europe/Brussels del día anterior a la entrega.** Cuenta la hora de llegada de la petición. Después de esa hora el envío se rechaza (409).
- Número de precios por día: **24 normal, 23 el día del cambio de hora de marzo, 25 el de octubre** (las 02:00 se repiten; enviar ambas). Para 15 min: 96 / 92 / 100.
- Precios en €/MWh, entre −1000 y 4000, máximo 2 decimales (si no, error 422; la plataforma no redondea).
- Reenviar antes del cutoff crea una nueva versión; solo puntúa la última.
- Hasta 10 modelos por cuenta. Cada `model_code` distinto es una fila distinta en el ranking.

## API de envío
- Endpoint: `POST https://ingest.forecastscore.eu/v1/forecasts`
- Cabeceras: `X-API-Key: <clave>`, `Idempotency-Key: <UUID nuevo por cada cambio; reutilizar solo para reintentar la misma petición>`, `Content-Type: application/json`
- Cuerpo:
  ```json
  {"forecasts": [{
    "model_code": "cris-es-da-v1",
    "market": "ES",
    "resolution": "PT60M",
    "delivery_date": "YYYY-MM-DD",
    "prices": [24 números en orden desde las 00:00 hora local]
  }]}
  ```
  Alternativa a `prices`: `"values": [{"period_start": "<instante ISO 8601 UTC>", "value": 72.5}, ...]` (recomendado en días de cambio de hora).
- Comprobar periodos y cutoff de un día: `GET https://ingest.forecastscore.eu/v1/markets/ES/delivery-periods?date=YYYY-MM-DD&resolution=PT60M` (con `X-API-Key`). Devuelve `period_count`, `periods` (con `period_start` UTC), `submission_deadline` y `state`.
- Respuesta correcta: `201 Created`. Errores: 400 (Idempotency-Key), 401 (clave), 403 (términos no aceptados), 409 (cutoff pasado), 413 (>1 MiB), 422 (nº de precios, rango, decimales…).
- Cancelar: `DELETE https://ingest.forecastscore.eu/v1/forecasts/{id}` con las mismas cabeceras.
- Documentación: https://forecastscore.eu/docs/submit y https://forecastscore.eu/docs/rules

## Cómo puntúa la plataforma
- Cada error se normaliza por la variabilidad típica del precio en ese mercado y franja horaria.
- Forecast Score diario (0–100): 82,5 % error medio, 12,5 % percentil 90 del error, 5 % sesgo.
- Benchmark: martes a viernes = precio del día anterior a la misma hora; lunes, sábado y domingo = mismo día de la semana anterior; festivo nacional = último domingo no festivo.
- Métricas a replicar en el backtest local: MAE, RMSE, sesgo, MedAE, P90 del error absoluto, rMAE (MAE / MAE del benchmark), Skill Score (1 − rMAE).
- Documentación: https://forecastscore.eu/docs/scoring

## Información disponible en el momento de predecir (evitar fugas de datos)
A las ~11:30 del día D se predice D+1. Se conocen:
- Precios day-ahead de D y anteriores (los de D se publicaron el día D−1).
- Previsiones para D+1: demanda, generación eólica y solar (ENTSO-E / REE), meteorología (Open-Meteo).
- Precio del gas (MIBGAS / TTF) hasta D.
NO usar en el entrenamiento ningún dato que no estuviera disponible a esa hora (por ejemplo, la generación real de D+1).

## Fuentes de datos
- ENTSO-E Transparency Platform (token en la variable de entorno `ENTSOE_API_KEY`; librería `entsoe-py`): precios day-ahead ES, previsión de carga, previsión eólica y solar.
- Open-Meteo (gratis, sin clave): previsiones meteorológicas.
- REE e·sios (opcional, token propio).
- Festivos nacionales de España (librería `holidays`).
- Comprobar que los precios usados coinciden con los que la plataforma muestra en https://forecastscore.eu/market-data (desde el paso a periodos de 15 minutos, revisar cómo se obtiene el precio horario).

## Plan de trabajo
1. **v0 (benchmark)**: script que reproduce el benchmark de la plataforma y envía la previsión. Sirve para validar el envío de punta a punta.
2. **Datos**: descarga y limpieza de al menos 2–3 años de histórico; almacenamiento local (parquet).
3. **Backtest**: validación temporal tipo walk-forward simulando el cutoff real; comparar siempre contra el benchmark con las métricas de arriba.
4. **v1**: modelo de gradient boosting (LightGBM) con variables de precios rezagados, demanda, eólica, solar, meteorología, gas y calendario.
5. **Automatización**: GitHub Actions programado a diario antes del cutoff (secretos `FORECASTSCORE_API_KEY` y `ENTSOE_API_KEY`), con plan de respaldo: si falla la descarga de datos o el modelo, enviar la previsión del benchmark y notificar por email.
6. Reentrenamiento periódico (mensual) y mejoras sucesivas como modelos nuevos (`-v2`, `-v3`…).

## Convenciones
- Python 3.11+, entorno virtual, dependencias en `requirements.txt`.
- Claves SIEMPRE en variables de entorno o archivo `.env` (incluido en `.gitignore`). Nunca en el código ni en el repositorio.
- Todas las fechas internas en UTC; convertir a `Europe/Madrid` solo para el calendario del mercado.
- Probar explícitamente los días de cambio de hora (23 y 25 periodos).
- Antes de enviar, validar: número de periodos, rango de precios, 2 decimales, sin NaN.
- Explicaciones y comentarios en español.
