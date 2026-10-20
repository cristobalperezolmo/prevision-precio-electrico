# Previsión del precio eléctrico day-ahead en España

Modelo que predice el precio horario del mercado diario español (OMIE / ENTSO-E) para el día
siguiente y lo envía cada día a [Forecast Score](https://forecastscore.eu), que lo puntúa en un ranking público.

> Estado: **v0** (réplica del benchmark de la plataforma) para validar el envío de punta a punta.

## Uso

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env              # y rellenar las claves
python enviar_prevision.py --dry-run   # calcula y valida sin enviar
python enviar_prevision.py             # envía el próximo día de entrega abierto
python -m pytest -q
```

## Estructura

```
prevision/calendario.py     periodos del día (23/24/25), cutoff, festivos
prevision/omie.py           precios day-ahead de OMIE (cuartohorarios -> media horaria)
prevision/benchmark.py      benchmark oficial de la plataforma
prevision/validacion.py     comprobaciones antes de enviar
prevision/forecastscore.py  cliente de la API de envío
enviar_prevision.py         script diario (con respaldo si falla la descarga)
.github/workflows/          envío automático con GitHub Actions
```
