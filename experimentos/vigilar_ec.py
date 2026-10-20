"""Apunta a qué hora (UTC) aparecen en Energy-Charts las previsiones day-ahead de España para un día
y si cambian después. Uso: python -m experimentos.vigilar_ec 2026-10-22"""
import sys, time, datetime as dt
import pandas as pd, requests
from prevision.historico import CARPETA

dia = sys.argv[1]
fin = dt.datetime.fromisoformat(dia).replace(tzinfo=dt.timezone.utc)  # parar a las 00 UTC del día de entrega
visto = {}
while dt.datetime.now(dt.timezone.utc) < fin:
    ahora = dt.datetime.now(dt.timezone.utc)
    for tipo in ("load", "wind_onshore", "solar"):
        try:
            j = requests.get("https://api.energy-charts.info/public_power_forecast", timeout=60, params={
                "country": "es", "production_type": tipo, "forecast_type": "day-ahead", "start": dia, "end": dia}).json()
            s = pd.Series(j["forecast_values"], index=pd.to_datetime(j["unix_seconds"], unit="s", utc=True), dtype=float)
            s = s[s.index.tz_convert("Europe/Madrid").date == pd.Timestamp(dia).date()]
            n = int(s.notna().sum())
            if n and tipo not in visto:
                visto[tipo] = ahora
                print(f"{ahora:%Y-%m-%d %H:%M} UTC: aparece {tipo} para {dia} ({n} valores, media {s.mean():.0f})", flush=True)
            if n:
                s.to_frame(tipo).to_parquet(CARPETA / "snapshots" / f"ec_{tipo}_{dia}_{ahora:%Y%m%dT%H%M}.parquet")
        except Exception as e:  # noqa: BLE001
            print(f"{ahora:%H:%M} error {tipo}: {e}", flush=True)
        time.sleep(5)
    time.sleep(600)
print("Resumen:", {k: f"{v:%Y-%m-%d %H:%M} UTC" for k, v in visto.items()}, flush=True)
