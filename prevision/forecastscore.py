"""Cliente mínimo de la API de envío de Forecast Score."""

import uuid
from datetime import date

import pandas as pd
import requests

URL_BASE = "https://ingest.forecastscore.eu/v1"


class ErrorForecastScore(Exception):
    pass


class Cliente:
    def __init__(self, api_key: str, timeout: int = 30):
        if not api_key:
            raise ErrorForecastScore("Falta FORECASTSCORE_API_KEY")
        self.sesion = requests.Session()
        self.sesion.headers["X-API-Key"] = api_key
        self.timeout = timeout

    def periodos(self, entrega: date, mercado="ES", resolucion="PT60M") -> dict:
        """Periodos, deadline y estado (open/closed/frozen) de un día de entrega."""
        r = self.sesion.get(f"{URL_BASE}/markets/{mercado}/delivery-periods",
                            params={"date": entrega.isoformat(), "resolution": resolucion},
                            timeout=self.timeout)
        _comprobar(r)
        return r.json()

    def enviar(self, modelo: str, entrega: date, precios: pd.Series,
               mercado="ES", resolucion="PT60M", reintentos=3) -> dict:
        """Envía una previsión usando "values" con el inicio de cada periodo en UTC
        (así no hay ambigüedad en los días de cambio de hora)."""
        cuerpo = {"forecasts": [{
            "model_code": modelo,
            "market": mercado,
            "resolution": resolucion,
            "delivery_date": entrega.isoformat(),
            "values": [{"period_start": t.strftime("%Y-%m-%dT%H:%M:%SZ"), "value": float(v)}
                       for t, v in precios.items()],
        }]}
        # Una clave nueva por envío; se reutiliza solo para reintentar ESTA misma petición
        clave = str(uuid.uuid4())
        ultimo_error = None
        for _ in range(reintentos):
            try:
                r = self.sesion.post(f"{URL_BASE}/forecasts", json=cuerpo, timeout=self.timeout,
                                     headers={"Idempotency-Key": clave})
            except requests.RequestException as e:  # timeout o red: se puede reintentar
                ultimo_error = e
                continue
            if r.status_code >= 500:
                ultimo_error = ErrorForecastScore(f"{r.status_code}: {r.text[:300]}")
                continue
            _comprobar(r)  # 4xx: reintentar no lo arregla
            return r.json()["forecasts"][0]
        raise ErrorForecastScore(f"Envío fallido tras {reintentos} intentos: {ultimo_error}")


def _comprobar(r: requests.Response):
    if not r.ok:
        raise ErrorForecastScore(f"{r.request.method} {r.url} -> {r.status_code}: {r.text[:500]}")
