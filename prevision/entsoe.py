"""
Cliente mínimo de la API de ENTSO-E Transparency (token en ENTSOE_API_KEY).

Precios day-ahead (documentType A44) de España. Ojo con el formato:
- puede haber varias series (TimeSeries) por día, con resolución PT15M o PT60M;
- con curveType A03 se omiten las posiciones cuyo precio es igual al anterior: hay que rellenarlas;
- desde las subastas intradía europeas (IDA) el documento A44 incluye también sus precios
  (contract_MarketAgreement.type = A07): se descartan, solo interesa el mercado diario (A01).
"""

import os
import xml.etree.ElementTree as ET
from datetime import date, timedelta

import pandas as pd
import requests

URL = "https://web-api.tp.entsoe.eu/api"
ES = "10YES-REE------0"


def _token() -> str:
    tok = os.getenv("ENTSOE_API_KEY", "").strip()
    if not tok:
        from dotenv import dotenv_values
        tok = (dotenv_values(".env").get("ENTSOE_API_KEY") or "").strip()
    if not tok:
        raise RuntimeError("Falta ENTSOE_API_KEY")
    return tok


def _consulta(params: dict) -> str:
    tok = _token()
    try:
        r = requests.get(URL, params={"securityToken": tok, **params}, timeout=90)
    except requests.RequestException as e:  # no mostrar la URL (lleva el token)
        raise RuntimeError(f"ENTSO-E no responde ({type(e).__name__})") from None
    if r.status_code != 200:
        raise RuntimeError(f"ENTSO-E respondió {r.status_code}: {r.text.replace(tok, '***')[:200]}")
    return r.text


def _series(xml: str) -> list[pd.Series]:
    """Convierte un documento de ENTSO-E en series (una por TimeSeries/Period), índice UTC."""
    raiz = ET.fromstring(xml)
    ns = {"n": raiz.tag.split("}")[0].strip("{")}
    salida = []
    for ts in raiz.findall("n:TimeSeries", ns):
        # A01 = subasta del mercado diario; A07 = subastas intradía (IDA), que vienen en el mismo documento
        if ts.findtext("n:contract_MarketAgreement.type", default="A01", namespaces=ns) != "A01":
            continue
        curva = ts.findtext("n:curveType", default="A01", namespaces=ns)
        for periodo in ts.findall("n:Period", ns):
            inicio = pd.Timestamp(periodo.findtext("n:timeInterval/n:start", namespaces=ns))
            fin = pd.Timestamp(periodo.findtext("n:timeInterval/n:end", namespaces=ns))
            paso = pd.Timedelta(periodo.findtext("n:resolution", namespaces=ns))
            puntos = {int(p.findtext("n:position", namespaces=ns)): float(p.findtext("n:price.amount", namespaces=ns))
                      for p in periodo.findall("n:Point", ns)}
            n = int((fin - inicio) / paso)
            serie = pd.Series([puntos.get(i) for i in range(1, n + 1)],
                              index=pd.date_range(inicio, periods=n, freq=paso), dtype=float)
            if curva == "A03":  # posiciones omitidas = mismo precio que la anterior
                serie = serie.ffill()
            serie.attrs["resolucion"] = str(paso)
            salida.append(serie)
    return salida


def precios_day_ahead(desde: date, hasta: date) -> dict[str, pd.Series]:
    """Precios day-ahead de España por resolución {'0 days 00:15:00': serie, '0 days 01:00:00': serie}."""
    a = pd.Timestamp(desde).tz_localize("Europe/Madrid").tz_convert("UTC")
    b = pd.Timestamp(hasta + timedelta(days=1)).tz_localize("Europe/Madrid").tz_convert("UTC")
    xml = _consulta({"documentType": "A44", "in_Domain": ES, "out_Domain": ES,
                     "periodStart": a.strftime("%Y%m%d%H%M"), "periodEnd": b.strftime("%Y%m%d%H%M")})
    por_res: dict[str, list] = {}
    for s in _series(xml):
        por_res.setdefault(s.attrs["resolucion"], []).append(s)
    return {k: pd.concat(v).sort_index().pipe(lambda s: s[~s.index.duplicated()]) for k, v in por_res.items()}
