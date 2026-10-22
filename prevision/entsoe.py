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


FR = "10YFR-RTE------C"
PT = "10YPT-REN------W"


def _rango(desde: date, hasta: date) -> dict:
    a = pd.Timestamp(desde).tz_localize("Europe/Madrid").tz_convert("UTC")
    b = pd.Timestamp(hasta + timedelta(days=1)).tz_localize("Europe/Madrid").tz_convert("UTC")
    return {"periodStart": a.strftime("%Y%m%d%H%M"), "periodEnd": b.strftime("%Y%m%d%H%M")}


def _series_valor(xml: str, etiqueta: str) -> pd.Series:
    """Como _series pero para documentos con <quantity> u otra etiqueta de valor; une todas las series."""
    raiz = ET.fromstring(xml)
    ns = {"n": raiz.tag.split("}")[0].strip("{")}
    trozos = []
    for ts in raiz.findall("n:TimeSeries", ns):
        curva = ts.findtext("n:curveType", default="A01", namespaces=ns)
        for periodo in ts.findall("n:Period", ns):
            inicio = pd.Timestamp(periodo.findtext("n:timeInterval/n:start", namespaces=ns))
            fin = pd.Timestamp(periodo.findtext("n:timeInterval/n:end", namespaces=ns))
            paso = pd.Timedelta(periodo.findtext("n:resolution", namespaces=ns).replace("P7D", "P7DT0H"))
            puntos = {int(p.findtext("n:position", namespaces=ns)): float(p.findtext(f"n:{etiqueta}", namespaces=ns))
                      for p in periodo.findall("n:Point", ns)}
            n = max(1, int((fin - inicio) / paso))
            s = pd.Series([puntos.get(i) for i in range(1, n + 1)],
                          index=pd.date_range(inicio, periods=n, freq=paso), dtype=float)
            trozos.append(s.ffill() if curva == "A03" else s)
    if not trozos:
        return pd.Series(dtype=float)
    s = pd.concat(trozos).sort_index()
    return s[~s.index.duplicated(keep="last")]


def precio_zona(zona: str, desde: date, hasta: date) -> pd.Series:
    """Precio day-ahead horario (media de cuartos si es cuartohoraria) de otra zona, índice UTC."""
    xml = _consulta({"documentType": "A44", "in_Domain": zona, "out_Domain": zona, **_rango(desde, hasta)})
    trozos = [s.resample("1h").mean() for s in _series(xml)]
    s = pd.concat(trozos).sort_index()
    return s[~s.index.duplicated(keep="last")]


def capacidad_prevista(origen: str, destino: str, desde: date, hasta: date) -> pd.Series:
    """Capacidad de intercambio prevista day-ahead (A61, MW) de origen a destino, horaria."""
    xml = _consulta({"documentType": "A61", "contract_MarketAgreement.Type": "A01",
                     "in_Domain": destino, "out_Domain": origen, **_rango(desde, hasta)})
    return _series_valor(xml, "quantity").resample("1h").mean()


def embalses(desde: date, hasta: date) -> pd.Series:
    """Energía almacenada en embalses (A72, MWh), semanal."""
    xml = _consulta({"documentType": "A72", "processType": "A16", "in_Domain": ES, **_rango(desde, hasta)})
    return _series_valor(xml, "quantity")


# --- Indisponibilidad de unidades de generación (A80): paradas programadas (A53) y averías (A54) ---
TIPOS_CENTRAL = {"B14": "nuclear", "B04": "gas", "B05": "carbon", "B10": "bombeo", "B11": "hidraulica",
                 "B12": "hidraulica"}


def _leer_indisponibilidad(xml: str) -> list[dict]:
    """Un aviso → filas con los tramos de potencia disponible (MW) de la unidad."""
    raiz = ET.fromstring(xml)
    ns = {"n": raiz.tag.split("}")[0].strip("{")}
    base = {"mrid": raiz.findtext("n:mRID", namespaces=ns),
            "revision": int(raiz.findtext("n:revisionNumber", default="1", namespaces=ns)),
            "creado": pd.Timestamp(raiz.findtext("n:createdDateTime", namespaces=ns).strip()),
            "estado": raiz.findtext("n:docStatus/n:value", default="", namespaces=ns)}
    filas = []
    for ts in raiz.findall("n:TimeSeries", ns):
        psr = ts.findtext("n:production_RegisteredResource.pSRType.psrType", namespaces=ns)
        nominal = ts.findtext("n:production_RegisteredResource.pSRType.powerSystemResources.nominalP", namespaces=ns)
        comun = {**base, "tipo_aviso": ts.findtext("n:businessType", namespaces=ns), "psr": psr,
                 "unidad": ts.findtext("n:production_RegisteredResource.pSRType.powerSystemResources.name", namespaces=ns),
                 "nominal": float(nominal) if nominal else float("nan")}
        for periodo in ts.findall("n:Available_Period", ns):
            inicio = pd.Timestamp(periodo.findtext("n:timeInterval/n:start", namespaces=ns))
            fin = pd.Timestamp(periodo.findtext("n:timeInterval/n:end", namespaces=ns))
            paso = pd.Timedelta(periodo.findtext("n:resolution", namespaces=ns))
            puntos = sorted((int(p.findtext("n:position", namespaces=ns)), float(p.findtext("n:quantity", namespaces=ns)))
                            for p in periodo.findall("n:Point", ns))
            for i, (pos, q) in enumerate(puntos):
                a = inicio + (pos - 1) * paso
                b = inicio + (puntos[i + 1][0] - 1) * paso if i + 1 < len(puntos) else fin
                filas.append({**comun, "inicio": a, "fin": b, "disponible": q})
    return filas


def indisponibilidades(desde: date, hasta: date, tipo: str = "A53") -> pd.DataFrame:
    """Todos los avisos (última versión) que afectan a [desde, hasta], paginando de 200 en 200."""
    import io
    import time
    import zipfile
    tok = _token()
    filas, offset = [], 0
    while True:
        try:
            r = requests.get(URL, timeout=180, params={"securityToken": tok, "documentType": "A80",
                                                       "biddingZone_Domain": ES, "businessType": tipo,
                                                       "offset": offset, **_rango(desde, hasta)})
        except requests.RequestException as e:
            raise RuntimeError(f"ENTSO-E no responde ({type(e).__name__})") from None
        if r.content[:2] != b"PK":
            if "No matching data" in r.text:
                break
            raise RuntimeError(f"ENTSO-E respondió {r.status_code}: {r.text.replace(tok, '***')[:300]}")
        z = zipfile.ZipFile(io.BytesIO(r.content))
        nombres = z.namelist()
        for n in nombres:
            filas += _leer_indisponibilidad(z.read(n).decode())
        if len(nombres) < 200:
            break
        offset += 200
        time.sleep(0.5)
    return pd.DataFrame(filas)
