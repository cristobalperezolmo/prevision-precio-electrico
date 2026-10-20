"""
v0: calcula la previsión benchmark de un día de entrega y la envía a Forecast Score.

Uso:
    python enviar_prevision.py                    # próximo día de entrega abierto
    python enviar_prevision.py --fecha 2026-10-25 # un día concreto
    python enviar_prevision.py --dry-run          # calcula y valida, pero no envía

Respaldo: si no se pueden descargar los precios del día de referencia, se prueba con
otros días ya publicados (ayer, hace una semana...). Mejor enviar algo razonable que nada.
Si al final no se envía, el script termina con error para que GitHub Actions avise por email.
"""

import argparse
import os
import sys
from datetime import date, timedelta

from dotenv import load_dotenv

from prevision import benchmark, omie, validacion
from prevision.calendario import cutoff, periodos_utc, proxima_entrega_abierta
from prevision.forecastscore import Cliente

MODELO_POR_DEFECTO = "cris-es-da-v0"


def prevision_benchmark(entrega: date):
    """Devuelve (precios, día usado). Prueba primero el día de referencia oficial y luego respaldos."""
    referencia = benchmark.dia_referencia(entrega)
    candidatos = [referencia] + [entrega - timedelta(days=d) for d in (1, 7, 2, 14)]
    errores = []
    for dia in dict.fromkeys(candidatos):  # sin duplicados, manteniendo el orden
        try:
            precios = omie.descargar_dia(dia)
            return benchmark.copiar_por_hora_local(entrega, precios), dia
        except Exception as e:  # noqa: BLE001 - cualquier fallo: probar el siguiente día
            errores.append(f"{dia}: {e}")
            print(f"AVISO: no se pudo usar el día {dia} ({e})", file=sys.stderr)
    raise RuntimeError("Ningún día de referencia disponible:\n" + "\n".join(errores))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--fecha", type=date.fromisoformat, help="día de entrega (YYYY-MM-DD)")
    parser.add_argument("--modelo", default=os.getenv("MODEL_CODE", MODELO_POR_DEFECTO))
    parser.add_argument("--dry-run", action="store_true", help="no envía nada")
    args = parser.parse_args()

    load_dotenv()
    entrega = args.fecha or proxima_entrega_abierta()
    print(f"Día de entrega: {entrega} ({len(periodos_utc(entrega))} periodos), "
          f"cutoff {cutoff(entrega):%Y-%m-%d %H:%M %Z}, modelo {args.modelo}")

    api_key = os.getenv("FORECASTSCORE_API_KEY")
    cliente = Cliente(api_key) if api_key else None

    # 1. Comprobar con la plataforma que los periodos coinciden con nuestro calendario
    if cliente:
        info = cliente.periodos(entrega)
        inicios_api = [p["period_start"] for p in info["periods"]]
        nuestros = [t.strftime("%Y-%m-%dT%H:%M:%SZ") for t in periodos_utc(entrega)]
        print(f"Plataforma: estado={info['state']}, periodos={info['period_count']}, "
              f"deadline={info['submission_deadline']}")
        if [s.replace("+00:00", "Z") for s in inicios_api] != nuestros:
            raise SystemExit(f"Los periodos de la plataforma no coinciden con los nuestros:\n"
                             f"{inicios_api}\n{nuestros}")
        if info["state"] != "open" and not args.dry_run:
            raise SystemExit(f"El día {entrega} no está abierto (estado {info['state']})")
    elif not args.dry_run:
        raise SystemExit("Falta FORECASTSCORE_API_KEY (en .env o en los secretos de GitHub)")

    # 2. Calcular y validar
    precios, dia_usado = prevision_benchmark(entrega)
    precios = validacion.preparar(entrega, precios)
    oficial = benchmark.dia_referencia(entrega)
    print(f"Referencia: {dia_usado}" + ("" if dia_usado == oficial else f" (RESPALDO, la oficial era {oficial})"))
    print(precios.to_string())

    # 3. Enviar
    if args.dry_run:
        print("Dry run: no se envía nada.")
        return
    respuesta = cliente.enviar(args.modelo, entrega, precios)
    print(f"Enviado: id={respuesta['id']} versión={respuesta['version']} "
          f"estado={respuesta['status']} a las {respuesta['submitted_at']}")
    if dia_usado != oficial:
        raise SystemExit("Enviado con un día de respaldo: revisar por qué falló la referencia oficial")


if __name__ == "__main__":
    main()
