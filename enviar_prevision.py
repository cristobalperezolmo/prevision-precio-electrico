"""
Calcula la previsión de un día de entrega y la envía a Forecast Score.

Modelos:
    v1 (por defecto): LightGBM con precios, meteorología y gas -> cris-es-da-v1
    v0: réplica del benchmark de la plataforma           -> cris-es-da-v0

Uso:
    python enviar_prevision.py                    # v1, próximo día de entrega abierto
    python enviar_prevision.py --version v0       # benchmark
    python enviar_prevision.py --fecha 2026-10-25 # un día concreto
    python enviar_prevision.py --dry-run          # calcula y valida, pero no envía
    python enviar_prevision.py --resolucion PT15M # previsión a 15 minutos (96/92/100 periodos)

Respaldo: si el v1 falla (datos o modelo), se envía el benchmark con el código del v1.
Si no se pueden descargar los precios del día de referencia, se prueba con otros días ya
publicados. Mejor enviar algo razonable que nada. Cuando se usa un respaldo, o no se envía,
el script termina con error para que GitHub Actions avise por email.
"""

import argparse
import os
import sys
from datetime import date, timedelta

from dotenv import load_dotenv

from prevision import benchmark, omie, validacion
from prevision.calendario import cutoff, periodos_utc, proxima_entrega_abierta
from prevision.forecastscore import Cliente

CODIGOS = {"v0": "cris-es-da-v0", "v1": "cris-es-da-v1"}


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
    parser.add_argument("--version", choices=list(CODIGOS), default="v1")
    parser.add_argument("--resolucion", choices=["PT60M", "PT15M"], default="PT60M")
    parser.add_argument("--dry-run", action="store_true", help="no envía nada")
    args = parser.parse_args()
    minutos = 15 if args.resolucion == "PT15M" else 60

    load_dotenv()
    entrega = args.fecha or proxima_entrega_abierta()
    args.modelo = CODIGOS[args.version]
    print(f"Día de entrega: {entrega} ({len(periodos_utc(entrega, minutos))} periodos {args.resolucion}), "
          f"cutoff {cutoff(entrega):%Y-%m-%d %H:%M %Z}, modelo {args.modelo}")

    api_key = os.getenv("FORECASTSCORE_API_KEY")
    cliente = Cliente(api_key) if api_key else None

    # 1. Comprobar con la plataforma que los periodos coinciden con nuestro calendario
    if cliente:
        info = cliente.periodos(entrega, resolucion=args.resolucion)
        inicios_api = [p["period_start"] for p in info["periods"]]
        nuestros = [t.strftime("%Y-%m-%dT%H:%M:%SZ") for t in periodos_utc(entrega, minutos)]
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
    avisos = []
    precios = None
    cruda = None  # previsión del v1 sin corregir el sesgo: se guarda en el registro
    if args.version == "v1":
        try:
            from prevision import modelo
            prevision, avisos_modelo, cruda = modelo.predecir(entrega)
            precios = validacion.preparar(entrega, prevision)  # la horaria se valida siempre
            avisos += avisos_modelo  # fuentes secundarias caídas: se envía igual, pero se avisa
            print("Previsión del modelo v1 lista")
        except Exception as e:  # noqa: BLE001 - cualquier fallo: caer al benchmark
            avisos.append(f"El modelo v1 falló ({type(e).__name__}: {e}); se envía el benchmark")
            print(f"AVISO: {avisos[-1]}", file=sys.stderr)
    if precios is None:
        precios, dia_usado = prevision_benchmark(entrega)
        precios = validacion.preparar(entrega, precios)
        oficial = benchmark.dia_referencia(entrega)
        print(f"Benchmark con referencia {dia_usado}")
        if dia_usado != oficial:
            avisos.append(f"Benchmark con día de respaldo {dia_usado} (el oficial era {oficial})")
        es_benchmark = True
    else:
        es_benchmark = False

    # 2b. A 15 minutos: horaria + forma reciente dentro de cada hora (o benchmark cuartohorario)
    if minutos == 15:
        from prevision import cuartos
        try:
            q = cuartos.cuartos_recientes(entrega)
        except Exception as e:  # noqa: BLE001
            q = None
            avisos.append(f"Sin precios cuartohorarios recientes ({e}); se reparte la horaria sin forma")
        precios15 = None
        if es_benchmark and q is not None:
            try:
                precios15 = cuartos.benchmark_cuartos(entrega, q)
            except Exception as e:  # noqa: BLE001
                avisos.append(f"Benchmark cuartohorario no disponible ({e})")
        if precios15 is None:
            precios15 = cuartos.a_cuartos(precios, entrega, q)
        precios = validacion.preparar(entrega, precios15, minutos)
    print(precios.to_string())

    # 3. Enviar
    if args.dry_run:
        print("Dry run: no se envía nada.")
        return
    respuesta = cliente.enviar(args.modelo, entrega, precios, resolucion=args.resolucion)
    print(f"Enviado: id={respuesta['id']} versión={respuesta['version']} "
          f"estado={respuesta['status']} a las {respuesta['submitted_at']}")
    if cruda is not None and minutos == 60:
        from prevision import sesgo
        sesgo.guardar_en_registro(cruda)
        print(f"Registro actualizado: {sesgo.REGISTRO.name}")
        try:  # archivo de los datos tal como estaban (no debe afectar al envío si falla)
            from prevision import archivo, modelo
            ruta = archivo.guardar(entrega, modelo.ULTIMOS_DATOS)
            print(f"Archivo de datos: {ruta.name} ({ruta.stat().st_size // 1024} KB)")
        except Exception as e:  # noqa: BLE001
            print(f"AVISO: no se pudo archivar los datos ({e})", file=sys.stderr)
    if avisos:
        raise SystemExit("Enviado, pero con respaldo:\n" + "\n".join(avisos))


if __name__ == "__main__":
    main()
