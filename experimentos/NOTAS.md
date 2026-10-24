# Cuaderno de experimentos

Protocolo: walk-forward, recalibración mensual (salvo que se diga otra cosa), test desde ene-2025.
Se elige con **2025 (ajuste)** y se confirma con **2026 (confirmación, ene-oct)**. rMAE = MAE / MAE del benchmark.

## Referencia
| Modelo | rMAE 2025 | rMAE 2026 |
|---|---|---|
| v1 inicial (1 LightGBM) | 0,598 | 0,639 |
| v1 en producción (media de 4 LightGBM, ext) | 0,563 | 0,600 |

## Ronda 1 (20-oct): fuentes ext (GFS/ICON, REData) y variantes — ver `ronda` de mejoras_v1.py
- Fuentes nuevas: 0,598 → 0,572 / 0,639 → 0,612.
- Pesos de la métrica, vida media 180, árboles de 15 hojas: no aportan.
- Un modelo por estación: mucho peor (0,64 / 0,73). Por tramos del día: igual que el modelo único.
- Media de 4 variantes: 0,563 / 0,600 → **en producción**.

## Ronda 2: fuentes ext2 (MIBGAS, CO2, intercambios y bombeo de REData, índices eólico/solar con 24 puntos)
- extras: 0,572 → 0,567 / 0,612 → 0,588
- reciente365: 0,575 → 0,567 / 0,607 → 0,583
- hojas63: 0,577 → 0,565 / 0,609 → 0,587 · tramos: 0,573 → 0,566 / 0,615 → 0,598
- **Media de las 4 (ext2): 0,555 / 0,578** (producción: 0,563 / 0,600)
- Media de 4 (ext2) + 15 % LEAR: **0,550 / 0,575**

## LEAR (Lasso por hora, epftoolbox)
- Todo el histórico, recalibración cada 28 días: 0,660 / 0,710 solo; **20 % LEAR + 80 % LightGBM: 0,557 / 0,592**.
- Recalibración semanal: todo 0,641 / 0,697 · ventana 364 d 0,675 / 0,733 · ventana 182 d 0,752 / 0,821.
- Media de ventanas (todo+364+182): 0,653 / 0,709 → no ayuda (ventanas cortas = pocas filas para ~350 variables).
- En la mezcla con LightGBM ext2 (15 %): semanal 0,548 / 0,573 vs cada 28 días 0,550 / 0,575 → diferencia mínima.

## 15 minutos (PT15M)
- Desde oct-2025 OMIE es cuartohorario. Pasar la previsión horaria a cuartos:
  plano 0,625 · forma de D 0,595 · **forma media de 7 días 0,588** · interpolación 0,594 (rMAE frente al benchmark a 15 min).
- Ranking PT15M de ES: un competidor activo (rogarui, 2 modelos, MAE 34 y 31 el 20-oct;
  nuestro backtest ese día: 22). Ranking PT60M: casi vacío.

## Corrección del sesgo reciente (sobre media ext2 + 15 % LEAR)
A las 10 UTC de D se conocen los precios reales de D, así que se conoce el error de nuestra previsión de D.
Corrección: restar k × (media exponencial de los errores pasados a esa misma hora).
- Sin corrección: 0,550 / 0,575
- Media móvil 14 días por hora, k=0,5: 0,541 / 0,557
- **Media exponencial, vida media 14 días, k=0,7: 0,539 / 0,555** (vida media 7–28 y k 0,5–1 dan 0,539–0,547 / 0,554–0,560: estable)
- Para producción: guardar cada día la previsión (sin corregir) para calcular los errores.

## Ronda 3 (tabla ext3 = ext2 + perfil 24 h de D), variante única "extras" (ext2: 0,567 / 0,588)
- Ventana 730 días: 0,568 / 0,581 (en 2025 casi igual porque apenas hay más de 2 años)
- Objetivo = diferencia con el día de referencia del benchmark: 0,597 / 0,630 → peor
- Perfil 24 h de D como variables: 0,562 / 0,620 → mejora 2025 pero empeora mucho 2026 (inestable): descartado

## Red neuronal (MLP 256-128, 24 salidas, entradas de LEAR, 3 semillas, recalibración mensual)
- Sola: 0,710 / 0,721. En la mezcla (+10 %): 0,548/0,573 → 0,547/0,570. Mejora mínima: no compensa la complejidad.
- Objetivo = precio directo: 0,578 / 0,617 → peor
- Más árboles con tasa menor (2000 × 0,015): 0,565 / 0,586 → casi igual
- **Reentrenamiento semanal: 0,559 / 0,568** (y el sesgo de 2026 pasa de -1,5 a -0,4) → mejora clara
- Semanal + corrección de sesgo: 0,556 / 0,566 (la corrección aporta menos: se solapan)
- Semanal + 15 % LEAR semanal: 0,549 / 0,561 → con corrección: **0,544 / 0,556** (un solo LightGBM)

## Ronda 4 (tabla ext2, variante "extras" = 0,567 / 0,588): qué fuente aporta (quitándola)
- Sin MIBGAS: 0,566 / 0,599 → MIBGAS aporta en 2026 (se queda)
- Sin CO2: 0,563 / 0,590 → neutro: **se quita** (una fuente menos que pueda fallar)
- Sin índices eólico/solar de 24 puntos: 0,583 / 0,608 → es lo que más aporta (se queda)
- Sin balance (intercambios/bombeo): 0,564 / 0,593 → neutro
- Ventana de 730 días en las otras variantes: hojas63 0,570/0,586 (vs 0,565/0,587), tramos 0,572/0,598 (vs 0,566/0,598),
  reciente365 0,570/0,583 (vs 0,567/0,583) → no aporta: se entrena con todo el histórico

## Ronda 5 (tabla ext4 = ext2 sin CO2 + factor de carga eólico y radiación de cada uno de los 24 puntos)
- extras 0,561/0,590 · reciente365 0,561/0,586 · hojas63 0,557/0,588 · tramos 0,560/0,593
- Media de 4: ext2 0,555/0,578 → **ext4 0,548/0,580** · + 15 % LEAR: 0,541/0,573 · + corrección de sesgo: **0,531 / 0,556**
- Se elige ext4 (mejor en 2025, igual en 2026)

## Ronda 6: reentrenamiento SEMANAL, tabla ext2 sin CO2
- extras 0,557/0,564 · reciente365 0,556/0,560 · hojas63 0,553/0,562 · tramos 0,555/0,568
- Media de 4: **0,544 / 0,553** (mensual: 0,555 / 0,578) · + 15 % LEAR semanal: 0,538 / 0,551
- + corrección de sesgo: **0,533 / 0,544** (producción actual: 0,563 / 0,600)

## Ronda 8: previsiones oficiales ENTSO-E de España vía Energy-Charts (gratis, sin token), tabla ext5, mensual
Disponibilidad: eólica/solar day-ahead para E se publican hacia las 18:00 de D → NO se usan (fuga).
La demanda prevista para E debe publicarse ≤ 2 h antes del cierre de la subasta (10:00 Bruselas) → sí, en principio.
- extras 0,538/0,568 · reciente365 0,535/0,556 · hojas63 0,532/0,561 · tramos 0,546/0,577
- Media de 4: **0,526 / 0,555** (ext4: 0,548 / 0,580) · + LEAR + corrección: **0,515 / 0,533** (mensual)
- Ronda 9 (sin la demanda prevista para E): 0,559/0,589 → TODA la mejora viene de la demanda prevista para E.
- ⚠️ PENDIENTE verificar la hora real de publicación y si el valor se revisa después
  (experimentos/vigilar_ec.py, snapshots en data/snapshots/). Hasta verificarlo, no pasar a producción.

## Ronda 7: SEMANAL, tabla ext4 (= juego "sin_ec" de producción)
- extras 0,549/0,567 · reciente365 0,548/0,558 · hojas63 0,547/0,565 · tramos 0,552/0,571
- Media de 4: 0,537 / 0,555 · + 15 % LEAR: 0,532 / 0,552 · + corrección: **0,527 / 0,545**

## LEAR ampliado (idx eólico/solar, MIBGAS, demanda ENTSO-E): 0,614 / 0,696 (LEAR: 0,641 / 0,697)
En la mezcla final no cambia nada (0,511/0,528 vs 0,513/0,528) → se mantiene el LEAR original (no depende de ENTSO-E).

## Ronda 10: SEMANAL, tabla ext5 (= juego "con_ec" de producción)
- extras 0,530/0,536 · reciente365 0,523/0,531 · hojas63 0,525/0,534 · tramos 0,535/0,545
- Media de 4: 0,518 / 0,527 · + 15 % LEAR: 0,513 / 0,528 · + corrección: **0,511 / 0,521**

# SISTEMA FINAL (experimentos/final.py)
Semanal · 4 LightGBM (ext5) + 15 % LEAR · corrección de sesgo (vida media 14 d, k = 0,7)
- **Total ene-2025 a oct-2026: MAE 11,38 €/MWh (benchmark 22,09) → rMAE 0,515**; todos los meses < 1 (máx. 0,75, may-2025)
- 2025: 0,511 · 2026: 0,521 (producción actual: 0,563 / 0,600; v1 inicial: 0,598 / 0,639)
- A 15 minutos (forma media de 7 días): rMAE 0,527 (MAE 12,0 frente a 22,8)
- Respaldo sin la demanda de ENTSO-E (juego sin_ec): total 0,535 (2025 0,527 / 2026 0,545)

## Ronda 14 (22-oct-2026): previsiones D+1 de Red Eléctrica (e·sios)
Indicadores 1777 (eólica), 1779 (FV) y 1775 (demanda) D+1, publicados hacia las 09:00 UTC de D
(comprobado: 08:53–08:59 UTC). Variables para E y D, cambio, medias diarias y demanda residual (ext8 = ext5 + e·sios).
Reentrenamiento mensual, media de las 4 variantes:

| | 2025 | 2026 |
|---|---|---|
| ext5 (v1.2) | 0,526 | 0,555 |
| ext8 (+ e·sios) | **0,461** | **0,497** |

P90 25,7/28,2 → 22,6/25,7. Adoptado en producción como v1.3 (juego `esios`, con el v1.2 de respaldo y como
sombra en registro/sombra.csv). Pendiente: confirmar que REE no revisa estas previsiones a posteriori
(foto del 22-oct 09:16 UTC en el scratchpad) y comparar v1.3 con la sombra con precios reales.

## Rondas 15-21 (24-oct-2026): futuros de OMIP y curvas de oferta recientes → v1.4
- r15 (ext9 = ext8 + futuros OMIP de la sesión de D-1 o anterior; la de D es el precio ya casado): 0,461/0,497 → 0,458/0,491.
- r16 (variantes nuevas en ext8): l1 ayuda algo en la media; lento, hojas127 y ventana de 730 días no.
- LEAR con exógenas de e·sios: 0,641/0,697 → 0,569/0,666 solo; en la mezcla, +0,003.
- r17 (ext10 = ext9 + curva de oferta reciente: mediana del precio en las 12 horas de los 14 días previos con
  residual REE más parecida): 0,439/0,464. La mayor mejora desde e·sios (el sábado 24 fallamos por anclarnos a D).
- r18 (objetivo = diferencia con la curva / mixta): solos parecidos; combinados con r17 mejoran 2026.
- r19 (ext11 = + curvas 7d/k6, 28d/k24, 14d/k4): 0,438/0,456. r20 (curvas por tramo del día): no aporta.
- r21 (ext11, objetivo diferencia con la curva) + r19 + LEAR e·sios 0,1: 0,435/0,455.
- Corrección de sesgo: K=0,7 y vida media 14 siguen siendo las mejores (rejilla K 0,3-1 × vida 3-28).
- Sistema completo con sesgo (ene-2025 a oct-2026): v1.3 rMAE 0,465 (MAE 10,27) → v1.4 0,439 (MAE 9,70);
  15 min 0,477 → 0,452. En producción desde el 24-oct-2026 (envío del 25).
