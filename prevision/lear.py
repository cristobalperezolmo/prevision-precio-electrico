"""
LEAR (Lasso Estimated AutoRegressive), el modelo lineal de referencia en previsión de precios eléctricos
(Lago et al., 2021, epftoolbox). Un modelo Lasso por hora del día, con:
  - precios de las 24 horas de D, D-1, D-2 y D-6 (E-1, E-2, E-3, E-7)
  - previsiones exógenas horarias para E y para D (viento, sol, temperatura...)
  - variables diarias (gas, nuclear, hidráulica...) y día de la semana
Precios y variables se normalizan con mediana/MAD y se les aplica asinh (estabiliza los picos).
El Lasso (con alpha elegido por AIC) selecciona automáticamente las variables útiles.
"""

import warnings

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LassoCV, LassoLarsIC
from sklearn.model_selection import TimeSeriesSplit

from prevision.calendario import MADRID
from prevision.variables import tabla_diaria

EXOGENAS_HORARIAS = ["mm_viento", "mm_viento3", "mm_sol", "mm_temp", "mm_nubes"]
DIARIAS = ["ttf", "ttf_cambio", "festivo", "r_nuclear", "r_hidraulica", "r_ciclo", "r_demanda",
           "r_eolica", "r_solar", "r_hidraulica_media7"]


def matriz_diaria(X: pd.DataFrame, precios: pd.Series, extra_diarias: dict | None = None):
    """Devuelve (F, Y): una fila por día de entrega. F = variables, Y = 24 precios del día."""
    P = tabla_diaria(precios)
    P = P.reindex(columns=range(24)).interpolate(axis=1, limit_direction="both")  # 02:00 de marzo
    dias = sorted(X["entrega"].unique())
    idx = pd.Index(dias, name="entrega")

    bloques = []
    for k in (1, 2, 3, 7):
        lag = P.reindex([d - pd.Timedelta(days=k) for d in pd.to_datetime(dias)])
        lag.index = idx
        lag.columns = [f"p{k}_h{h}" for h in range(24)]
        bloques.append(lag)

    for col in EXOGENAS_HORARIAS:
        for sufijo, nombre in (("", "E"), ("_D", "D")):
            c = col + sufijo
            if c in X:
                t = X.pivot_table(index="entrega", columns="hora", values=c, aggfunc="mean")
                t = t.reindex(index=dias, columns=range(24)).interpolate(axis=1, limit_direction="both")
                t.index = idx
                t.columns = [f"{col}_{nombre}_h{h}" for h in range(24)]
                bloques.append(t)

    diarias = X.groupby("entrega")[[c for c in DIARIAS if c in X]].first().reindex(dias)
    diarias.index = idx
    bloques.append(diarias)
    for nombre, serie in (extra_diarias or {}).items():
        bloques.append(pd.DataFrame({nombre: serie.reindex(dias).values}, index=idx))
    semana = pd.get_dummies(pd.to_datetime(pd.Series(dias)).dt.weekday, prefix="dow").astype(float)
    semana.index = idx
    bloques.append(semana)

    F = pd.concat(bloques, axis=1)
    Y = X.pivot_table(index="entrega", columns="hora", values="objetivo", aggfunc="mean")
    Y = Y.reindex(index=dias, columns=range(24)).interpolate(axis=1, limit_direction="both")
    Y.index = idx
    return F, Y


class Escalador:
    """Mediana/MAD + asinh (el 'invariant scaler' de epftoolbox)."""

    def ajustar(self, A: np.ndarray):
        self.med = np.nanmedian(A, axis=0)
        mad = np.nanmedian(np.abs(A - self.med), axis=0) / 0.6745
        self.mad = np.where(mad > 1e-9, mad, 1.0)
        return self

    def transformar(self, A):
        return np.arcsinh((A - self.med) / self.mad)

    def invertir(self, Z):
        return np.sinh(Z) * self.mad + self.med


class LEAR:
    """24 modelos Lasso (uno por hora) con sus escaladores. Se puede guardar con joblib."""

    def fit(self, F_train: pd.DataFrame, Y_train: pd.DataFrame):
        ok = F_train.notna().all(axis=1) & Y_train.notna().all(axis=1)
        F_train, Y_train = F_train[ok], Y_train[ok]
        columnas = [c for c in F_train.columns if F_train[c].std() > 0]
        self.dummies = [c for c in columnas if c.startswith("dow") or c == "festivo"]
        self.continuas = [c for c in columnas if c not in self.dummies]
        self.ex = Escalador().ajustar(F_train[self.continuas].values)
        self.ey = Escalador().ajustar(Y_train.values)
        A = self._preparar(F_train)
        Z = self.ey.transformar(Y_train.values)
        self.modelos = []
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            for h in range(24):
                if len(A) > 1.5 * A.shape[1]:
                    # Como epftoolbox: alpha por criterio de información (rápido)
                    m = LassoLarsIC(criterion="aic", max_iter=2000).fit(A, Z[:, h])
                else:
                    # Pocas filas para tantas variables: alpha por validación cruzada temporal
                    m = LassoCV(alphas=40, cv=TimeSeriesSplit(4), max_iter=5000, n_jobs=4).fit(A, Z[:, h])
                self.modelos.append(m)
        return self

    def _preparar(self, F):
        A = self.ex.transformar(F.reindex(columns=self.continuas)
                                .fillna(pd.Series(self.ex.med, index=self.continuas)).values)
        return np.hstack([A, F.reindex(columns=self.dummies).fillna(0).values])

    def predict(self, F: pd.DataFrame) -> np.ndarray:
        """Devuelve (n_dias, 24) precios previstos."""
        A = self._preparar(F)
        return self.ey.invertir(np.column_stack([m.predict(A) for m in self.modelos]))


def ajustar_y_predecir(F_train, Y_train, F_test) -> np.ndarray:
    """Entrena 24 Lasso (uno por hora) y predice las filas de F_test. Devuelve (n_test, 24)."""
    return LEAR().fit(F_train, Y_train).predict(F_test)


def walk_forward_lear(F: pd.DataFrame, Y: pd.DataFrame, desde_test, ventana_dias: int | None,
                      cada_dias: int = 7) -> pd.DataFrame:
    """Recalibra cada `cada_dias` días con los últimos `ventana_dias` días (None = todo)."""
    dias_test = [d for d in F.index if d >= desde_test]
    preds = []
    for i in range(0, len(dias_test), cada_dias):
        bloque = dias_test[i:i + cada_dias]
        inicio = bloque[0]
        train = F.index[(F.index < inicio)]
        if ventana_dias:
            train = train[train >= inicio - pd.Timedelta(days=ventana_dias)]
        p = ajustar_y_predecir(F.loc[train], Y.loc[train], F.loc[bloque])
        preds.append(pd.DataFrame(p, index=bloque, columns=range(24)))
    return pd.concat(preds)


def a_horario(pred_diaria: pd.DataFrame, X: pd.DataFrame) -> pd.Series:
    """Pasa la predicción [día x hora local] a la tabla horaria de X (índice UTC)."""
    claves = list(zip(X["entrega"], X["hora"]))
    valores = [pred_diaria.at[d, h] if d in pred_diaria.index else np.nan for d, h in claves]
    return pd.Series(valores, index=X.index)
