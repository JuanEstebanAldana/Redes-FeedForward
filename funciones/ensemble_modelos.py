"""Ensemble de redes Keras guardadas: promedio de probabilidades.

No modifica los pesos ni vuelve a entrenar las redes.
Se presupone que todos los modelos predicen las MISMAS clases en el MISMO orden.
Cada entrada debe haber sido procesada igual que durante el entrenamiento
correspondiente (ventana, salto y normalizacion del histograma).
"""

from pathlib import Path
from typing import Optional, Sequence, Union

import numpy as np
import pandas as pd


Array = np.ndarray


def promediar_modelos(
    rutas_modelos: Sequence[Union[str, Path]],
    entradas: Union[Array, Sequence[Array]],
    pesos: Optional[Sequence[float]] = None,
    batch_size: int = 32,
    verbose: int = 0,
):
    """Carga modelos .keras/.h5 y promedia sus probabilidades softmax.

    Parametros
    ----------
    rutas_modelos : lista de rutas a archivos de modelos guardados.
    entradas :
        - Un array (N, D): cuando TODOS los modelos usan las mismas entradas.
        - Una lista de arrays [X1, X2, ...]: uno por modelo, si cada uno
          usa histogramas o preprocesamientos diferentes.
        Todos los arrays deben tener la misma cantidad N de muestras y el
        mismo orden de filas.
    pesos : pesos del promedio; None = todos iguales. No necesitan sumar 1.

    Devuelve
    --------
    probabilidades : (N, C) con el promedio ponderado de cada clase.
    predicciones : (N,) con los indices de la clase mas probable (0, ..., C-1).
    """
    # Se importa aqui para que el modulo pueda inspeccionarse sin cargar Keras.
    try:
        from keras.models import load_model
    except ImportError:
        from tensorflow.keras.models import load_model

    rutas = [Path(ruta) for ruta in rutas_modelos]
    if not rutas:
        raise ValueError("Debes indicar al menos un archivo de modelo.")

    for ruta in rutas:
        if not ruta.is_file():
            raise FileNotFoundError(f"No existe el modelo: {ruta}")

    # Si los modelos tienen diferentes entradas, pasarlas en una lista.
    if isinstance(entradas, np.ndarray):
        entradas_por_modelo = [entradas] * len(rutas)
    else:
        entradas_por_modelo = list(entradas)
        if len(entradas_por_modelo) != len(rutas):
            raise ValueError(
                "Si proporcionas una lista de entradas, debe contener exactamente "
                "un array por modelo, en el mismo orden."
            )

    entradas_por_modelo = [np.asarray(x) for x in entradas_por_modelo]
    n_muestras = len(entradas_por_modelo[0])
    if n_muestras == 0:
        raise ValueError("No hay muestras que predecir.")
    for i, x in enumerate(entradas_por_modelo):
        if x.ndim < 2:
            raise ValueError(f"La entrada {i + 1} debe incluir muestras y caracteristicas.")
        if len(x) != n_muestras:
            raise ValueError("Todas las entradas deben tener igual numero de filas.")

    if pesos is None:
        ponderaciones = np.ones(len(rutas), dtype=np.float64)
    else:
        ponderaciones = np.asarray(pesos, dtype=np.float64)
        if ponderaciones.shape != (len(rutas),):
            raise ValueError("Debe existir un peso por cada modelo.")
        if not np.all(np.isfinite(ponderaciones)) or np.any(ponderaciones < 0):
            raise ValueError("Los pesos deben ser numeros finitos no negativos.")
    if ponderaciones.sum() <= 0:
        raise ValueError("Al menos un peso debe ser mayor que cero.")
    ponderaciones /= ponderaciones.sum()

    probabilidades_media = None
    for i, (ruta, x, peso) in enumerate(zip(rutas, entradas_por_modelo, ponderaciones)):
        modelo = load_model(str(ruta), compile=False)
        forma = modelo.input_shape
        if isinstance(forma, list):
            raise ValueError(f"El modelo {ruta} tiene multiples entradas: no esta soportado.")
        if len(forma) != x.ndim or any(
            esperado is not None and esperado != real
            for esperado, real in zip(forma[1:], x.shape[1:])
        ):
            raise ValueError(
                f"Entrada incompatible para {ruta.name}: "
                f"el modelo espera {forma}, pero recibio {x.shape}. "
                "Regenera el histograma con la ventana, salto y normalizacion correctos."
            )

        p = np.asarray(modelo.predict(x, batch_size=batch_size, verbose=verbose))
        if p.ndim != 2 or p.shape[0] != n_muestras:
            raise ValueError(f"Salida no compatible con clasificacion multiclase en {ruta}.")
        if (not np.all(np.isfinite(p)) or np.any(p < -1e-6)
                or not np.allclose(p.sum(axis=1), 1.0, atol=1e-3)):
            raise ValueError(
                f"El modelo {ruta.name} no devuelve probabilidades softmax. "
                "Para promediarlas directamente, cada fila debe sumar 1."
            )

        if probabilidades_media is None:
            probabilidades_media = np.zeros_like(p, dtype=np.float64)
        if p.shape[1] != probabilidades_media.shape[1]:
            raise ValueError("Los modelos no tienen la misma cantidad de clases.")
        probabilidades_media += peso * p
        print(f"Modelo {i + 1}/{len(rutas)} procesado: {ruta.name}")

    # Se devuelve float32, suficiente para probabilidades de una red neuronal.
    probabilidades_media = probabilidades_media.astype(np.float32)
    predicciones = np.argmax(probabilidades_media, axis=1)
    return probabilidades_media, predicciones


def guardar_predicciones(predicciones: Array, archivo: Union[str, Path] = "entrega_promedio.csv"):
    """Exporta una columna y_pred, con encabezado y sin columna de indice."""
    pd.DataFrame({"y_pred": np.asarray(predicciones).reshape(-1)}).to_csv(
        archivo, index=False
    )
    print(f"Predicciones guardadas en: {archivo}")


# Ejemplo de uso desde Jupyter:
# from ensemble_modelos import promediar_modelos, guardar_predicciones
# rutas = ["modelo_A.keras", "modelo_B.keras", "modelo_C.keras"]
# probabilidades, y_pred = promediar_modelos(rutas, X_hist_test)
# guardar_predicciones(y_pred, "entrega_promedio.csv")
#
# Si algun modelo fue entrenado con un histograma distinto, proporcionar
# un array diferente POR MODELO:
# probabilidades, y_pred = promediar_modelos(
#     ["modelo_8bits.keras", "modelo_12bits.keras"],
#     [X_hist_test_8bits, X_hist_test_12bits],
# )
