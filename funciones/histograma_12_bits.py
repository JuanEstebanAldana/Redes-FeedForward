"""Histograma de ventanas de 12 bits con desplazamiento de 4 bits.

Cada fragmento produce 4096 características; el target queda fuera.
Se recorre cada byte del bit más significativo al menos significativo.
No se agregan ceros para completar ventanas al final.

Uso con un DataFrame cuya última columna es el target (antes de escalar):
    X = dataset.iloc[:, :-1].to_numpy()
    y = dataset.iloc[:, -1].to_numpy()
    X_hist = np.stack([
        histograma_12_bits(fila, normalizar=True) for fila in X
    ])
    # X_hist tiene forma (cantidad_de_fragmentos, 4096).
    # y conserva exactamente las etiquetas originales.
"""

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


def histograma_12_bits(fragmento, normalizar=False):
    """Recibe bytes o un vector de valores enteros entre 0 y 255.

    Devuelve 4096 conteos, o frecuencias relativas si normalizar=True.
    Un fragmento de menos de 2 bytes produce un vector de ceros.
    """
    if isinstance(fragmento, (bytes, bytearray, memoryview)):
        datos = np.frombuffer(fragmento, dtype=np.uint8)
    else:
        datos = np.asarray(fragmento)

    if datos.ndim != 1:
        raise ValueError("Se espera un solo fragmento: un vector de bytes.")
    if datos.dtype.kind not in "iuf":
        raise ValueError("Los bytes deben ser números enteros entre 0 y 255.")
    if not np.all(np.isfinite(datos)) or np.any((datos < 0) | (datos > 255)):
        raise ValueError("Los bytes deben estar entre 0 y 255, sin NaN ni infinitos.")
    if np.any(datos != np.floor(datos)):
        raise ValueError("Usá los bytes originales, antes de normalizar o escalar.")

    # Cada byte se expande a 8 bits, empezando por el más significativo.
    bits = np.unpackbits(datos.astype(np.uint8), bitorder="big")
    histograma = np.zeros(16384, dtype=np.int64)

    if bits.size >= 14:
        # Inicios en los bits 0, 4, 8, 12, ...; solo ventanas completas.
        ventanas = sliding_window_view(bits, 14)[::2]

        # Los 12 bits representan un índice entre 0 y 4095.
        pesos = 2 ** np.arange(13, -1, -1, dtype=np.uint16)
        valores = ventanas @ pesos
        histograma = np.bincount(valores, minlength=16384)

    if normalizar:
        histograma = histograma.astype(np.float32)
        total = histograma.sum()
        if total > 0:
            histograma /= total

    return histograma


if __name__ == "__main__":
    ejemplo = bytes.fromhex("AB CD EF")
    histograma = histograma_12_bits(ejemplo)
    print("Ventanas:", [format(i, "03X") for i in np.flatnonzero(histograma)])
    print("Cantidad de ventanas:", histograma.sum())
    print("Cantidad de características:", histograma.shape[0])

