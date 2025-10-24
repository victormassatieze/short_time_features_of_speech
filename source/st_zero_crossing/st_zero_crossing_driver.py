import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

class ShortTimeZeroCrossingDriver ():
    def __init__ (self, signal_, window_):
        self.signal = signal_
        self.window = window_

    def short_time_zero_crossing (self):
        """
        Calculates Short-Time Zero Crossing Z[l] of signal x[n] with window w[n] according to:
        M[l] = frac{1}{2 L_{eff}} sum_{m=-infty}^{infty} |sgn(x[m]) - sgn(x[m-1])| w[l-m]
        """
        window_range = len(self.window)
        slided_signal = sliding_window_view (self.signal, (window_range,))

        zeroed_signal = np.zeros_like (slided_signal)
        zeroed_signal [np.where (np.diff (np.sign (slided_signal)))] = 1

        effective_window_length = np.sum (self.window)

        result = np.matmul(zeroed_signal, np.transpose(self.window))/2/effective_window_length

        return result
