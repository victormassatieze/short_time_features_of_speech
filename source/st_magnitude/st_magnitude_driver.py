import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

class ShortTimeMagnitudeDriver ():
    def __init__ (self, signal_, window_):
        self.signal = signal_
        self.window = window_

    def short_time_magnitude (self):
        """
        Calculates Short-Time Magnitude M[l] of signal x[n] with window w[n] according to:
        M[l] = sum_{m=-infty}^{infty} |x[m]w[l-m]|
        """
        window_range = len(self.window)
        slided_signal = sliding_window_view (self.signal, (window_range,))
        windowed_signal = slided_signal * self.window
        result = np.sum (np.abs(windowed_signal), axis=1)
        return result