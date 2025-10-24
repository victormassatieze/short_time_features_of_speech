import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

class ShortTimeEnergyDriver ():
    def __init__ (self, signal_, window_):
        self.signal = signal_
        self.window = window_
    
    def short_time_energy (self):
        """
        Calculates Short-Time Energy E[l] of signal x[n] with window w[n] according to:
        E[l] = sum_{m=-infty}^{infty} (x[m]w[l-m])^2
        """
        window_range = len(self.window)
        slided_signal = sliding_window_view (self.signal, (window_range,))
        windowed_signal = slided_signal * self.window
        result = np.sum (windowed_signal ** 2, axis=1)
        return result