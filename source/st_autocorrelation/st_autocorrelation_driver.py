import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy import signal

class ShortTimeAutocorrelationDriver ():
    def __init__ (self, signal_, window_):
        self.signal = signal_
        self.window = window_
        self.full_correlation = self.short_time_autocorrelation ()

    def short_time_autocorrelation (self):
        window_range = len(self.window)
        slided_signal = sliding_window_view (self.signal, (window_range,))
        self.windowed_signal = slided_signal[::(window_range // 4), :] * self.window

        for i in range (self.windowed_signal.shape[0]):
            if i == 0:
                result = np.correlate (self.windowed_signal[i, :], self.windowed_signal[i, :], mode='full')
            else:
                result = np.vstack ((result, np.correlate (self.windowed_signal[i, :], self.windowed_signal[i, :], mode='full')))
            print (f"Processing frame {i+1} of {self.windowed_signal.shape[0]}", end='\r')

        return result[:, result.shape[1] // 2 : ]