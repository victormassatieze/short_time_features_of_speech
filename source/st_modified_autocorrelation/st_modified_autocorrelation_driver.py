import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy import signal

class ShortTimeModifiedAutocorrelationDriver ():
    def __init__ (self, signal_, window1_, window2_):
        self.signal = signal_
        self.window1 = window1_
        self.window2 = window2_
        self.full_correlation = self.short_time_modified_autocorrelation ()

    def short_time_modified_autocorrelation (self):
        window_range1 = len(self.window1)
        window_range2 = len(self.window2)
        min_window_range = min (window_range1, window_range2)
        max_window_range = max (window_range1, window_range2)
        self._pad_small_window ()
        slided_signal = sliding_window_view (self.signal, (max_window_range,))
        self.windowed_signal1 = slided_signal[::(min_window_range // 4), :] * self.window1
        self.windowed_signal2 = slided_signal[::(min_window_range // 4), :] * self.window2

        for i in range (min (self.windowed_signal1.shape[0], self.windowed_signal2.shape[0])):
            if i == 0:
                result = np.correlate (self.windowed_signal1[i, :], self.windowed_signal2[i, :], mode='full')
            else:
                result = np.vstack ((result, np.correlate (self.windowed_signal1[i, :], self.windowed_signal2[i, :], mode='full')))
            print (f"Processing frame {i+1} of {min (self.windowed_signal1.shape[0], self.windowed_signal2.shape[0])}", end='\r')

        return result[:, result.shape[1] // 2 : ]
    
    def _pad_small_window(self):
        length_diff = len (self.window1) - len (self.window2)

        if (length_diff == 0): 
            return
        
        elif (length_diff > 0):
            if (length_diff % 2 == 0):
                self.window2 = np.pad (self.window2, (length_diff // 2, length_diff // 2), 'constant')
            else:
                self.window2 = np.pad (self.window2, (length_diff // 2, length_diff // 2 + 1), 'constant')

        elif (length_diff < 0):
            if (length_diff % 2 == 0):
                self.window1 = np.pad (self.window1, (-length_diff // 2, -length_diff // 2), 'constant')
            else:
                self.window1 = np.pad (self.window1, (-length_diff // 2, -length_diff // 2 + 1), 'constant')