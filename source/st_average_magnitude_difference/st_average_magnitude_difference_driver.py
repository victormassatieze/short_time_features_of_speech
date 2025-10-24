import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy import signal

class ShortTimeAverageMagnitudeDifferenceDriver ():
    def __init__ (self, signal_, window1_, window2_):
        self.signal = signal_
        self.window1 = window1_
        self.window2 = window2_
        self.full_magnitude_difference = self.short_time_average_magnitude_difference ()

    def short_time_average_magnitude_difference (self):
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
                result = self.average_magnitude_difference (self.windowed_signal1[i, :], self.windowed_signal2[i, :])
            else:
                result = np.vstack ((result, self.average_magnitude_difference (self.windowed_signal1[i, :], self.windowed_signal2[i, :])))
            print (f"Processing frame {i+1} of {min (self.windowed_signal1.shape[0], self.windowed_signal2.shape[0])}", end='\r')

        return result
    
    def average_magnitude_difference (self, frame1, frame2):
        if (len (frame1) != len (frame2)):
            raise 1

        for lag in range (len (frame1)):
            if lag == 0:
                result = np.sum (np.abs (frame1 - frame2))
            else:
                result = np.append (result, np.sum (np.abs (frame1 - np.roll (frame2, lag))))

        return result
    
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