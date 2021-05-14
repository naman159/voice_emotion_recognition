# -*- coding: utf-8 -*-

import numpy as np
import math
from scipy.signal import lfilter
from audiolazy import lpc
from scipy.signal import find_peaks
from python_speech_features import mfcc

class FeatureExtractor():
    def __init__(self, debug=True):
        self.debug = debug

    def _compute_formants(self, audio_buffer):
        N = len(audio_buffer)
        Fs = 8000 # sampling frequency
        hamming_window = np.hamming(N)
        window = audio_buffer * hamming_window
    
        filtered_buffer = lfilter([1], [1., 0.63], window)
        
        ncoeff = 2 + Fs / 1000
        A = lpc(filtered_buffer, int(ncoeff))
        A = np.array([list(A)[0][i] for i in range(0,10)])

        roots = np.roots(A)
        roots = [r for r in roots if np.imag(r) >= 0]
    
        angz = np.arctan2(np.imag(roots), np.real(roots))

        unsorted_freqs = angz * (Fs / (2 * math.pi))
        
        freqs = sorted(unsorted_freqs)

        indices = np.argsort(unsorted_freqs)
        sorted_roots = np.asarray(roots)[indices]
        
        bandwidths = -1/2. * (Fs/(2*math.pi))*np.log(np.abs(sorted_roots))

        if self.debug:
            print("Identified {} formants.".format(len(freqs)))
    
        return freqs, bandwidths
        
    def _compute_formant_features(self, window):
        freq, bandw =  self._compute_formants(window)
        hist, bin_edges = np.histogram(freq, bins = 50)
        return hist
    
    def _compute_mfcc(self, window):
        mfccs = mfcc(window,8000,winstep=.0125)
        if self.debug:
            print("{} MFCCs were computed over {} frames.".format(mfccs.shape[1], mfccs.shape[0]))
        return mfccs
    
    def power(self, window):
        m = np.mean(window*window)
        return m
    
    def intensity(self, window):
        p_0 = 2*(10**-5)
        p = self.power(window)
        intensity = 10 * math.log(p/p_0, 10)
        return intensity 
    
    def peak(self, window):
        p,_ = find_peaks(window)
        return len(p)
    
    def _fft(self, window):
        return np.mean(np.fft.fft(window, axis = 0).astype(float)) 
    
    def maxVal(self, window):
        return np.amax(window);
    
    def minVal(self, window):
        return np.amin(window);
        
    def _compute_delta_coefficients(self, window, n=2):
        mfccs = self._compute_mfcc(window)
        d = []
        denom = 2*5;
        
        for t in range(n, 79-n):
            a = 1*(mfccs[t+1] - mfccs[t-1])
            b = 2*(mfccs[t+2] - mfccs[t-2])
            d.append(np.add(a, b)/denom)
        
        d_array = np.array(d);
        d_array = d_array.flatten();

        return d_array;
        
    def extract_features(self, window, debug=True):
        x = []
        x = np.append(x, self._compute_formant_features(window))
        x = np.append(x, self._compute_delta_coefficients(window))
        x = np.append(x, self.power(window))
        x = np.append(x, self.intensity(window))
        x = np.append(x, self._fft(window))
        x = np.append(x, self.peak(window))
        x = np.append(x, self.maxVal(window))
        x = np.append(x, self.minVal(window))
        return x    
