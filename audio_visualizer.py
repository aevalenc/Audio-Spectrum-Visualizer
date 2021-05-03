# Copyright (C) 2021  Alejandro Valencia
# 
# audio_visualizer is free software; you can redistribute it and/or 
# modify it under the terms of the GNU General Public License as
# published by the Free Software Foundation; either version 3 of the
# License, or (at your option) any later version.
# 
# audio_visualizer is distributed in the hope that it will be fun and
# enjoyable. See the GNU General Public License for more details.
# 
# See <http://www.gnu.org/licenses/> for a copy of the GNU General Public
# License 
# 
# Author of current file: Alejandro Valencia
# Last Update: May 3, 2021

# /**********************************************************************
# *   Modules                                                           *
# **********************************************************************/

import sys
import numpy as np
import pyaudio
import pyqtgraph as pg
import struct
from pyqtgraph.Qt import QtGui, QtCore
from scipy.fftpack import fft
from scipy.signal import butter, sosfilt


# /**********************************************************************
# * Class: audio_visualizer                                             *
# **********************************************************************/


class audio_visualizer(object):

    # [A]:Initialize class
    def __init__(self):
        self.traces = dict()


        ## [B]:Setup pyqtgraph window
        pg.setConfigOptions(antialias=True)
        self.app = QtGui.QApplication(sys.argv)
        self.win = pg.plot(title='Audio Visualizer')
        self.win.resize(1000,400)
        self.win.setTitle("See the Beats")
	

        ## [C]:Parameters
        self.CHUNK    = 1024 * 2              # Samples per frame
        self.FORMAT   = pyaudio.paInt16       # Audio format
        self.CHANNELS = 1                     # Single channel for microphone
        self.RATE     = 44100                 # Sample rate [samples/s]
        self.NYQ      = self.RATE / 2         # Nyquist frequency
        self.N        = int(self.CHUNK/2 - 1) # Number of samples
        self.LFREQ    = 20                    # lower bound frequency
        self.UFREQ    = 8000                  # Upper bound frequency
        
        # Chosen one-third octave bands
        self.CBANDS   = [25, 31.5, 40, 50, 63, 80, 100, 125,
                         160, 200, 250, 315, 400, 500, 630, 800, 1000, 1250,
                         1600, 2000, 2500, 3150, 4000, 5000, 6300, 8000,
                         10000]
        self.CN       = len(self.CBANDS)

        # RGB Color map
        self.colors   = np.linspace(0, 255, 16) # Color range
        self.color    = (0, 255, 255)           # Initial color cyan


        ## [D]:Create pyaudio instance
        self.p = pyaudio.PyAudio()


        ## [E]:Stream Object
        self.stream = self.p.open(
            format=self.FORMAT,
            channels=self.CHANNELS,
            rate=self.RATE,
            input=True,
            output=True,
            frames_per_buffer=self.CHUNK
        )

        
        ## [F]:Create Frequency Bins
        self.f    = np.linspace(20, self.NYQ, self.N)
        self.bins = np.linspace(1, self.CN, self.CN)
        self.spectrum = pg.BarGraphItem(
            x=self.bins,
            height=np.random.rand(len(self.f)),
            width=0.2
        )
        self.win.addItem(self.spectrum)

    #end __init__



    ## [G]:Make sure audio_visualizer is ready to go
    def start(self):
        if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
            QtGui.QApplication.instance().exec_()
        # end if
    # end start



    ## [H]:Set BarGraphItem with current data
    def set_plotdata(self, name, data_y):
        if name in self.traces:
            if np.max(data_y) >= 0.4:
                self.color = (np.random.choice(self.colors), np.random.choice(self.colors), np.random.choice(self.colors))

            self.spectrum.setOpts(height=data_y, brush=self.color)
        else:
            if name == 'spectrum':
                self.traces[name] = self.spectrum.drawPicture()
            # end if

        # end if

    # end set_plotdata



    ## [I]:Filter Data
    def band_pass_filter(self, signal_data):
        order = 3
        sos   = butter(order, [self.LFREQ, self.UFREQ], btype='band', output='sos', fs=self.RATE)
        filtered_data = sosfilt(sos, signal_data)

        return filtered_data
    # end band_pass_filter



    # [J]:RMS Function
    def rms(self, signal_data):

        # Grab length of signal data
        N = len(signal_data)

        # Compute the sum of squares
        sum = 0
        for i in range(0, N):
            sum += signal_data[i] ** 2
        #end

        rms_data = np.sqrt(sum / N)

        return rms_data
    #end rms


    ## [K]:Organize Frequenices to One-third Octave Bands
    def octave_band(self, signal_data):

        rms_data = np.zeros(self.CN)
        for i in range(0, self.CN):
            lband = self.CBANDS[i] / np.power(2, 1/6)
            uband = self.CBANDS[i] * np.power(2, 1/6)

            indxlb, = np.where(self.f <= lband)
            indxlb  = indxlb[-1]
            indxub, = np.where(self.f >= uband)
            indxub  = indxub[0]

            band_data   = signal_data[indxlb:indxub+1]
            rms_data[i] = self.rms(band_data)

        #end i

        return rms_data
    #end octave_band



    # [L]:Update Graph
    def update(self):
        # Binary data
        data = self.stream.read(self.CHUNK)

        # Convert data to integers, make np array, then offset by 127
        # (255 / 2 Remember bias 0)
        data_int = struct.unpack(str(2 * self.CHUNK) + 'B', data)

        # Perform the Fast Fourier Transform and Obtain Spectrum
        filtered_data = self.band_pass_filter(data_int)
        sp_data = fft(filtered_data)  
        sp_data = np.abs(sp_data[1:int(self.CHUNK / 2)]) * 2 / (128 * self.CHUNK)
        sp_data = np.array(sp_data)
        sp_data = self.octave_band(sp_data)
        self.set_plotdata(name='spectrum', data_y=sp_data)
    #end update
        


    # []:Animation
    def animation(self):
        timer = QtCore.QTimer()
        timer.timeout.connect(self.update)
        timer.start(30)
        self.start()
    #end animation




# /**********************************************************************
# * Main Program                                                        *
# **********************************************************************/

if __name__ == '__main__':
    p = audio_visualizer()
    p.animation()
