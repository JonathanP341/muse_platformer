from pythonosc.dispatcher import Dispatcher
from pythonosc.osc_server import BlockingOSCUDPServer

from scipy.signal import welch
from scipy.integrate import simpson

import numpy as np
import heartpy as hp

import multiprocessing
import threading

class EEGReceiver:
    def __init__(self, ip="127.0.0.1", port=5000):
        #Set up values
        self.ip = ip
        self.port = port
        self.bands = {"delta": [0.5, 4], "theta": [4, 8], "alpha": [8, 12], "beta": [12, 30], "gamma": [30, 50]}
        self.PPG_SAMPLE_RATE = 64
        self.PPG_WINDOW_SIZE = 8 * self.PPG_SAMPLE_RATE
        self.EEG_SAMPLE_RATE = 256
        self.EEG_WINDOW_SIZE = 3 * self.EEG_SAMPLE_RATE

        #Synchronous values
        self.TP9Buffer = [] #LT
        self.AF7Buffer = [] #LF
        self.AF8Buffer = [] #RF
        self.TP10Buffer = [] #RT
        self.AUXBuffer = [] #X
        self.ppg_buffer = []

        self.latest_bandpower = []

        #Baseline values
        #WIP - Not sure the exact metrics to look at
        #Will include HRV and some bandpower thing for sure
        self.baseline_metrics = None

        # Set up for the dispatcher 
        self.dispatcher = Dispatcher()
        self.dispatcher.map("/muse/eeg", self.on_eeg)
        self.dispatcher.map("/muse/ppg", self.on_ppg)

        self.server = BlockingOSCUDPServer((ip, port), self.dispatcher) #Might need to switch this to AsyncIOOSCUDPServer

    def start(self):
        """Starting the OSC Server on a background thread"""
        thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        thread.start()
    
    def on_eeg(self, address, *values):
        """
        Processing the EEG data by putting it in the respective buffers

        address -- The address the values are being sent to 
        values -- Contains an array with 5 values [TP9, AF7, AF8, TP10, AUX]
        """
        self.TP9Buffer.append(values[0])
        self.AF7Buffer.append(values[1])
        self.AF8Buffer.append(values[2])
        self.TP10Buffer.append(values[3])
        self.AUXBuffer.append(values[4])

        if len(self.TP9Buffer) > self.EEG_WINDOW_SIZE:
            del self.TP9Buffer[:self.EEG_SAMPLE_RATE]
            del self.TP10Buffer[:self.EEG_SAMPLE_RATE]
            del self.AF7Buffer[:self.EEG_SAMPLE_RATE]
            del self.AF8Buffer[:self.EEG_SAMPLE_RATE]
            del self.AUXBuffer[:self.EEG_SAMPLE_RATE]
    
    def on_ppg(self, address, *values):
        self.ppg_buffer.append(values[0]) #Only storing PPG1

        if len(self.ppg_buffer) > self.PPG_WINDOW_SIZE:
            del self.ppg_buffer[:self.PPG_SAMPLE_RATE]

    def process_signal(self, buffer, win_sec):
        """
        Processing the signals to find the bandpower of each region averaged out

        This means we will find delta, theta, alpha, and beta(no gamma) for the channels 
        of my choice. For this I will just do AF7 and AF8 because they are the most 
        reliable on a muse headset.
        """  
        values = [0, 0, 0, 0, 0]
        if len(buffer) == 512:
            values[0] = round(self.bandpower(buffer, 256, self.bands['delta'], 2), 2)
            values[1] = round(self.bandpower(buffer, 256, self.bands['theta'], 2), 2)
            values[2] = round(self.bandpower(buffer, 256, self.bands['alpha'], 2), 2)
            values[3] = round(self.bandpower(buffer, 256, self.bands['beta'], 2), 2)
            values[4] = round(self.bandpower(buffer, 256, [0, 0], 2, total=True))
            del buffer[:44]
        return values

    def bandpower(self, data, sf, band, window_sec=None, relative=False, total=False):
        """Compute the average power of the signal x in a specific frequency band
        
        data -- The data we are getting the information from for the bandpower
        sf  -- The amount of samples per second
        band -- The ranges we want to analyze
        window_sec -- Getting the range of the window we want to analyze
        relative -- Set to true to get the perecentage of the total bandpower
        total -- Set to true to get the total bandpower itself  
        """
        low, high = band
        if window_sec is not None:
            nperseg = window_sec * sf
        else:
            nperseg = (2 / low) * sf

        #Compute Welch
        freqs, psd = welch(data, sf, nperseg=nperseg)

        # Frequency resolution
        freq_res = freqs[1] - freqs[0]

        if total: #Returning the total value, no index band
            return simpson(psd, dx=freq_res)
        
        #Find closest indecies of band in freq vector
        idx_band = np.logical_and(freqs >= low, freqs <= high)

        #Integral approx of area
        bp = simpson(psd[idx_band], dx=freq_res)

        if relative:
            bp /= simpson(psd, dx=freq_res)
        return bp

    def compute_hrv(self):
        """Getting the HRV from the PPG1 data"""
        if len(self.ppg_buffer) < 7 * 64:
            return None #Not enough data to get heart rate

        data = self.ppg_buffer[-self.PPG_SAMPLE_RATE*7:]

        try:
            wd, m = hp.process(data, sample_rate=self.PPG_SAMPLE_RATE)
            return m
        except hp.exceptions.HeartPyError:
            return None #Bad segment




#Potential methods I could introduce later
def get_epoch(window_sec):
    """This method would be if I change how the buffers are stored to get the last little bit of data"""

def preprocess_buffers():
    """This method would be to filter and flag the buffers"""

def find_baseline():
    """This would be to find the users baseline so we can compare"""

def get_tilt_score():
    """Using the baseline, we would get the tilt score with this method"""