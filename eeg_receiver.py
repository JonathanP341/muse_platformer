import warnings
warnings.filterwarnings("ignore", category=UserWarning, module='pkg_resources')

from pythonosc.dispatcher import Dispatcher
from pythonosc.osc_server import BlockingOSCUDPServer

from scipy.signal import welch
from scipy.integrate import simpson
from collections import deque

import numpy as np
import heartpy as hp

import multiprocessing
import threading
import time

class EEGReceiver:
    def __init__(self, ip="0.0.0.0", port=5000):
        #Set up values
        self.ip = ip
        self.port = port
        self.bands = {"delta": [0.5, 4], "theta": [4, 8], "alpha": [8, 12], "beta": [12, 30], "gamma": [30, 50]}
        self.PPG_SAMPLE_RATE = 64
        self.PPG_WINDOW_SIZE = 30 * self.PPG_SAMPLE_RATE
        self.EEG_SAMPLE_RATE = 256
        self.EEG_WINDOW_SIZE = 3 * self.EEG_SAMPLE_RATE

        #Synchronous values
        self.TP9Buffer = deque(maxlen=self.EEG_WINDOW_SIZE) #LT
        self.AF7Buffer = deque(maxlen=self.EEG_WINDOW_SIZE) #LF
        self.AF8Buffer = deque(maxlen=self.EEG_WINDOW_SIZE) #RF
        self.TP10Buffer = deque(maxlen=self.EEG_WINDOW_SIZE) #RT
        self.ppg_buffer = deque(maxlen=self.PPG_WINDOW_SIZE) 

        self.latest_bandpower = []

        #Baseline values
        #WIP - Not sure the exact metrics to look at
        #Will include HRV and some bandpower thing for sure
        self.baseline_metrics = {}


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
        values -- Contains an array with 5 values [TP9, AF7, AF8, TP10, AUX], we ignore AUX
        """
        raw_values = np.array(values[:4]) #Ignoring AUX
        eeg_data = np.array(raw_values, dtype=float) 

        #Scrubbing out bad values
        if np.isnan(eeg_data).any() or (eeg_data == 0.0).any():
            return 
        
        self.TP9Buffer.append(eeg_data[0])
        self.AF7Buffer.append(eeg_data[1])
        self.AF8Buffer.append(eeg_data[2])
        self.TP10Buffer.append(eeg_data[3])
    
    def on_ppg(self, address, *values):
        """
        Processing the PPG data by appending it to its respective buffers, only considering
        PPG1 data as its the most reliable
        Args:
            address -- The address the values are being sent to 
            values -- Contains an array with 3 values [PPG1, PPG2, PPG3], we ignore PPG2/3
        """
        try:
            val = float(values[0])
            if not np.isnan(val) or val != 0.0:
                self.ppg_buffer.append(val) #Only storing PPG1
        except (ValueError, TypeError):
            pass #Ignore bad values
        

    def process_signal(self, buffer):
        """
        Processing the signals to find the bandpower of each region averaged out

        This means we will find delta, theta, alpha, and beta and total for the channels 
        of my choice.
        """  
        data = list(buffer)
        if len(data) < 512:
            return None

        #Storing the values and getting the most recent data
        values = [0, 0, 0, 0, 0]
        recent_data = data[-512:]

        values[0] = round(self.bandpower(recent_data, 256, self.bands['delta'], 2), 2)
        values[1] = round(self.bandpower(recent_data, 256, self.bands['theta'], 2), 2)
        values[2] = round(self.bandpower(recent_data, 256, self.bands['alpha'], 2), 2)
        values[3] = round(self.bandpower(recent_data, 256, self.bands['beta'], 2), 2)
        values[4] = round(self.bandpower(recent_data, 256, [0, 0], 2, total=True))
        return values

    def bandpower(self, data, sf, band, window_sec=None, relative=False, total=False):
        """Compute the average power of the signal x in a specific frequency band
        Args:
            data -- The data we are getting the information from for the bandpower
            sf  -- The amount of samples per second
            band -- The ranges we want to analyze
            window_sec -- Getting the range of the window we want to analyze
            relative -- Set to true to get the perecentage of the total bandpower
            total -- Set to true to get the total bandpower itself  
        Returns:
            list -- Bandpower for the specified section
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
        """
        Getting the HRV from the PPG1 data
        
        Returns:
            dict, dict -- A working dictionary with all necessary data about the users heart rate
            None -- If not enough data or if there is a heart py exeption
        """
        data = list(self.ppg_buffer)
        if len(data) < 20 * self.PPG_SAMPLE_RATE:
            return None #Not enough data to get heart rate

        try:
            filtered_data = hp.filter_signal(data, cutoff=[0.7, 3.5], sample_rate=self.PPG_SAMPLE_RATE, order=3, filtertype='bandpass')

            #Processing the filtered data
            wd, m = hp.process(filtered_data, sample_rate=self.PPG_SAMPLE_RATE)
            return wd, m
        except:
            print("Signal too messy for heart metrics")
            return None #Bad segment

    def find_baevsky_index(self, wd):
        """
        Docstring for find_baevsky_index
        
        Args:
            wd -- Working dictionary from heart py with the RR data necessary to find Baevsky stress
        Returns:
            float -- Baevsky index score
        """
        #Extracting and cleaning the data
        rr_list = wd['RR_list']
        rrs = np.array(rr_list)
        rrs = rrs[rrs > 0]
        #If not enough data
        if len(rrs) < 10:
            return 0.0 
        
        # Set up histogram bins
        bin_width = 50
        min_rr = np.min(rrs)
        max_rr = np.max(rrs)

        #Create bins from min to max with 50ms steps
        bins = np.arange(min_rr, max_rr + bin_width, bin_width)
        #If not enough bins
        if len(bins) < 2:
            return 0.0
        #Calculate histogram
        hist, bin_edges = np.histogram(rrs, bins=bins)

        #Deriving components
        max_bin_index = np.argmax(hist)
        #Mode(Mo): Most frequent RR interval in seconds, dominant heart rhythm
        Mo = (bin_edges[max_bin_index] + bin_edges[max_bin_index+1]) / 2 / 1000.0 #Convert to seconds

        #Amplitude of Mode(AMo): Percent of total beats in the bin
        AMo = (hist[max_bin_index] / len(rrs)) * 100.0
        #Variational Range(MxDMn): Max RR - Min RR in seconds
        MxDMn = (max_rr - min_rr) / 1000.0
        if (MxDMn == 0 or Mo == 0):
            return 0.0
        #Calculate Stress Index(si)
        si = AMo / (2 * Mo * MxDMn)

        return si

    def find_biometric_values(self):
        """
        Finding the biometric values that we will use to check if the user is stressed or not
        
        Returns:
            tuple -- Contains (alpha, beta, ratio, baevsky, hrv, ibi)
        """
        
        #Finding the biometric values, will use this for baseline AND to find the usual values
        result = self.compute_hrv()

        if result == None:
            print("Not enough data...Try again soon.")
            return None
        wd, m = result
        
        #Getting all of the bandpowers for each value
        tp9Values = self.process_signal(self.TP9Buffer)
        tp10Values = self.process_signal(self.TP10Buffer)
        af7Values = self.process_signal(self.AF7Buffer)
        af8Values = self.process_signal(self.AF8Buffer)

        if tp9Values == None or tp10Values == None or af7Values == None or af8Values == None:
            return None

        #Combining the values to get an average
        #{"delta": [0.5, 4], "theta": [4, 8], "alpha": [8, 12], "beta": [12, 30], "total"}
        values = [0, 0, 0, 0, 0] #Delta, Theta, alpha, beta, total 
        for i in range(len(tp9Values)):
            values[i] = (tp9Values[i] + tp10Values[i] + af7Values[i] + af8Values[i]) / 4
        
        #Converting the alpha and beta ratios to be relative
        alpha = round(values[2], 3)
        beta = round(values[3], 3)
        ratio = round(beta / alpha, 3)

        baevsky = self.find_baevsky_index(wd)

        bpm = m['bpm']
        ibi = m['ibi']
        hrv = m['rmssd']

        return {"alpha_waves": alpha, 
                "beta_waves": beta, 
                "alpha_beta_ratio": ratio, 
                "baevsky": baevsky, 
                "bpm": bpm, 
                "hrv": hrv,
                "ibi": ibi}

    def find_baseline(self):
        """
        Finding our baseline values using biometrics
        """
        #Resetting all values
        self.AF7Buffer = []
        self.AF8Buffer = []
        self.TP9Buffer = []
        self.TP10Buffer = []
        self.AUXBuffer = []
        self.ppg_buffer = []
        self.latest_bandpower = []
        self.PPG_WINDOW_SIZE = 30 * self.PPG_SAMPLE_RATE

        calibration_duration = 30 #We will wait 30 seconds to get a proper baseline

        for i in range(calibration_duration):
            time.sleep(1)
            if i % 2 == 0:
                print(f"Calibrating... {calibration_duration-i} seconds left.")

        self.baseline_metrics = self.find_biometric_values() 
        
        if self.baseline_metrics == None:
            print("Something went wrong in baseline metrics, please try again.")
            return None

        print("Baseline established.")
        return self.baseline_metrics

    def preprocess_ppg_buffer(self):
        """This method would be to filter and flag the buffers"""


        return


if __name__ == '__main__':
    print("Starting EEG Receiver Class")
    #Setting up the class
    eeg = EEGReceiver()
    eeg.start()
    time.sleep(1)
    print(eeg.ppg_buffer)
    if len(eeg.AF7Buffer) > 1:
        values = eeg.find_baseline()
        print(f"------Baseline Metrics-------")
        print(f"Alpha Waves: {values['alpha_waves']}")
        print(f"Beta Waves: {values['beta_waves']}")
        print(f"Alpha/Beta Ratio: {values['alpha_beta_ratio']}")

    print("Done")
        

#Potential methods I could introduce later
def get_epoch(window_sec):
    """This method would be if I change how the buffers are stored to get the last little bit of data"""

def get_tilt_score():
    """Using the baseline, we would get the tilt score with this method"""