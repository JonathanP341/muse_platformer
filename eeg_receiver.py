import warnings
warnings.filterwarnings("ignore", category=UserWarning, module='pkg_resources')

from pythonosc.dispatcher import Dispatcher
from pythonosc.osc_server import BlockingOSCUDPServer

from scipy.signal import welch, periodogram
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
        self.PPG_WINDOW_SIZE = 18 * self.PPG_SAMPLE_RATE
        self.EEG_SAMPLE_RATE = 256
        self.EEG_WINDOW_SIZE = 3 * self.EEG_SAMPLE_RATE

        #Synchronous values
        self.AF7Buffer = deque(maxlen=self.EEG_WINDOW_SIZE) 
        self.AF8Buffer = deque(maxlen=self.EEG_WINDOW_SIZE) 
        self.ppg_buffer = deque(maxlen=self.PPG_WINDOW_SIZE) #PPG2 only

        self.latest_bandpower = {}
        self.latest_bpm = -1
        self.previous_tilt_score = 0.0

        #Baseline values
        self.baseline_metrics = {"nasa": 0.0, "faa": 0.0, "bpm": 0.0}
        self.baseline_tilt_score = 1 #Set to 1 to avoid divsion by 0 issues

        # Set up for the dispatcher 
        self.dispatcher = Dispatcher()
        self.dispatcher.map("/muse/eeg", self.on_eeg)
        self.dispatcher.map("/muse/ppg", self.on_ppg)
        self.server = BlockingOSCUDPServer((ip, port), self.dispatcher)

    def start(self):
        """Starting the OSC Server on a background thread"""
        thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        thread.start()
    
    def on_eeg(self, address, *values):
        """
        Processing the EEG data by putting it in the respective buffers

        address -- The address the values are being sent to 
        values -- Contains an array with 5 values [TP9, AF7, AF8, TP10, AUX], we ignore AUX, TP9 and TP10
        """
        raw_values = np.array(values[1:3]) #Ignoring TP9, TP10 and AUX
        eeg_data = np.array(raw_values, dtype=float) 

        #Scrubbing out bad values
        if np.isnan(eeg_data).any() or (eeg_data == 0.0).any():
            return 
        
        self.AF7Buffer.append(eeg_data[0])
        self.AF8Buffer.append(eeg_data[1])
    
    def on_ppg(self, address, *values):
        """
        Processing the PPG data by appending it to its respective buffers, only considering
        PPG1 data as its the most reliable
        Args:
            address -- The address the values are being sent to 
            values -- Contains an array with 3 values [PPG1, PPG2, PPG3], we ignore PPG1/3
        """
        try:
            val = float(values[1])
            if not np.isnan(val) and val != 0.0:
                self.ppg_buffer.append(val) #Only storing PPG2
        except (ValueError, TypeError):
            pass #Ignore bad values    

    def process_eeg_signal(self, buffer):
        """
        Processing the signals to find the bandpower of each region averaged out

        This means we will find delta, theta, alpha, and beta and total for the channels 
        of my choice.
        """  
        data = list(buffer)
        if len(data) < 512:
            print("Not enough data in the buffer to process signal.")
            return None
        recent_data = np.array(data[-512:])
        
        delta = round(self.bandpower(recent_data, 256, self.bands['delta'], 2), 2)

        if delta > 5000: #Arbitrary threshold to remove bad data
            print("Delta bandpower too high, likely bad data.")
            #Just returning the previous data to avoid crash
            return self.latest_bandpower if self.latest_bandpower != {} else None
        
        theta = round(self.bandpower(recent_data, 256, self.bands['theta'], 2), 2)
        alpha = round(self.bandpower(recent_data, 256, self.bands['alpha'], 2), 2)
        beta = round(self.bandpower(recent_data, 256, self.bands['beta'], 2), 2)
        total = theta + alpha + beta

        return {"alpha": alpha, "beta": beta, "theta": theta, "total": total}

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
        #Compute periodogram
        freqs, psd = periodogram(data, sf)
        # Frequency resolution
        freq_res = freqs[1] - freqs[0]
        total_power = simpson(psd, dx=freq_res)

        if total: #Returning the total value, no index band
            return total_power
        
        #Find closest indecies of band in freq vector
        idx_band = np.logical_and(freqs >= low, freqs <= high)
        #Integral approx of area
        bp = simpson(psd[idx_band], dx=freq_res)

        if relative:
            if total_power == 0:
                return 0.0
            bp /= simpson(psd, dx=freq_res)
        return bp

    def process_ppg_signal(self, buffer):
        data = np.array(buffer)
        
        if len(data) < self.PPG_SAMPLE_RATE * 16:
            return None
        sampling_rate = 55 #Aproximating sampling rate, generally tends to be around 50-55 on BCI-Py
        filtered = hp.filter_signal(data, cutoff=[0.75, 2.5], sample_rate=52.5, order=4, filtertype='bandpass')
        wd, m = hp.process(filtered, sample_rate=sampling_rate)

        if 'bpm' in m:
            return m['bpm']
        else:
            return None


    def find_baseline(self):
        """
        Finding our baseline values using biometrics
        """
        print("Clearing buffers for calibration...")
        # Use clear() to keep the deque properties (maxlen)
        self.AF7Buffer.clear()
        self.AF8Buffer.clear()
        self.ppg_buffer.clear()
        self.latest_bandpower = {}
            
        # Calibration settings
        calibration_duration = 30 
            
        # Wait for buffers to fill up before starting (we need ~2 seconds of data)
        print("Waiting for data buffers to fill...")
        while len(self.AF7Buffer) < 512: # Need at least 512 samples for process_eeg_signal
            time.sleep(0.1)

        print("Buffers filled. Starting calibration...")

        nasa_engagement_sum = 0.0
        faa_sum = 0.0
        bpm_sum = 0.0
        valid_samples = 0
        valid_bpm_samples = 0

        # Change '_' to 'i' so we can use it in the print statement
        for i in range(calibration_duration):
            # 1. Get the result
            result = self.get_raw_tilt_score(self.AF7Buffer, self.AF8Buffer, self.ppg_buffer)
                
            # 2. Check if valid (process_eeg_signal returns None if data is bad/empty)
            if result is not None:
                nasa, faa, bpm = result
                    
                # 3. Add separately (Fixes the SyntaxError)
                nasa_engagement_sum += nasa
                faa_sum += faa
                valid_samples += 1

                if bpm is not None:
                    bpm_sum += bpm
                    valid_bpm_samples += 1   
            time.sleep(1)
                
            if i % 2 == 0:
                print(f"Calibrating... {calibration_duration - i} seconds left.")

        # Prevent division by zero if something went wrong
        if valid_samples > 0:
            self.baseline_metrics['nasa'] = nasa_engagement_sum / valid_samples
            self.baseline_metrics['faa'] = faa_sum / valid_samples
        else:
            print("Warning: Calibration failed (no valid data). Using default baselines.")
            self.baseline_metrics['nasa'] = 1.0
            self.baseline_metrics['faa'] = 0.0
        
        if valid_bpm_samples > 0:
            self.baseline_metrics['bpm'] = bpm_sum / valid_bpm_samples
        else:
            print("Warning: No valid BPM data found during calibration. Using default BPM.")
            self.baseline_metrics['bpm'] = 73.0 

        print(f"Baseline established: {self.baseline_metrics}")
        return self.baseline_metrics

    def get_raw_tilt_score(self, af7, af8, ppg):
        """
        Getting the raw tilt score without comparing to baseline, since there likely is no baseline yet
        Returns:
            float, float -- NASA engagement index and FAA values
        """
        left = self.process_eeg_signal(af7)
        right = self.process_eeg_signal(af8)

        if left == None or right == None:
            return None

        alpha = (left['alpha'] + right['alpha']) / 2.0
        beta = (left['beta'] + right['beta']) / 2.0
        theta = (left['theta'] + right['theta']) / 2.0
        total = (left['total'] + right['total']) / 2.0
        self.latest_bandpower = {"alpha": alpha, "beta": beta, "theta": theta, "total": total}

        #Finding the NASA Engagement Index using formula Beta / Alpha + Theta
        L_engagement = (left['beta'] + 1e-6) / (left['alpha'] + left['theta'] + 1e-6)
        R_engagement = (right['beta'] + 1e-6) / (right['alpha'] + right['theta'] + 1e-6)
        engagement_index = (L_engagement + R_engagement) / 2.0

        #Finding the Frontal Alpha Symmetry, ln(Right Alpha) - ln(Left Alpha)
        faa = np.log(right['alpha'] + 1e-6) - np.log(left['alpha'] + 1e-6)

        bpm = self.process_ppg_signal(ppg)
        if bpm is not None:
            self.latest_bpm = bpm

        return engagement_index, faa, bpm

    def get_tilt_score(self, compare_to_baseline=True):
        """
        Using the baseline, we would get the tilt score with this method.

        We are primarily comparing the FAA, NASA stress index and BPM to see if the user is stressed.

        0.0 is zen, monk in the himalayas and 1.0 is full tilt literally punching your monitor

        In the future I could use machine learning to determine this score better, for now thresholds will work.  
        """
        current = self.get_raw_tilt_score(self.AF7Buffer, self.AF8Buffer, self.ppg_buffer)

        if current == None:
            return self.previous_tilt_score
        nasa, faa, bpm = current

        # A. NASA Index: Percentage Increase
        nasa_change = (nasa - self.baseline_metrics["nasa"]) / self.baseline_metrics["nasa"]
        workload_score = min(max(0.0, nasa_change), 1.0)

        # B. FAA: Absolute Drop <- Have a feeling this is weird
        faa_drop = self.baseline_metrics['faa'] - faa
        emotion_score = min(max(0.0, faa_drop) / 0.3, 1.0)

        # C. BPM: Percent Increase
        final_tilt = 0.0
        if bpm is not None:
            bpm_change = (bpm - self.baseline_metrics['bpm']) / self.baseline_metrics['bpm']
            bpm_score = min(max(0.0, bpm_change), 1.0)

            final_tilt = (workload_score * 0.3) + (emotion_score * 0.4) + (bpm_score * 0.3)
        else:
            final_tilt = (workload_score * 0.4) + (emotion_score * 0.6)

        self.previous_tilt_score = final_tilt
        return final_tilt


if __name__ == '__main__':
    print("Starting EEG Receiver Class")
    #Setting up the class
    eeg = EEGReceiver()
    eeg.start()
    while len(eeg.AF7Buffer) < 1:
        time.sleep(0.5)
    if len(eeg.AF7Buffer) > 1:
        values = eeg.find_baseline()
        print(values)
    tilt = eeg.get_tilt_score()
    print(f"Length AF7: {len(eeg.AF7Buffer)}, Length AF8: {len(eeg.AF8Buffer)}, Length PPG: {len(eeg.ppg_buffer)}")
    print(f"Tilt Score: {tilt}")
    time.sleep(10) #Waiting another 10 seconds to update data
    tilt = eeg.get_tilt_score()
    print(f"Tilt Score: {tilt}")
    print(f"Length AF7: {len(eeg.AF7Buffer)}, Length AF8: {len(eeg.AF8Buffer)}, Length PPG: {len(eeg.ppg_buffer)}")
    for i in range(5):
        print("sleeping 5 seconds...")
        time.sleep(5)
        tilt = eeg.get_tilt_score()
        print(f"Tilt Score: {tilt}")
        print(f"Length AF7: {len(eeg.AF7Buffer)}, Length AF8: {len(eeg.AF8Buffer)}, Length PPG: {len(eeg.ppg_buffer)}")
    print("Done.")
