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

def find_biometric_values(AF7Buffer, AF8Buffer, compute_hrv, process_signal):
        """
        Finding the biometric values that we will use to check if the user is stressed or not
        
        Returns:
            dict -- Contains (alpha, beta, ratio, baevsky, hrv, ibi)
        """
        #Finding the biometric values, will use this for baseline AND to find the usual values
        result = compute_hrv()

        if result != None:
            wd, m = result
            baevsky = find_baevsky_index(wd)

            bpm = m['bpm']
            ibi = m['ibi']
            rmssd = m['rmssd']
        else:
            print("Not enough data for heart rate data...Using dummy variables.")
            bpm = -1
            ibi = -1
            rmssd = -1
            baevsky = -1
        
        #Getting all of the bandpowers for each value
        af7Values = process_signal(AF7Buffer)
        af8Values = process_signal(AF8Buffer)

        if af7Values == None or af8Values == None:
            print("The EEG channels do not have enough data and are currently None, try again soon.")
            return None

        #Combining the values to get an average so that we can save incase we have bad data in the future
        #{"delta": [0.5, 4], "theta": [4, 8], "alpha": [8, 12], "beta": [12, 30], "total"} 
        alpha = (af7Values['alpha'] + af8Values['alpha']) / 2.0
        beta = (af7Values['beta'] + af8Values['beta']) / 2.0
        theta = (af7Values['theta'] + af8Values['theta']) / 2.0
        total = (af7Values['total'] + af8Values['total']) / 2.0
        latest_bandpower = {"alpha": alpha, "beta": beta, "theta": theta, "total": total}

        baseline = {"af7_alpha": af7Values['alpha'],
                    "af7_beta": af7Values['beta'],
                    "af7_theta": af7Values['theta'],
                    "af7_total": af7Values['total'],
                    "af8_alpha": af8Values['alpha'],
                    "af8_beta": af8Values['beta'],
                    "af8_theta": af8Values['theta'],
                    "af8_total": af8Values['total'],
                    "baevsky": baevsky, 
                    "bpm": bpm, 
                    "rmssd": rmssd,
                    "ibi": ibi}
        for key in baseline.keys():
            baseline[key] = round(baseline[key], 5)
        return baseline

def find_baevsky_index(wd):
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

def compute_hrv(ppg_buffer, PPG_SAMPLE_RATE):
        """
        Getting the HRV from the PPG1 data
        
        Returns:
            dict, dict -- A working dictionary with all necessary data about the users heart rate
            None -- If not enough data or if there is a heart py exeption
        """
        data = list(ppg_buffer)
        if len(data) < 20 * PPG_SAMPLE_RATE:
            return None #Not enough data to get heart rate

        try:
            #Restricting range to BPM of 45-150
            filtered_data = hp.filter_signal(data, cutoff=[0.75, 2.5], sample_rate=PPG_SAMPLE_RATE, order=4, filtertype='bandpass')

            #Processing the filtered data
            wd, m = hp.process(filtered_data, sample_rate=PPG_SAMPLE_RATE)
            return wd, m
        except:
            print("Signal too messy for heart metrics")
            return None #Bad segment