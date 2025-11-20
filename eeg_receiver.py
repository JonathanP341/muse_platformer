from pylsl import StreamInlet, resolve_byprop
import numpy as np
from scipy.signal import welch
from scipy.integrate import simpson

bands = {
    "delta": [0.5, 4],
    "theta": [4, 8],
    "alpha": [8, 12],
    "beta": [12, 30]
}

TP9Buffer = [] #LT
AF7Buffer = [] #LF
AF8Buffer = [] #RF
TP10Buffer = [] #RT
AUXBuffer = [] #X
ppgBufferRaw = [0, 0, 0]


def create_and_resolve_inlet(name):
    """Create a new inlet to receive data from based on the specified type

    Keyword Arugments:
    name -- The type of data to create an inlet for. Will be PPG or EEG for now
    """
    streams = resolve_byprop('type', name)
    if len(streams) == 0:
        print("No streams found")
        raise RuntimeError("Cannot find stream")
    inlet = StreamInlet(streams[0])
    return inlet

def receive_LSL_signal():
    """ 
    Create inlets and receive the data EEG & PPG from the MUSE headset

    This method will be run in the background as a separate thread
    """
    eeg_inlet = create_and_resolve_inlet('EEG')
    ppg_inlet = create_and_resolve_inlet('PPG')
    eeg_sample = []
    ppg_sample = []

    #Running a loop to
    while True:
        eeg_chunk, _ = eeg_inlet.pull_chunk(timeout=0.02, max_samples=32)
        ppg_chunk, _ = ppg_inlet.pull_chunk(timeout=0.02, max_samples=8)

        if eeg_chunk:
            valid_rows = [row for row in eeg_chunk if 0.0 not in row]
            if valid_rows:
                eeg_sample = valid_rows[-1]
                process_eeg_data(eeg_sample)

        if ppg_chunk:
            valid_rows = [row for row in ppg_chunk if 0.0 not in row]
            if valid_rows:
                ppg_sample = valid_rows[-1]
                process_ppg_data(ppg_sample)

def process_eeg_data(eeg_sample):
    """
    Processing the EEG data by putting it in the respective buffers

    eeg_sample -- Contains an array with 5 values [TP9, AF7, AF8, TP10, AUX]
    """
    TP9Buffer.append(eeg_sample[0])
    AF7Buffer.append(eeg_sample[1])
    AF8Buffer.append(eeg_sample[2])
    TP10Buffer.append(eeg_sample[3])
    AUXBuffer.append(eeg_sample[4])

def process_ppg_data(ppg_sample):
    ppgBufferRaw.append(ppg_sample)


def process_signal(buffer, win_sec):
    """
    Processing the signals to find the bandpower of each region averaged out

    This means we will find delta, theta, alpha, and beta(no gamma) for the channels 
    of my choice. For this I will just do AF7 and AF8 because they are the most 
    reliable on a muse headset.
    """  
    values = [0, 0, 0, 0, 0]
    if len(buffer) == 512:
        values[0] = round(bandpower(buffer, 256, bands['delta'], 2), 2)
        values[1] = round(bandpower(buffer, 256, bands['theta'], 2), 2)
        values[2] = round(bandpower(buffer, 256, bands['alpha'], 2), 2)
        values[3] = round(bandpower(buffer, 256, bands['beta'], 2), 2)
        values[4] = round(bandpower(buffer, 256, [0, 0], 2, total=True))
        del buffer[:22]
    return values


def bandpower(data, sf, band, window_sec=None, relative=False, total=False):
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

def compute_hrv():
    """Getting the HRV from the PPG data"""


#Potential methods I could introduce later
def get_epoch(window_sec):
    """This method would be if I change how the buffers are stored to get the last little bit of data"""

def preprocess_buffers():
    """This method would be to filter and flag the buffers"""

def find_baseline():
    """This would be to find the users baseline so we can compare"""

def get_tilt_score():
    """Using the baseline, we would get the tilt score with this method"""