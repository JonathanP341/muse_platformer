from pylsl import StreamInlet, resolve_byprop
import numpy as np

class Band:
    Delta = 0
    Theta = 1
    Alpha = 2
    Beta = 3

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

def receive_LSL_signal(eeg_inlet, ppg_inlet):
    """ Create inlets and receive the data EEG & PPG from the MUSE headset"""
    eeg_sample = []
    ppg_sample = []

    #Getting the rate of transfer of 
    info = eeg_inlet.info()
    fs = int(info.nominal_srate())

    eeg_chunk, _ = eeg_inlet.pull_chunk(timeout=0.2, max_samples=32)
    ppg_chunk, _ = ppg_inlet.pull_chunk(timeout=0.2, max_samples=8)

    if eeg_chunk:
        valid_rows = [row for row in eeg_chunk if 0.0 not in row]
        if valid_rows:
            eeg_sample = valid_rows[-1]

    if ppg_chunk:
        valid_rows = [row for row in ppg_chunk if 0.0 not in row]
        if valid_rows:
            ppg_sample = valid_rows[-1]
    
    return process_signals(eeg_sample, ppg_sample)

def process_signals(eeg_signals: list, ppg_signals: list):
    """Processing the signals to determine the tilt factor of the player"""  
    # Need to figure out how to get bandpower from the signals, do I need a buffer? well see
        
