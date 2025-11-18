from muselsl import stream, list_muses

if __name__ == "__main__":

    muses = list_muses()

    if not muses:
        raise RuntimeError("Cannot find Muse")
    

    stream(muses[0]['address'], ppg_enabled=True) #Only need PPG and EEG data

    # Note: Streaming is synchronous, so code here will not execute until the stream has been closed
    print('Stream has ended')
