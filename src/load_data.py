import scipy.io as sio
import numpy as np


def load_single_file(path):
    data = sio.loadmat(path, squeeze_me=True, struct_as_record=False)

    # Use DS (Drive Side)
    sensor = data['DS']

    raw_data = np.array(sensor.rawData)      # (N, 16384)
    rpm = np.array(sensor.RPM)               # (N,)
    labels = np.array(sensor.label)          # (N,)
    fs = float(sensor.samplingRate[0])       # 640 Hz

    fault_freqs = sensor.faultFrequencies

    fault_multiples = {
        "BPFI": float(fault_freqs.BPFIMultiple),
        "BPFO": float(fault_freqs.BPFOMultiple),
        "BSF": float(fault_freqs.BPFMultiple),
        "FTF": float(fault_freqs.FTFMultiple),
    }

    # Filter out zero-RPM samples.
    mask = rpm > 0
    raw_data = raw_data[mask]
    rpm = rpm[mask]
    labels = labels[mask]

    return raw_data, rpm, labels, fs, fault_multiples