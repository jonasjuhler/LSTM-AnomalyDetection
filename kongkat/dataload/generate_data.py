import numpy as np


def gendata(T=96, outlier_type="None"):

    # Create spike for solar energy during day
    peak = (np.sin(np.arange(-1/2*np.pi, -1/2*np.pi+(2*np.pi), 0.10)) + 1) / 2.2
    peak_len = peak.shape[0]

    # Define number of nighttime (no solar) timesteps
    nighttime = T - peak_len
    t = int((T - peak_len)//2)  # number of timesteps before peak starts

    if outlier_type == "Snow":
        # Create normal distributed noise instead of spike and insert
        anomaly = np.random.randn(peak_len - 8) * 0.03 + 0.1
        anomaly[np.where(anomaly < 0)] = 0
        peak[4:-4] = anomaly
        anom_start_stop = (t+3, (t+4)+anomaly.shape[0])
    elif outlier_type == "Cloud":
        # Create normal distributed noise instead of spike and insert
        anomaly = np.random.randn(peak_len - 32) * 0.1 + 0.5
        anomaly[np.where(anomaly < 0)] = 0
        peak[16:-16] = anomaly
        anom_start_stop = (t+15, (t+16)+anomaly.shape[0])
    elif outlier_type == "Fault":
        # Set a random range of 10 values to 10 to simulate inverter fault
        start = np.random.randint(10, peak_len-20)
        stop = start+10
        peak[start:stop] = 0
        anom_start_stop = (t+start-1, t+stop)
    elif outlier_type == "Spike":
        # Choose random starting point for spike
        start = np.random.randint(10, peak_len-20)
        stop = start+10
        step_start = int(
            np.ceil(peak[start] / (peak[stop] + peak[start]) * 10))
        step_stop = int(10 - step_start)

        # Calculate step size
        step_size_start = (peak[start] - 0.05) / step_start
        step_size_stop = (0.05 - peak[stop]) / step_stop

        # Create and insert spike
        start_spike = np.arange(peak[start], 0.05, -step_size_start)
        stop_spike = np.arange(0.05, peak[stop], -step_size_stop)
        spike = np.concatenate((start_spike, stop_spike))
        peak[start:stop] = spike

        anom_start_stop = (t+start, t+stop)

    elif outlier_type == "Shade":

        # Drop coefficient
        d_coeff = 0.5

        # Choose random starting point for spike
        start = np.random.randint(10, peak_len-20)
        stop = start+10
        step_start = int(
            np.ceil(peak[start] / (peak[stop] + peak[start]) * 10))
        step_stop = int(10 - step_start)

        # Calculate step size
        step_size_start = np.pi / step_start
        step_size_stop = np.pi / step_stop

        # Create and insert shade
        start_spike = np.arange(1/2*np.pi, 3/2*np.pi, step_size_start)
        stop_spike = np.arange(3/2*np.pi, 5/2*np.pi, step_size_stop)

        # Define height of starting and middle point
        drop_start = d_coeff * (peak[start] / (peak[start] + peak[stop]))
        lowest = peak[start] - drop_start
        amp = (peak[start] - lowest)/2
        start_spike = (amp*np.sin(start_spike) - amp + peak[start])

        # Define height of end point
        amp = (peak[stop] - lowest)/2
        stop_spike = (amp*np.sin(stop_spike) - amp + peak[stop])

        spike = np.concatenate((start_spike, stop_spike))
        peak[start:stop] = spike

        anom_start_stop = (t+start, t+stop)

    # Add nighttime s.t. length of sequence is 96
    if nighttime % 2 == 0:
        energy = np.concatenate((np.zeros(t), peak, np.zeros(t)))
    else:
        energy = np.concatenate((np.zeros(t), peak, np.zeros(t+1)))

    if outlier_type != "None":
        return energy, anom_start_stop
    else:
        return energy
