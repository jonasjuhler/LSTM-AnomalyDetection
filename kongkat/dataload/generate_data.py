import numpy as np


def gendata(T=96, outlier_type="None"):

    # Set the length of the peak to 65% of sequence length
    peak_len = int(T*0.65)

    # Create spike for solar energy during day
    peak = (np.sin(np.linspace(-1/2*np.pi, 3/2*np.pi, peak_len)) + 1) / 2.2

    # Define number of nighttime (no solar) timesteps
    nighttime = T - peak_len
    t = int((T - peak_len)//2)  # number of timesteps before peak starts

    if outlier_type in ["Snow", "Cloud"]:
        # Define length and normal dist. parameters based on outlier type
        perc = 0.85 if outlier_type == "Snow" else 0.5
        deviation = 0.03 if outlier_type == "Snow" else 0.1
        increase = 0.1 if outlier_type == "Snow" else 0.5

        # Create normal distributed noise instead of spike
        anom_len = int(peak_len*perc)
        anomaly = np.random.randn(anom_len) * deviation + increase
        anomaly[np.where(anomaly < 0)] = 0

        # Insert anomaly into peak
        cut = int((peak_len - anom_len) / 2)
        anom_start_stop = (t+cut-1, t+cut+anom_len)
        if (peak_len - anom_len) % 2 == 0:
            peak[cut:-cut] = anomaly
        else:
            peak[cut:-(cut+1)] = anomaly

    elif outlier_type == "Fault":
        # Define length of anomaly
        anom_len = int(peak_len*0.15)
        # Set a random range of values to 0 to simulate inverter fault
        start = np.random.randint(anom_len, peak_len-anom_len*2)
        stop = start + anom_len
        peak[start:stop] = 0
        anom_start_stop = (t+start-1, t+stop)

    elif outlier_type == "Spike":
        # Define length of anomaly
        anom_len = int(peak_len*0.15)
        # Choose random starting point for spike
        start = np.random.randint(anom_len, peak_len-anom_len*2)
        stop = start + anom_len
        start_h = peak[start]
        stop_h = peak[stop-1]
        start_height_weight = start_h / (stop_h + start_h)
        start_steps = int(np.ceil(start_height_weight * anom_len))
        stop_steps = int(anom_len - start_steps)

        # Create and insert spike
        start_spike = np.linspace(start_h, 0.05, start_steps)
        stop_spike = np.linspace(0.05, stop_h, stop_steps)
        spike = np.concatenate((start_spike, stop_spike))
        peak[start:stop] = spike

        anom_start_stop = (t+start, t+stop)

    elif outlier_type == "Shade":

        # Define length and drop amount of anomaly
        anom_len = int(peak_len*0.15)

        # Choose random starting point for spike
        start = np.random.randint(anom_len, peak_len-(anom_len*2))
        stop = start+anom_len
        start_h = peak[start]
        stop_h = peak[stop-1]
        height_dist = start_h / (stop_h + start_h)
        step_start = int(np.ceil(height_dist * anom_len))
        step_stop = int(anom_len - step_start)

        d_add = np.abs(start_h - stop_h)
        d_coeff = 0.3 + d_add

        # Create and insert shade
        start_spike = np.linspace(1/2*np.pi, 3/2*np.pi, step_start)
        stop_spike = np.linspace(3/2*np.pi, 5/2*np.pi, step_stop)

        # Define height of starting and middle point
        drop_start = d_coeff * height_dist
        lowest = start_h - drop_start
        amp = (start_h - lowest)/2
        start_spike = (amp*np.sin(start_spike) - amp + start_h)

        # Define height of end point
        amp = (stop_h - lowest)/2
        stop_spike = (amp*np.sin(stop_spike) - amp + stop_h)

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
        return energy, (t, t + peak_len)


def data_generator(i=50):
    for _ in range(i):
        yield gendata()


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    types = ["None", "Snow", "Cloud", "Fault", "Spike", "Shade"]

    fig, axes = plt.subplots(2, 3, figsize=(12, 8))

    for i, ax in enumerate(axes.flatten()):
        ts, (start, stop) = gendata(T=96, outlier_type=types[i])
        ax.plot(ts)
        ax.set_ylim(0, 1)
        ax.set_title(types[i])
        #ax.vlines([start, stop], 0, 1)

    plt.show()
