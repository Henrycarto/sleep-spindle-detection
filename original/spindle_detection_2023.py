import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import detrend, butter, filtfilt, find_peaks, sosfilt
from mne.time_frequency import morlet
from mne.time_frequency import psd_array_multitaper
from pyedflib import EdfReader
from sklearn.model_selection import train_test_split

def split_data(eeg_data_paths):
    train_paths, test_paths = train_test_split(eeg_data_paths, test_size=0.5, random_state=42)
    return train_paths, test_paths

def run_experiment(train_paths, test_paths):
    # Initialize variables to store metrics across all files
    all_training_sensitivity1, all_training_precision1, all_training_f1_score1 = [], [], []
    all_training_sensitivity2, all_training_precision2, all_training_f1_score2 = [], [], []

    all_test_sensitivity1, all_test_precision1, all_test_f1_score1 = [], [], []
    all_test_sensitivity2, all_test_precision2, all_test_f1_score2 = [], [], []

    for eeg_data_path in train_paths:

        print(f"Current train excerpt: {eeg_data_path}")

        eeg_data = EdfReader(eeg_data_path)
        eeg_signal = eeg_data.readSignal(0)
        scorer1_data = eeg_data.readSignal(1)
        scorer2_data = eeg_data.readSignal(2)

        sf = len(eeg_signal) / eeg_data.getFileDuration()

        # Perform spindle detection on training set
        detected_spindles = detect_spindles(eeg_signal, sf)

        # Convert binary scorer to 2D array
        sf1 = len(scorer1_data) / eeg_data.getFileDuration()
        sf2 = len(scorer2_data) / eeg_data.getFileDuration()
        scorer1_events = convert_binary_scorer_to_2d_array(scorer1_data, sf1)
        scorer2_events = convert_binary_scorer_to_2d_array(scorer2_data, sf2)

        # Calculate metrics for each scorer
        precision1, sensitivity1 = calculate_event_based_metrics(detected_spindles, scorer1_events, sf)
        precision2, sensitivity2 = calculate_event_based_metrics(detected_spindles, scorer2_events, sf)

        # Store metrics
        all_training_sensitivity1.append(sensitivity1)
        all_training_precision1.append(precision1)
        all_training_f1_score1.append(calculate_f1_score(precision1, sensitivity1))

        all_training_sensitivity2.append(sensitivity2)
        all_training_precision2.append(precision2)
        all_training_f1_score2.append(calculate_f1_score(precision2, sensitivity2))

    for eeg_data_path in test_paths:

        print(f"Current train excerpt: {eeg_data_path}")

        eeg_data = EdfReader(eeg_data_path)
        eeg_signal = eeg_data.readSignal(0)
        scorer1_data = eeg_data.readSignal(1)
        scorer2_data = eeg_data.readSignal(2)

        sf = len(eeg_signal) / eeg_data.getFileDuration()

        # Perform spindle detection on training set
        detected_spindles = detect_spindles(eeg_signal, sf)

        # Convert binary scorer to 2D array
        sf1 = len(scorer1_data) / eeg_data.getFileDuration()
        sf2 = len(scorer2_data) / eeg_data.getFileDuration()
        scorer1_events = convert_binary_scorer_to_2d_array(scorer1_data, sf1)
        scorer2_events = convert_binary_scorer_to_2d_array(scorer2_data, sf2)

        # Calculate metrics for each scorer
        precision1, sensitivity1 = calculate_event_based_metrics(detected_spindles, scorer1_events, sf)
        precision2, sensitivity2 = calculate_event_based_metrics(detected_spindles, scorer2_events, sf)

        # Store metrics
        all_test_sensitivity1.append(sensitivity1)
        all_test_precision1.append(precision1)
        all_test_f1_score1.append(calculate_f1_score(precision1, sensitivity1))

        all_test_sensitivity2.append(sensitivity2)
        all_test_precision2.append(precision2)
        all_test_f1_score2.append(calculate_f1_score(precision2, sensitivity2))

    # Average metrics across all files for each scorer
    avg_training_sensitivity1 = np.mean(all_training_sensitivity1)
    avg_training_precision1 = np.mean(all_training_precision1)
    avg_training_f1_score1 = np.mean(all_training_f1_score1)

    avg_training_sensitivity2 = np.mean(all_training_sensitivity2)
    avg_training_precision2 = np.mean(all_training_precision2)
    avg_training_f1_score2 = np.mean(all_training_f1_score2)

    print("Training Data:")

    print(f"Avg. Training Sensitivity Scorer 1: {avg_training_sensitivity1 * 100}%")
    print(f"Avg. Training Precision Scorer 1: {avg_training_precision1 * 100}%")
    print(f"Avg. Training F1 Score Scorer 1: {avg_training_f1_score1 * 100}%")

    print(f"Avg. Training Sensitivity Scorer 2: {avg_training_sensitivity2 * 100}%")
    print(f"Avg. Training Precision Scorer 2: {avg_training_precision2 * 100}%")
    print(f"Avg. Training F1 Score Scorer 2: {avg_training_f1_score2 * 100}%")

    # Average metrics across all files for each scorer
    avg_test_sensitivity1 = np.mean(all_test_sensitivity1)
    avg_test_precision1 = np.mean(all_test_precision1)
    avg_test_f1_score1 = np.mean(all_test_f1_score1)

    avg_test_sensitivity2 = np.mean(all_test_sensitivity2)
    avg_test_precision2 = np.mean(all_test_precision2)
    avg_test_f1_score2 = np.mean(all_test_f1_score2)

    print("Test Data:")

    print(f"Avg. Test Sensitivity Scorer 1: {avg_test_sensitivity1 * 100}%")
    print(f"Avg. Test Precision Scorer 1: {avg_test_precision1 * 100}%")
    print(f"Avg. Test F1 Score Scorer 1: {avg_test_f1_score1 * 100}%")

    print(f"Avg. Test Sensitivity Scorer 2: {avg_test_sensitivity2 * 100}%")
    print(f"Avg. Test Precision Scorer 2: {avg_test_precision2 * 100}%")
    print(f"Avg. Test F1 Score Scorer 2: {avg_test_f1_score2 * 100}%")


def calculate_f1_score(precision, sensitivity):
    if precision + sensitivity == 0:
        return 0
    f1_score = 2 * (precision * sensitivity) / (precision + sensitivity)
    return f1_score

def calculate_event_based_metrics(detected_spindles, scorer, sf):
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    true_negatives = 0

    for spindle in detected_spindles:
        start_time, duration = spindle

        start_time = start_time / sf

        end_time = start_time + duration

        # Check if the spindle overlaps with any scorer event
        overlapping_events = [event for event in scorer if event[0] <= end_time and (event[0] >= start_time or event[0] + event[1] >= start_time)]

        if len(overlapping_events) > 0:
            true_positives += 1
        else:
            false_positives += 1

    # Calculate false negatives and true negatives
    for event in scorer:
        start_time, duration = event
        end_time = start_time + duration
        false_spindles = [spindle for spindle in detected_spindles if spindle[0] <= end_time and spindle[0] + spindle[1] >= start_time]

        if len(false_spindles) == 0:
            false_negatives += 1

    precision = true_positives / (true_positives + false_positives)
    sensitivity = true_positives / (true_positives + false_negatives)

    return precision, sensitivity

def convert_binary_scorer_to_2d_array(binary_array, sample_rate):
    events = []
    in_event = False
    start_time = None

    for i, value in enumerate(binary_array):
        if value == 1 and not in_event:
            in_event = True
            start_time = i / sample_rate
        elif value == 0 and in_event:
            in_event = False
            end_time = i / sample_rate
            duration = end_time - start_time
            events.append([start_time, duration])

    # Check if there's an event still in progress at the end of the array
    if in_event:
        end_time = len(binary_array) / sample_rate
        duration = end_time - start_time
        events.append([start_time, duration])

    return events

def downsample_detected_spindle(detected_binary, target_length):
    factor = len(detected_binary) // target_length
    downsampled = np.mean(detected_binary.reshape(-1, factor), axis=1)
    for i in range(len(downsampled)):
        if(downsampled[i] != 0):
            downsampled[i] = 1
    return downsampled

def calculate_metrics(detected_binary, ground_truth):
    # Calculate True Positives, False Negatives, and False Positives
    true_positives = np.sum(np.logical_and(ground_truth, detected_binary))
    false_negatives = np.sum(np.logical_and(ground_truth, np.logical_not(detected_binary)))
    false_positives = np.sum(np.logical_and(np.logical_not(ground_truth), detected_binary))

    # Calculate sensitivity and accuracy
    sensitivity = true_positives / (true_positives + false_negatives)
    accuracy = (true_positives + np.sum(np.logical_not(ground_truth))) / len(ground_truth)
    precision = true_positives / (true_positives + false_positives)

    return sensitivity, accuracy, precision

def create_binary_array(detected_spindles, total_samples):
    # Filter out spindles with value 0
    filtered_spindles = [spindle for spindle in detected_spindles if spindle[0] != 0]

    binary_array = np.zeros(total_samples)
    for spindle in filtered_spindles:
        start_index = int(spindle[0])  # Assuming 10 samples per second
        end_index = int((spindle[0] + spindle[1]))  # Assuming 10 samples per second
        binary_array[start_index:end_index] = 1
    return binary_array

def plot_binary_array(binary_array):
    plt.figure(figsize=(8, 2))
    plt.plot(binary_array, color='black', linewidth=2)
    plt.ylim(-0.1, 1.1)  # Set y-axis limits to show 0 and 1
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.title('Binary Array')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show()

# Method for plotting the original data
def plot_original(eeg_signal):
    # plot the original data
    plt.figure(figsize=[15, 6], dpi=200)
    plt.figure()
    plt.subplot(5, 1, 1)
    plt.plot(eeg_signal, lw=0.5)
    plt.ylim([-100, 100])
    plt.title('Original EEG')
    plt.show()
    return 0

#Method for plotting the template spindle
def plot_template(spindle,fs):
    t = np.arange(spindle.size) / fs
    fig, ax = plt.subplots(1, 1, figsize=(7, 4))
    ax.plot(t, spindle)
    plt.ylim(-0.4, 0.4)
    plt.xlim(t[0], t[-1])
    plt.xlabel('Time [seconds]')
    plt.ylabel('Amplitude [a.u.]')
    plt.show()
    return 0

#Method for plotting the original data with suspected spindles
def plot_sus_spindles(filtered,val_spindles):
    plt.figure(figsize=[15, 6], dpi=200)
    plt.figure()
    plt.subplot(5, 1, 1)
    plt.plot(filtered, lw=0.5)
    plt.plot(val_spindles, color='indianred', alpha=.8)
    plt.ylim([-100, 100])
    plt.title('Original EEG')
    plt.show()
    return 0

#Method for creating a dictionary and to print it
def create_dic(sp_dur, sp_freq, sp_amp):
    sp_params = {'Duration (ms)' : sp_dur, 'Frequency (Hz)': sp_freq,
                 'Amplitude (uV)': sp_amp}
    print(sp_params)
    return 0

# Input:
# eeg: Contains the input EEG signal (that was created with the EdfReaders readSignal() function)
# fs: Sampling rate of the EEG signal
# Output:
# spindles: Contains a two dimensional numpy array where
# the first dimension indicates the spindle event and
# the second dimension is 2 and contains in the first column the start time
# of the spindle event in seconds from the start of the recording
# and the second column contains the duration of the spindle in seconds.
def detect_spindles(eeg, fs):

    combined_spindles = []

    # Apply bandpass filter
    filtered_data = butter(8, [11,16], btype='band', output='sos', fs=fs)
    filtered = sosfilt(filtered_data, eeg)

    nc = 20     # Number of oscillations in the spindles
    j = 0       # Variable to determine the position in the spindle array

    # Compute the power spectrum and find the peak 11-16 Hz frequency.
    psd, freqs = psd_array_multitaper(filtered, sf, fmin=11, fmax=16, verbose=0)
    cf = freqs[np.argmax(psd)]

    print('Central frequency: %.2f Hz' % cf)

    # Compute the wavelet
    wlt = morlet(fs, [cf], n_cycles=nc)[0]

    # Convolve the wavelet and extract magnitude and phase
    analytic = np.convolve(eeg, wlt, mode='same')
    magnitude = np.abs(analytic)
    phase = np.angle(analytic)

    # Square and normalize the magnitude from 0 to 1 (using the min and max)
    power = np.square(magnitude)
    norm_power = (power - power.min()) / (power.max() - power.min())

    # Define the threshold
    thresh = 0.05

    # Find supra-threshold values
    supra_thresh = np.where(norm_power >= thresh)[0]

    # Create vector for plotting purposes
    val_spindles = np.nan * np.zeros(len(eeg))
    val_spindles[supra_thresh] = eeg[supra_thresh]

    # Extract start and end of each spindles
    sp = np.split(supra_thresh, np.where(np.diff(supra_thresh) != 1)[0] + 1)
    idx_start_end = np.array([[k[0], k[-1]] for k in sp])

    # Extract the duration (in ms) of each spindles
    sp_dur = (np.diff(idx_start_end, axis=1) / fs * 1000).flatten()

    # Extract the peak-to-peak amplitude and frequency
    sp_amp, sp_freq = np.zeros(len(sp)), np.zeros(len(sp))

    for i in range(len(sp)):
        # Important: detrend the signal to avoid wrong peak-to-peak amplitude
        sp_amp[i] = np.ptp(detrend(eeg[sp[i]]))

        # Median of the instantaneous frequency of the spindles, where:
        # inst_freq = sf / 2pi * 1st-derivative of the phase of the analytic signal
        sp_freq[i] = np.median((fs / (2 * np.pi) * np.diff(phase[sp[i]])))

    # Create a dictionnary and print it out
    create_dic(sp_dur, sp_freq, sp_amp)

    if j == 0:
        spindles = np.zeros((len(idx_start_end), 2))

    for i in range(len(idx_start_end)):
        if sp_freq[i] <= 16:
            if sp_freq[i] >= 11:
                if sp_dur[i] < 3000.:
                    if sp_dur[i] > 500.:
                        spindles[j][0] = idx_start_end[i][0]
                        spindles[j][1] = sp_dur[i] / 1000
                        j = j + 1
    sorted_spindles = sorted(spindles, key=lambda x:x[0])
    # Initialize variables for tracking the last spindle
    last_end_time = None
    last_spindle = None

    for spindle in sorted_spindles:
        start_time, duration = spindle

        if last_end_time is not None and start_time - last_end_time < 2:
            # Spindles are too close, compare durations
            if duration > last_spindle[1]:
                # Keep the spindle with longer duration
                last_spindle = spindle
        else:
            if last_spindle is not None:
                combined_spindles.append(last_spindle)
            last_spindle = spindle

        last_end_time = start_time + duration

    if last_spindle is not None:
        combined_spindles.append(last_spindle)
    filtered_spindles = [spindle for spindle in combined_spindles if spindle[0] != 0]
    return filtered_spindles

# Load the data
eeg_data = EdfReader('trainingsData/excerpt5.prep.edf')

eeg_signal = eeg_data.readSignal(0)

# Define sampling frequency
sf = len(eeg_data.readSignal(0)) / eeg_data.getFileDuration()

#call detection algorithm
detected_spindles = detect_spindles(eeg_signal,sf)

for i in range(len(detected_spindles)):
    if detected_spindles[i][0] != 0:
        print('Detected Spindle at ' + str(detected_spindles[i][0] / sf) + ' s with a duration of ' + str(detected_spindles[i][1]) + 's')

print(len(detected_spindles))

scorer1_data = eeg_data.readSignal(1)
scorer2_data = eeg_data.readSignal(2)

sf1 = len(eeg_data.readSignal(1)) / eeg_data.getFileDuration()
sf2 = len(eeg_data.readSignal(2)) / eeg_data.getFileDuration()

scorer1_events = convert_binary_scorer_to_2d_array(scorer1_data, sf1)
scorer2_events = convert_binary_scorer_to_2d_array(scorer2_data, sf2)

precision1, sensitivity1 = calculate_event_based_metrics(detected_spindles, scorer1_events, sf)
precision2, sensitivity2 = calculate_event_based_metrics(detected_spindles, scorer2_events, sf)

print(f"Precision Scorer 1: {precision1 * 100}%")
print(f"Sensitivity Scorer 1: {sensitivity1 * 100}%")

print(f"Precision Scorer 2: {precision2 * 100}%")
print(f"Sensitivity Scorer 2: {sensitivity2 * 100}%")

scorer1_f1_score = calculate_f1_score(precision1, sensitivity1)
scorer2_f1_score = calculate_f1_score(precision2, sensitivity2)

print(f"Scorer 1 F1 score: {scorer1_f1_score * 100}%")
print(f"Scorer 2 F1 score: {scorer2_f1_score * 100}%")

eeg_data.close()

file_paths = [
    'trainingsData/excerpt1.prep.edf',
    'trainingsData/excerpt2.prep.edf',
    'trainingsData/excerpt3.prep.edf',
    'trainingsData/excerpt4.prep.edf',
    'trainingsData/excerpt5.prep.edf',
    'trainingsData/excerpt6.prep.edf'
]

# Call split_data method
train_paths, test_paths = split_data(file_paths)

run_experiment(train_paths, test_paths)

















