"""https://mne.tools/stable/auto_tutorials/evoked/30_eeg_erp.html"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mne


# ERP stuff

data_path = '/to/be/set'
raw_data = mne.io.read_raw_eeglab(data_path, preload=True, verbose=False)

## Pre-process

### Average re-referencing
mne.set_eeg_reference(raw_data, 'average', copy=False, verbose=False)

### High pass filtering to remove drifts
raw_data.filter(l_freq=0.1, h_freq=None)

### Epoch data using markers (from -0.3s to +0.7s)
events, event_id = mne.events_from_annotations(raw_data, event_id='auto', verbose=False)
epochs = mne.Epochs(raw_data, events, event_id=event_id, tmin=-0.3, tmax=0.7,
                    preload=True)

### Reject some epochs that have to large magnitude to have nicer ERP plots
reject_criteria = dict(eeg=100e-6)  # 100 µV
epochs.drop_bad(reject=reject_criteria)
epochs.plot_drop_log() # to viz how much dropped epochs

## Compute ERP

### Average using conditions --> now it is not "Epochs" object anymore but "Evoked"
l_frequent = epochs['10'].average()
l_target = epochs['20'].average()

### Plot ERP
fig1 = l_frequent.plot()
fig2 = l_target.plot(spatial_colors=True)

### Plot averaged topomap on distinct timestamps
l_target.plot_topomap(times=[-0.2, 0.1, 0.4], average=0.05)

### Plot Topo + ERP 
l_frequent.plot_joint()

### Comparing conditions on a subset of electrodes
evokeds = dict(frequent=l_frequent, target=l_target)
mne.viz.plot_compare_evokeds(evokeds, picks='AF7')
mne.viz.plot_compare_evokeds(evokeds, picks='AF8')
mne.viz.plot_compare_evokeds(evokeds, picks=['AF7', 'AF8'], combine='mean')

### If you want confidence interval you can also use `iter_evoked` to have each one separately instead of combined
evokeds = dict(target=list(epochs['10'].iter_evoked()),
               frequent=list(epochs['20'].iter_evoked()))
mne.viz.plot_compare_evokeds(evokeds, picks='AF7')
mne.viz.plot_compare_evokeds(evokeds, picks='AF8')
mne.viz.plot_compare_evokeds(evokeds, picks=['AF7', 'AF8'], combine='mean')

# Compare conditions by substraction
freq_minus_target = mne.combine_evoked([l_frequent, l_target], weights=[1, -1])
freq_minus_target.plot_join()

## Compute ERP amplitude and latency

def print_peak_measures(ch, tmin, tmax, lat, amp):
    """Define a function to print out the channel (ch) containing the
    peak latency (lat; in msec) and amplitude (amp, in µV), with the
    time range (tmin and tmax) that was searched.
    This function will be used throughout the remainder of the tutorial."""
    print(f'Channel: {ch}')
    print(f'Time Window: {tmin * 1e3:.3f} - {tmax * 1e3:.3f} ms')
    print(f'Peak Latency: {lat * 1e3:.3f} ms')
    print(f'Peak Amplitude: {amp * 1e6:.3f} µV')

### Use the `get_peak()` method of Evoked object on the best channel
good_tmin, good_tmax = 0.08, 0.12
ch, lat, amp = l_target.get_peak(ch_type='eeg', tmin=good_tmin, tmax=good_tmax,
                              mode='pos', return_amplitude=True)  # ch is the channel with the best peak
print_peak_measures(ch, good_tmin, good_tmax, lat, amp) 

### Use get_peak on a specific channel (can also be used with the mne.combine_evoked)
l_target_AF8 = l_target.copy().pick('AF8')
ch_roi, lat_roi, amp_roi = l_target_AF8.get_peak(
    tmin=good_tmin, tmax=good_tmax, mode='pos', return_amplitude=True)
print_peak_measures(ch_roi, good_tmin, good_tmax, lat_roi, amp_roi)

## Compute ERP amplitude

### Compute amplitude for a condition
channels = ['AF7', 'AF8']
l_target_mean = l_target.copy().pick(channels).crop(
    tmin=good_tmin, tmax=good_tmax)

# Extract mean amplitude in µV over time
mean_amp = l_target_mean.data.mean(axis=1) * 1e6
