# Some Imports
import uproot 
import matplotlib
import collections
import sys
import site
import glob
import re

import numpy             as np
import awkward           as ak
import matplotlib.pyplot as plt
import pandas            as pd

from tqdm import tqdm

# matplotlib.use('TkAgg')

# Figure Settings
plt.tight_layout()
matplotlib.rcParams['figure.figsize'] = (15, 11)
font = {'size'   : 16}
matplotlib.rc('font', **font)
plt.rcParams["legend.markerscale"] = 3

# Function to extract the number after "P" in the file names
def extract_p_number(file):
    match = re.search(r"P(\d+)", file)
    if match:
        return int(match.group(1))
    return -1

# Select the run number and file
run = 514
run_duration = 1920 # Run duration in seconds
files = glob.glob(f"/eos/experiment/wcte/data/readout_commissioning/offline/dataR{run}S*P*.root")

# Sort the files using extract_p_number
files = sorted(files, key=extract_p_number)
print(f"Run {run} has {len(files)} files")

"""
Window Filtering
First, we're just keeping the "parts" of the run (those files in which the run is split) that have at least 1 000 events (windows).
Second, we're selecting the windows that are separated less or equal tha 20 times the nominal 524 288 ns separation.
"""

tree = uproot.open(files[0]+":WCTEReadoutWindows")

event_numbers = [tree['event_number'].array().to_numpy()]
window_times  = [tree['window_time'].array().to_numpy()]

for f in tqdm(files[1:], desc="Reading all parts"):
    tree2 = uproot.open(f+":WCTEReadoutWindows")
    event_numbers.append(tree2['event_number'].array().to_numpy()+event_numbers[-1].max()+1)
    window_times.append(tree2['window_time'].array().to_numpy()) 

# Plot Unfiltered Data
# for e,t in zip(event_numbers, window_times):
#     plt.scatter(e, t/1e9, marker='.')

# plt.xlabel("Event Number")
# plt.ylabel("Window Times [s]")
# plt.ylim(-50, 3000);
# plt.hlines(run_duration, 0, np.max(event_numbers[-1]));
# plt.show()

# Remove the parts with less than 1 000 events
def remove_small_files(events, threshold=1000):
    valid_indices = []
    for i, e in enumerate(events):
        if len(e) > threshold:
            valid_indices.append(i)

    return valid_indices

valid_indices = remove_small_files(event_numbers)
print(f"We're keeping files {valid_indices}")

selected_files = [files[i] for i in valid_indices]

"""
Now, visualizing the data I've come to realize that the average window separation is something between 1 and 20 times the nominal
separation (~50 ms), so again, we're not selecting any event separated more than 20 x 50 ms from its previous window, and this should
maintain a tendence, this means the algorithm should look for the zone where the points follow a linear tendence, and keep that points,
removing those in the beggining or in the end that start to differ from the tendence. 

It is important to notice that the first part alway start diverging after many events, whilst other files start in non-sensical window 
time values and start to catch the tendence of the points. But there are parts, specially at the end of the run, that show herratic
behaviours and event data after the run has stopped. This makes the algorithm more complicated and makes it fail, so it needs a
revision as we need to ensure we are collecting all quality data, and just leaving behind corrupted or non-sensical data."""

def remove_times(times, thr):
    """
    Checks every window time and select those events that fullfull our requirements
    """

    # Filtered indices list, we start with the first event
    valid_time_indices = [0]  # We mantain the first event always since part 0 always diverge at the end and we reverse the other parts

    # Variables for tendence control
    last_valid_time = times[0]
    
    threshold = thr*524288
    
    for i in range(1, len(times)):
        diff = times[i] - last_valid_time

        # Verify if the difference is in agreement with the threshold setting
        if abs(diff - 524288) <= threshold:
            valid_time_indices.append(i)  # If it is valid, append the index
            last_valid_time = times[i]    # Update the last valid event
        else:
            # If the step is higher than the threshold, discard the value and nothing is updated
            continue
    
    return valid_time_indices

def remove_bad_window_time(files, threshold=20):
    """
    Remove from event_numbers and window_times collections those events and window times that are invalid
    because they don't match the threshold requirements"""
    valid_time_indices = []
    
    for f in files:
        file_part = int(f.split("P")[1].split(".")[0])
        tree  = uproot.open(f+":WCTEReadoutWindows")
        times = tree['window_time'].array().to_numpy()
        
        # This is the crucial part of the algorith.
        # In the first part, the correct tendence appears from the very beggining, so we can run our algorithm just at is is
        # and it will find the invalid events.
        # For the rest of the parts, this is inverted, we start with invalid events and then the correct tendence, this is why
        # we pass the times array reversed.
        # NOTE: As we go up in files, we may find a part that again presents part 0 behaviour, with the correct tendence in the
        # beggining. This is bad, because our algorithm won't be able to correctly filter those windows. This might be solved with
        # future ToolDAQ versions.
        if file_part == 0:
            valid_time_indices.append(remove_times(times, threshold))

        else:
            valid_time_indices.append(remove_times(times[::-1], threshold))
            
    return valid_time_indices

valid_time_indicess = remove_bad_window_time(selected_files)

def filter_and_merge_windows(files, indices):
    """
    Finally, we will re-read the events and time windows from scratch and use valid_time_indicess to select
    those events and times that match our requisites."""
    # Read first part file
    tree0  = uproot.open(files[0]+":WCTEReadoutWindows")
    event0 = tree['event_number'].array().to_numpy()
    time0  = tree['window_time'].array().to_numpy()

    # Creating a set with the valid_indices makes iteration much faster
    valid_time_indices_set = set(indices[0])

    # Select those events that we know that fullfill our requirements
    itt = len(event0)
    event_numbers = [[event0[i] for i in range(itt) if i in valid_time_indices_set]]
    window_times  = [[time0[i] for i in range(itt) if i in valid_time_indices_set]]

    for i, f in enumerate(files[1:]):
        tree2  = uproot.open(f+":WCTEReadoutWindows")
        event2 = tree2['event_number'].array().to_numpy()
        time2  = tree2['window_time'].array().to_numpy()

        # Remember that now we selected the indices in reverse, we need to re-reverse them
        event2_reversed = event2[::-1]
        time2_reversed  = time2[::-1]

        valid_time_indices_set = set(indices[i+1]) # Select the correct indices for this file
        last_event = np.max(event_numbers[-1])+1 # Stack events instead of restart every file

        itt = len(event2)
        event_numbers.append([event2_reversed[j]+last_event for j in range(itt) if j in valid_time_indices_set])
        window_times.append([time2_reversed[j] for j in range(itt) if j in valid_time_indices_set])

    return event_numbers, window_times

event_numbers, window_times = filter_and_merge_windows(selected_files, valid_time_indicess)

for i, (e,t) in enumerate(zip(event_numbers, window_times)):
    time_in_seconds = [i*1e-9 for i in t]
    
    plt.scatter(e, time_in_seconds, marker=".", label=f"Part {i}");
    
plt.xlabel("Event Number")
plt.ylabel("Window Start Time [s]")
plt.hlines(run_duration, 0, np.max(event_numbers[-1]));
plt.text(0, 1145, "Run Duration [s]");

plt.legend(loc="lower right");
plt.show()