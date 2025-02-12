# %%
import pickle
import sys

import pandas            as pd
import numpy             as np
import matplotlib.pyplot as plt

# %%
def open_pickle_file(file):
    """
    Opens a given pickle file. You need to assign it to a variable.
    """
    with open(file, 'rb') as file:
        return pickle.load(file)

# %%
def total_events_in_run(ev):
    total_events = 0
    for i in ev:
        total_events += len(i)
    
    return total_events

# %%
def compute_total_hits(ev, hits):
    total_hits = 0
    for i in range(len(ev)):
        for j in hits[i]:
            total_hits += len(j)

    return total_hits

# %%
run        = 515
run_time   = 30
pmt_num    = 19
mpmt_cards = 133

# %%
events   = open_pickle_file("data/"+str(run)+"_final_event_numbers.pickle")
hit_mpmt = open_pickle_file("data/"+str(run)+"_final_hit_mpmt_card_id.pickle")
hit_pmt  = open_pickle_file("data/"+str(run)+"_final_hit_pmt_channel.pickle")

total_events = total_events_in_run(events)
# %%
# Flatten hit_mpmt list
hit_mpmt_nparray = list(map(np.array, hit_mpmt[0]))
hit_mpmt_flatten = np.concatenate(hit_mpmt_nparray).ravel()
# %%
plt.hist(hit_mpmt_flatten, bins=133, histtype='step');
plt.xlabel("mPMT Card ID")
plt.yscale("log")
# %%
# Count hits in every mPMT
unique, counts = np.unique(hit_mpmt_flatten, return_counts=True)
d = dict(zip(unique, counts))

# hits/(PMT x event)
hpmte = d[100]/(pmt_num*len(events[0]))

# hits/(PMT x s)
hpmts = hpmte*total_events/(run_time*60)
print(hpmts)

# If you compute hpmts for run 515 or 514 and calculate rate increase from backgroun 
# run 516, you'll see a rate increase of ~25%. Uncertainty needs to be calculated, 
# that should be the next step and then start with QE calibration (?).
# %%
# Same but computing with all PMTs in all mPMTs (we expect much less rate since a lot
# of mPMTs are dead)
# total_hits = compute_total_hits(events, hit_pmt)
# print(total_hits/(mpmt_cards*pmt_num*run_time*60))
# %%
