# %%
import pickle
import sys

import pandas  as pd
import awkward as ak

from tqdm import tqdm

import matplotlib.pyplot as plt

# %%
def open_pickle_file(file):
    """
    Opens a given pickle file. You need to assign it to a variable.
    """
    with open(file, 'rb') as file:
        return pickle.load(file)
    
# %%
events   = open_pickle_file("data/515_final_event_numbers.pickle")
hit_mpmt = open_pickle_file("data/515_final_hit_mpmt_card_id.pickle")
hit_pmt  = open_pickle_file("data/515_final_hit_pmt_channel.pickle")
# %%
list_like_hit_mpmt_card_ids = []
for i in tqdm(range(len(events)), total=len(events), desc="Processing Files", colour="blue"):

    for j in tqdm(hit_mpmt[i], total=len(hit_mpmt[i]), desc=f"Processing card_ids {i}", leave=False, colour="yellow"):
        list_like_hit_mpmt_card_ids.append(j.to_list())

# hits = ak.flatten(list_like_hit_mpmt_card_ids).to_numpy()       
# %%
list_like_hit_pmt_card_ids = []
for i in tqdm(range(len(events)), total=len(events), desc="Processing Files", colour="blue"):

    for j in tqdm(hit_pmt[i], total=len(hit_pmt[i]), desc=f"Processing card_ids {i}", leave=False, colour="yellow"):
        list_like_hit_pmt_card_ids.append(j.to_list())
# %%
