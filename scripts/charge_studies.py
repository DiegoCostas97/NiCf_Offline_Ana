# %%
import pickle
import sys

import pandas            as pd
import numpy             as np
import matplotlib.pyplot as plt

from tqdm import tqdm

# %%
def open_pickle_file(file):
    """
    Opens a given pickle file. You need to assign it to a variable.
    """
    with open(file, 'rb') as file:
        return pickle.load(file)

# %%
run        = 514
run_time   = 32
pmt_num    = 19
mpmt_cards = 133

# %%
charges  = open_pickle_file("data/"+str(run)+"_final_hit_pmt_charges.pickle")
card_id  = open_pickle_file("data/"+str(run)+"_final_hit_mpmt_card_id.pickle")
channels = open_pickle_file("data/"+str(run)+"_final_hit_pmt_channel.pickle")

# %%
# The card_id array is related to the charges array as it gives the card number in 
# which the hit is produced.
cards  = card_id[0]
charge = charges[0]

charges_from_card_100 = []
for i in tqdm(range(len(cards)), total=len(cards), desc="Events", colour="blue"):
    for j in cards[i]:
        if j == 100:
            charges_from_card_100.append(charge[i][j])

# %%
c = [[charge[i][j] for j in cards[i] if j == 100] for i in range(len(cards))]
# %%
plt.hist(charges_from_card_100, bins=100, histtype="step")
plt.yscale('log')
plt.xlabel("Charge")

# %%


# %%
