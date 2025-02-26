# %%
import pickle
import sys

import pandas            as pd
import numpy             as np
import matplotlib.pyplot as plt
import awkward           as ak

from tqdm                            import tqdm
from matplotlib.backends.backend_pdf import PdfPages

# %%
def open_pickle_file(file):
    """
    Opens a given pickle file. You need to assign it to a variable.
    """
    with open(file, 'rb') as file:
        return pickle.load(file)

# %%
run        = 514
pdf_file = "figures/"+str(run)+"_card100.pdf"

# %%
charges  = open_pickle_file("data/"+str(run)+"_final_hit_pmt_charges.pickle")
card_id  = open_pickle_file("data/"+str(run)+"_final_hit_mpmt_card_id.pickle")
channels = open_pickle_file("data/"+str(run)+"_final_hit_pmt_channel.pickle")

# %%
# The card_id array is related to the charges array as it gives the card number in 
# which the hit is produced.
# card    = card_id[0]
# charge  = charges[0]
# channel = channels[0]

# charge_from_card_100_channel_1 = [
#     chg[(c == 100) & (ch == 1)] for c, ch, chg in tqdm(zip(card, channel, charge), 
#                                                        total=len(card))
# ]

# charge_from_card_100_channel_1 = ak.flatten(charge_from_card_100_channel_1)
# plt.hist(charge_from_card_100_channel_1, bins=100, histtype="step")
# plt.yscale('log')
# plt.xlabel("Charge")

# card = np.array(card, dtype=object)
# channel = np.array(channel, dtype=object)
# charge = np.array(charge, dtype=object)

# %%
pdf = PdfPages(pdf_file)

for pmt in range(19):
    total_charge = []
    for i in tqdm(range(len(card_id)), total=len(card_id), colour="blue"):
        card    = card_id[i]
        charge  = charges[i]
        channel = channels[i]

        charge_from_card_100_channel_1 = [
            chg[(c == 100) & (ch == pmt)] for c, ch, chg in tqdm(zip(card, channel, charge), 
                                                            total=len(card), 
                                                            colour="yellow", 
                                                            leave=False)
        ]

        charge_from_card_100_channel_1 = ak.flatten(charge_from_card_100_channel_1)

        total_charge.append(charge_from_card_100_channel_1)

    total_charge = ak.flatten(total_charge)

    fig, ax = plt.subplots(figsize=(20,12))

    ax.hist(total_charge, bins=10000, histtype="step")
    ax.set_yscale('log')
    ax.set_xlabel(f"Charge - Card 100 - Channel {pmt}")
    # ax.set_xlim(-150, 10000)
    pdf.savefig(fig)

pdf.close()
# %%
