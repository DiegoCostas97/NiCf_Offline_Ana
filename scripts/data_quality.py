# %%
import pickle

import awkward           as ak
import matplotlib.pyplot as plt
import numpy             as np

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
def hits_heatmap(events, cards, bins, file):
    pdf = PdfPages(file+".pdf")

    for i in tqdm(range(len(events)), total=len(events), desc="Processing Files", colour="blue"):
        list_like_hit_mpmt_card_ids = []

        for j in tqdm(cards[i], total=len(cards[i]), desc=f"Processing card_ids {i}", leave=True, colour="yellow"):
            list_like_hit_mpmt_card_ids.append(j.to_list())

        zeros = ak.zeros_like(list_like_hit_mpmt_card_ids)
        new_matrix = events[i] + zeros

        current_events   = ak.flatten(new_matrix).to_numpy()
        card_ids = ak.flatten(list_like_hit_mpmt_card_ids).to_numpy()

        if current_events.shape != card_ids.shape:
            raise ValueError("events and card_ids matrices don't have same shape")

        fig, ax = plt.subplots(figsize=(20,12))

        h = ax.hist2d(current_events, card_ids, bins=(1000,np.arange(bins)-0.5), norm='log', cmap='turbo')

        ax.set_xlabel("Event Number")
        ax.set_ylabel("Card Number")

        fig.colorbar(h[3], ax=ax)
        pdf.savefig(fig)

    pdf.close()

# We could do something like creating an array with all "list_like_hit_mpmt_card_ids", then create the 
# zeros matrix and sum it to a array with all final_event_numbers concatenated. This way we can make the 
# heatmap containing all the information fo the run instead of part by part. But this is enough for the
# moment for showing this data is not super good.

# %%
final_event_numbers = open_pickle_file("data/final_event_numbers.pickle")
final_hit_mpmt_card_id = open_pickle_file("data/final_hit_mpmt_card_id.pickle")

hits_heatmap(final_event_numbers, final_hit_mpmt_card_id, 133, "card_ids_heatmap")
# %%
# Some other data quality check could be taking one mPMT and checking that every channel inside
# it is getting the same amount of hits
final_event_numbers = open_pickle_file("data/final_event_numbers.pickle")
final_hit_pmt_channel = open_pickle_file("data/final_hit_pmt_channel.pickle")
# %%
hits_heatmap(final_event_numbers, final_hit_pmt_channel, 19, "channel_heatmap")
# %%
# The data is not very good and the problems seems to be somewhere in the aquisition,
# but we can't do anything, so let's continue with the actual useful variables for
# NiCf analysis.