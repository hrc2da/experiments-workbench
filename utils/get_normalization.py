import pickle as pkl
import numpy as np

with open("all_metrics.pkl", "rb") as f:
    metrics = pkl.load(f)
all_pop = np.array([v[0] for v in metrics])
all_pvi = np.array([v[1] for v in metrics])
all_comp = np.array([v[2] for v in metrics])

pop_mean = np.mean(all_pop)
pvi_mean = np.mean(all_pvi)
comp_mean = np.mean(all_comp)

mean_array = np.array([pop_mean, pvi_mean, comp_mean])
std_array = np.array([np.std(all_pop), np.std(all_pvi), np.std(all_comp)])
with open("new_metrics.pkl", "wb") as fout:
    pkl.dump((mean_array, std_array), fout)

