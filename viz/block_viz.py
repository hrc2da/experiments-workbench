import sys
import os
import numpy as np
import pickle as pkl
from matplotlib import pyplot as plt
import json

if (len(sys.argv) != 2):
    print("USAGE: python distopia_analyze_logs.py <data path>")
    exit(0)
plt.clf()
plt.xlim(100,900)
plt.ylim(100,900)

with open(sys.argv[1]) as jfile:
    j_file = json.load(jfile)
    for x in j_file[0]["episodes"]:
        if x["episode_no"] == 21:
            for y in x["run_log"]:

                for keys, val in y["design"].items():
                    coordx, coordy= val[0]
                    final_coords = [[int(coordx)], [int(coordy)]]
                    plt.plot([int(coordx)], [int(coordy)], 'ro', alpha = 0.01)
                    print(final_coords)
            plt.savefig(os.path.join(os.getcwd(),"{}.png".format("agent_sarsa_episodes_100_episodelen_1000_subsample_50_alpha_0.9_disc_0.2_optimistic_[-1.0.0.]test")))
            exit(0)
