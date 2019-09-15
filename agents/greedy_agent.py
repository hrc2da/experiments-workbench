from agents import Agent

class GreedyAgent(Agent):
    def __init__(self):
        pass
    def run(self,environment,n_steps):
        print("Running the Greedy Agent in {} for {} steps.".format(environment,n_steps))