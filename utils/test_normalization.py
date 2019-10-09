import pickle


with open('stripped_normalization.pkl', 'rb') as f:
    data = pickle.load(f)
    print(data)
