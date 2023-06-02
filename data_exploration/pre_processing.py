import pickle
import pandas


def read_content_of_pickle_file(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)
        print(data[22])


if __name__ == '__main__':
    filepath = '../data/accel/subj_accel_interp.pkl'
    read_content_of_pickle_file(filepath)
