import os
from collections import defaultdict
from tqdm.autonotebook import tqdm
import numpy as np
import scipy.sparse as sp
import pickle

def read_netflix_dataset(dir_path='./netflix/training_set'):
    movies_to_users = defaultdict(list)
    for fname in tqdm(os.listdir(dir_path)):
        with open(os.path.join(dir_path, fname), 'r') as f:
            movie = f.readline().split(':')[0]
            movies_to_users[movie] = [(line.split(',')[0], line.split(',')[1]) for line in f.readlines()]
    return movies_to_users

def get_adj_mat(data_dict, user_to_idx, movie_to_idx):
    data = []
    row = []
    col = []
    for mv in tqdm(data_dict):
        mvidx = movie_to_idx[mv]
        for usr, rating in data_dict[mv]:
            data.append(int(rating))
            row.append(user_to_idx[usr])
            col.append(mvidx)
    adj_mat = sp.csr_matrix((data, (row, col)), shape=(len(user_to_idx), len(movie_to_idx)), dtype=np.uint8)
    return adj_mat

def normalized_adj_mat(adj_mat):
    Du = adj_mat.sum(axis=0)
    Du = 1.0 / np.sqrt(Du)
    Dv = adj_mat.sum(axis=1)
    Dv = 1.0 / np.sqrt(Dv)
    nadj_mat = adj_mat.multiply(Du)
    nadj_mat = nadj_mat.T.multiply(Dv.T).T
    return nadj_mat

def store_data(adj_mat, user_to_idx, movie_to_idx):
    data = {
        'user_idx': user_to_idx,
        'movie_idx': movie_to_idx,
    }

    with open('NF-metadata.pkl', 'wb') as f:
        pickle.dump(data, f)
    sp.save_npz('NF-adj', adj_mat)

def read_full_data(adj_mat='NF-adjmat.h5', metadata='NF-metadata.pkl'):
    with open(metadata, 'rb') as f:
        metadata = pickle.load(f)
        user_to_idx = metadata['user_idx']
        movie_to_idx = metadata['movie_idx']
    adj_mat = sp.load_npz('NF-adj.npz')
    return adj_mat, user_to_idx, movie_to_idx

def read_reduced_data(rdata_fname='NF-Triples.pkl', metadata_fname='NF-reduced-metadata.pkl'):
    with open(metadata_fname, 'rb') as f:
        metadata = pickle.load(f)
        user_to_idx = metadata['user_idx']
        movie_to_idx = metadata['movie_idx']
    with open(rdata_fname, 'rb') as f:
        rdata = pickle.load(f)
    return rdata, user_to_idx, movie_to_idx
