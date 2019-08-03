import utils
from Trainer import Network
from BatchGenerator import BatchGenerator

print("Reading Dataset")
rdata, user_to_idx, movie_to_idx = utils.read_netflix_dataset()

print("Processing Dataset")
# Reduce Dataset size for smaller set training.
NUSER = 70000
NMOVIE = 7000
adj_mat, user_to_idx, movie_to_idx = read_data()
rdata, user_to_idx, movie_to_idx = read_data()
adj_mat = adj_mat[:NUSER, :NMOVIE]
user_to_idx = dict((u, i) for u, i in user_to_idx.items() if i < NUSER)
movie_to_idx = dict((m, i) for m, i in movie_to_idx.items() if i < NMOVIE)
adj_mat = sp.coo_matrix(adj_mat)
rdata = list(zip(adj_mat.row, adj_mat.col, adj_mat.data))
np.random.shuffle(rdata)

# Batch Generator for the data.
bg = BatchGenerator(rdata, 32768)

# Convert adjacency matrix to a sparse matrix as it allows for faster computations.
tr_data = [1 for _ in bg.data]
tr_row = [d[0] for d in bg.data]
tr_col = [d[1] for d in bg.data]
tr_adj_mat = sp.csr_matrix((tr_data, (tr_row, tr_col)), shape=(NUSER, NMOVIE))
tr_adj_mat = normalized_adj_mat(tr_adj_mat).todense()
tr_adj_mat[np.isinf(tr_adj_mat)] = 0
tr_adj_mat = torch.Tensor(tr_adj_mat).to(device)

print("Training Network")
net = Network(128, tr_adj_mat, user_to_idx, movie_to_idx, 0.2, 2, gvae=True, vae_mode=False).to(device)
net.trainer(bg, 20, 20)

# Save the output of the network.
torch.save(net.state_dict(), 'Network-AE-Embed.pt')
