import os
from tqdm.autonotebook import tqdm
import torch
import torch.nn.functional as F
device = ('cuda' if torch.cuda.is_available() else 'cpu')

from GraphVAE import GraphVAE
from VAE import VAE

class Network(torch.nn.Module):
    def __init__(self, nhidden, adj, user_to_idx, movie_to_idx, dropout, num_layers, gave=True, vae_mode=False):
        super(Network, self).__init__()
        if gvae:
            self.gvae = GraphVAE(
                nhidden, adj, user_to_idx, movie_to_idx, dropout, num_layers, vae_mode=vae_mode)
        else:
            self.gvae = VAE(nhidden, adj, user_to_idx, movie_to_idx, vae_mode=vae_mode)
        self.r_loss = torch.nn.NLLLoss()
        self.p_loss = torch.nn.NLLLoss()
        self.opt = torch.optim.Adam(self.parameters(), lr=1e-3)
        self.lr = torch.optim.lr_scheduler.ExponentialLR(self.opt, 0.9)
        self.vae_mode = vae_mode

    def train_iter(self, usr_idx, mvi_idx, ratings):
        self.opt.zero_grad()
        r_output, klloss = self.gvae.forward(usr_idx, mvi_idx)
        p_output = torch.exp(r_output)
        p_output = torch.log(torch.cat([torch.sum(p_output[:, :2], dim=1).unsqueeze(1), torch.sum(p_output[:, 2:], dim=1).unsqueeze(1)], dim=1))
        loss = self.r_loss(r_output, ratings[0]) + self.p_loss(p_output, ratings[1])
        if self.vae_mode:
            loss += (klloss / r_output.size()[0])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), 1)
        self.opt.step()
        return r_output.detach(), p_output.detach(), loss.item()
    
    def get_validation_accuracy(self, bg):
        self.eval()
        count = 0
        r_acc = 0.0
        p_acc = 0.0
        for x, y in bg.get_validation_data():
            r_output, _ = self.gvae.forward(x[0], x[1])
            p_output = torch.exp(r_output)
            p_output = torch.log(torch.cat([torch.sum(p_output[:, :2], dim=1).unsqueeze(1), torch.sum(p_output[:, 2:], dim=1).unsqueeze(1)], dim=1))
            r_acc += torch.sum(r_output.argmax(dim=1) == y[0]).item() / r_output.size()[0]
            p_acc += torch.sum(p_output.argmax(dim=1) == y[1]).item() / p_output.size()[0]
            count += 1
        self.train()
        return r_acc / count, p_acc / count
    
    def trainer(self, bg, max_epoch, summary_step, lr_epoch=2):
        step = 0
        train_loss = 0.0
        train_r_acc = 0.0
        train_p_acc = 0.0
        bcount = 0
        epoch = 0
        epoch_bar = tqdm(total=max_epoch)
        step_bar = tqdm(total=len(bg.data)//bg.batch_size, leave=False)
        for x, y, epoch_pass in bg:
            r_output, p_output, sloss = self.train_iter(x[0], x[1], y)
            train_loss += sloss
            bcount += 1
            step += 1
            step_bar.update(1)
            train_r_acc += torch.sum(r_output.argmax(dim=1) == y[0]).item() / r_output.size()[0]
            train_p_acc += torch.sum(p_output.argmax(dim=1) == y[1]).item() / p_output.size()[0]
            if step % summary_step == 0:
                print("Step: %d, Loss: %.3f" % (step, sloss))
            if epoch_pass:
                epoch += 1
                if epoch > max_epoch: break
                valid_r_acc, valid_p_acc = self.get_validation_accuracy(bg)
                print(
                    "Epoch: %d, Loss:%.3f, Train Rating Accuracy: %.3f, Train Link-Prediction Accuracy: %.3f, Validation Rating Accuracy: %.3f, Validation Link-Prediction Accuracy: %.3f" % (
                        epoch, train_loss / bcount, train_r_acc / bcount, train_p_acc / bcount,
                        valid_r_acc, valid_p_acc))
                if epoch % lr_epoch == 0:
                    self.lr.step()
                train_loss = 0.0
                bcount = 0.0
                train_r_acc = 0.0
                train_p_acc = 0.0
                step_bar.close()
                step_bar = tqdm(total=len(bg.data)//bg.batch_size, leave=False)
                epoch_bar.update(1)
        step_bar.close()
        epoch_bar.close()
