import torch.nn as nn
from collections import OrderedDict
import torch
import argparse
import torch.nn.init as init
import numpy as np
from sklearn.cluster import KMeans
from load_dataset import AutoEncoderDataset
from torch.utils.data import DataLoader
from load_dataset import *
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
#----------------------VERSION: MIXTURE OF SPARSE AUTOENDCODER KL PENALTY AND ENTROPY LOSS--------------------
import random
random.seed(10)

######### NOTE: ARGUMENT ########################
parser = argparse.ArgumentParser(description='Deep Clustering Network')

# Dataset parameters
parser.add_argument('--data_dir', default='./a_dataset/english/',
                    help='dataset directory')
parser.add_argument('--input_dim', type=int, default=384,
                    help='input dimension')
parser.add_argument('--n-classes', type=int, default=2,
                    help='output dimension')

# Training parameters
parser.add_argument('--lr', type=float, default=1e-3,
                    help='learning rate (default: 1e-4)')
parser.add_argument('--wd', type=float, default=1e-4,
                    help='weight decay (default: 5e-4)')
parser.add_argument('--batch-size', type=int, default=16,
                    help='input batch size for training')

# Model parameters
parser.add_argument('--lamda', type=float, default=1,
                    help='coefficient of the reconstruction loss')
parser.add_argument('--beta', type=float, default=0.01, 
                    help=('coefficient of the regularization term on '
                            'clustering'))
parser.add_argument('--hidden-dims', default=[256, 128, 64, 32],
                    help='learning rate (default: 1e-4)')
parser.add_argument('--latent_dim', type=int, default=2,
                    help='latent space dimension')
parser.add_argument('--n-clusters', type=int, default=2,
                    help='number of clusters in the latent space')


# Utility parameters
parser.add_argument('--n-jobs', type=int, default=1,
                    help='number of jobs to run in parallel')
parser.add_argument('--log-interval', type=int, default=20,
                    help=('how many batches to wait before logging the '
                            'training status'))
parser.add_argument("--window_length", type = float, default= 0.2, help="window length")
parser.add_argument("--overlap", type = float, default= 0, help="overlap")
parser.add_argument('--rho', type=float, default=0.2,
                    help='whether use pre-training')
parser.add_argument('--pretrain_epochs', type=int, default=20,
                    help='epochs for pretraining k-autoencoders')
parser.add_argument('--pretrain_epochs_main', type=int, default= 30,
                    help='epochs for pretraining the main autoencoder for the whole dataset')
parser.add_argument('--pretrain', type=bool, default=True,
                    help='whether use pre-training')
parser.add_argument('--main_train_epochs', type=int, default = 20,
                    help='main_train epochs')
parser.add_argument('--sparsity_param', type=float, default=0.2,
                    help='sparsity constract param')
parser.add_argument('--cl_loss_param', type=float, default= 0.1,
                    help='clasification loss param')
parser.add_argument('--collar', type=float, default= 0.0,
                    help='collar param for DER')

args = parser.parse_args()


class AutoEncoder(nn.Module):

    def __init__(self, args):
        super(AutoEncoder, self).__init__()
        self.args = args
        self.input_dim = args.input_dim
        self.output_dim = self.input_dim
        self.hidden_dims = args.hidden_dims
        self.hidden_dims.append(args.latent_dim)
        self.dims_list = (args.hidden_dims +
                          args.hidden_dims[:-1][::-1])  # mirrored structure
        self.n_layers = len(self.dims_list)
        self.latent_dim = args.latent_dim
        self.n_clusters = args.n_clusters
        self.RHO = args.rho

        # Validation check
        assert self.n_layers % 2 > 0
        assert self.dims_list[self.n_layers // 2] == self.latent_dim

        # Encoder Network
        layers = OrderedDict()
        for idx, hidden_dim in enumerate(self.hidden_dims):
            if idx == 0:
                layers.update(
                    {
                        'linear0': nn.Linear(self.input_dim, hidden_dim),
                        # 'linear0': CustomDense(self.input_dim, hidden_dim),
                        # 'activation0': nn.LeakyReLU()
                        # 'activation0': nn.ReLU()
                    }
                )
            else:
                layers.update(
                    {
                        'linear{}'.format(idx): nn.Linear(
                            self.hidden_dims[idx-1], hidden_dim),
                        # 'linear{}'.format(idx): CustomDense(self.hidden_dims[idx-1], hidden_dim),
                        # 'activation{}'.format(idx): nn.LeakyReLU(),
                        # 'activation{}'.format(idx): nn.ELU(),
                        'activation{}'.format(idx): nn.LeakyReLU(),
                        # 'dropout{}'.format(idx): nn.Dropout(0.5), 
                        'bn{}'.format(idx): nn.BatchNorm1d(
                            self.hidden_dims[idx]),
                        

                        # 'bn{}'.format(idx): nn.BatchNorm1d(
                        #     self.hidden_dims[idx])
                    }
                )
        self.encoder = nn.Sequential(layers)

        # Decoder Network
        layers = OrderedDict()
        tmp_hidden_dims = self.hidden_dims[::-1]
        for idx, hidden_dim in enumerate(tmp_hidden_dims):
            if idx == len(tmp_hidden_dims) - 1:
                layers.update(
                    {
                        'linear{}'.format(idx): nn.Linear(
                            hidden_dim, self.output_dim),
                    }
                )
            else:
                layers.update(
                    {
                        'linear{}'.format(idx): nn.Linear(
                            hidden_dim, tmp_hidden_dims[idx+1]),
                        'activation{}'.format(idx): nn.LeakyReLU(),
                        # 'dropout{}'.format(idx): nn.Dropout(0.5),
                        'bn{}'.format(idx): nn.BatchNorm1d(
                            tmp_hidden_dims[idx+1]),
                    }
                )
        self.decoder = nn.Sequential(layers)
        # Apply Xavier weight initialization to all linear layers
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.xavier_normal_(m.weight)
                init.constant_(m.bias, 0)  # Initialize biases to 0

    def __repr__(self):
        repr_str = '[Structure]: {}-'.format(self.input_dim)
        for idx, dim in enumerate(self.dims_list):
            repr_str += '{}-'.format(dim)
        repr_str += str(self.output_dim) + '\n'
        repr_str += '[n_layers]: {}'.format(self.n_layers) + '\n'
        repr_str += '[n_clusters]: {}'.format(self.n_clusters) + '\n'
        repr_str += '[input_dims]: {}'.format(self.input_dim)
        return repr_str

    def __str__(self):
        return self.__repr__()

    def forward(self, X, latent=False):
        output = self.encoder(X)
        if latent:
            return output
        return self.decoder(output)


class VAE(nn.Module):
    def __init__(self, args):
        super(VAE, self).__init__()
        self.args = args
        self.input_dim = args.input_dim
        self.output_dim = self.input_dim
        self.hidden_dims = args.hidden_dims
        self.latent_dim = args.latent_dim
        self.n_clusters = args.n_clusters

        # Encoder Network
        layers = OrderedDict()
        for idx, hidden_dim in enumerate(self.hidden_dims):
            if idx == 0:
                layers.update(
                    {
                        'linear0': nn.Linear(self.input_dim, hidden_dim),
                      
                    }
                )
            else:
                layers.update(
                    {
                        'linear{}'.format(idx): nn.Linear(
                            self.hidden_dims[idx-1], hidden_dim),
                       
                        'activation{}'.format(idx): nn.ReLU(),

                        'bn{}'.format(idx): nn.BatchNorm1d(
                            self.hidden_dims[idx])
                    }
                )
        self.encoder = nn.Sequential(layers)

        # Decoder Network
        layers = OrderedDict()
        tmp_hidden_dims = self.hidden_dims[::-1]
        for idx, hidden_dim in enumerate(tmp_hidden_dims):
            if idx == len(tmp_hidden_dims) - 1:
                layers.update(
                    {
                        'linear{}'.format(idx): nn.Linear(
                            hidden_dim, self.output_dim),
                    }
                )
            else:
                layers.update(
                    {
                        'linear{}'.format(idx): nn.Linear(
                            hidden_dim, tmp_hidden_dims[idx+1]),
                        'activation{}'.format(idx): nn.ReLU(),
                        'bn{}'.format(idx): nn.BatchNorm1d(
                            tmp_hidden_dims[idx+1])
                    }
                )
        self.decoder = nn.Sequential(layers)
        self.fc_mu = nn.Linear(self.hidden_dims[-1], self.latent_dim)
        self.fc_var = nn.Linear(self.hidden_dims[-1], self.latent_dim)

        self.decode_input_linear = nn.Linear(self.latent_dim, self.hidden_dims[-1])
        # Apply Xavier weight initialization to all linear layers
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.xavier_normal_(m.weight)
                init.constant_(m.bias, 0)  # Initialize biases to 0

    def encode(self, x):
        x = self.encoder(x)
        mu = self.fc_mu(x)
        log_var = self.fc_var(x)

        return [mu, log_var]
    
    def decode(self, x):
        x = self.decode_input_linear(x)
        x = self.decoder(x)
        return x

    def reparameterize(self, mu, logvar):
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu
    

    def loss_function(self, x_hat, x, mu, log_var, kld_weight = 1):
        """
        Computes the VAE loss function.
        KL(N(\mu, \sigma), N(0, 1)) = \log \frac{1}{\sigma} + \frac{\sigma^2 + \mu^2}{2} - \frac{1}{2}
        """

        rec_loss = torch.nn.functional.mse_loss(x_hat, x)

        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)

        loss = rec_loss + kld_weight * kld_loss

        return [loss, rec_loss.detach(), -kld_loss.detach()]

    def forward(self, x):
        """
        Forward VAE
        Return: [output, input, mu, var]
        """
        # Encoder
        mu, log_var = self.encode(x) 
        # Sample
        z = self.reparameterize(mu, log_var)
        # Decoder
        output = self.decode(z)

        return [output, x, mu, log_var]
    

class ClusterNet(nn.Module):

    def __init__(self, input_dim, hidden_dims = [128], n_clusters=2):
        """ClusterNet("""
        super(ClusterNet, self).__init__()
        layers = []
        for i in range(len(hidden_dims)):
            if i == 0:
                layers.append(nn.Linear(input_dim, hidden_dims[i]))
                layers.append(nn.LeakyReLU())
                # layers.append(nn.Dropout(0.5))
                # layers.append(nn.BatchNorm1d(hidden_dims[i])),
            else:
                layers.append(nn.Linear(hidden_dims[i-1], hidden_dims[i])),
                layers.append(nn.LeakyReLU())
                # layers.append(nn.Dropout(0.5))
                # layers.append(nn.BatchNorm1d(hidden_dims[i])),
        # Last layer
        layers.append(nn.Sequential(
            nn.Flatten(), 
            nn.Linear(hidden_dims[-1], n_clusters),
            nn.Softmax(dim = 1),
        ))

        self.layers = nn.Sequential(*layers)
        # Apply Xavier weight initialization to all linear layers
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.xavier_normal_(m.weight)
                init.constant_(m.bias, 0)  # Initialize biases to 0
    def forward(self, x):
        """Extract the feature vectors."""
        features = x
        for layer in self.layers:
            features = layer(features)
        return features
    

class MoESparseAutoencodersCL(nn.Module):
    """
    Mixture of Expert DNN-Autoencoder
    """
    def __init__(self, args):
        super(MoESparseAutoencodersCL, self).__init__()
        self.args = args
        self.input_dim = args.input_dim
        self.output_dim = self.input_dim
        self.hidden_dims = args.hidden_dims
        self.latent_dim = args.latent_dim
        self.n_clusters = args.n_clusters
        self.pretrain_epochs = args.pretrain_epochs
        self.pretrain_epochs_main = args.pretrain_epochs_main
        self.main_train_epochs = args.main_train_epochs
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Define main autoencoder at pretraining
        self.main_autoencoder = AutoEncoder(args=args)
        self.RHO = args.rho
        self.BETA = args.sparsity_param
        self.psedo_label = None
        # Clustering algorithm for pre-training
        self.cluster_algo = None
        self.cl_loss_param = args.cl_loss_param

        # Define autoencoder expert in mixture
        self.moe = {}
        for i in range(self.n_clusters):
            self.moe[i] = AutoEncoder(args)
        # Add cluster net (gating network) to moe 
        self.moe['cluster_net'] = ClusterNet(input_dim= self.input_dim, n_clusters=self.n_clusters)
    
    
    def kl_divergence(self, rho, rho_hat):
        rho_hat = torch.mean(F.sigmoid(rho_hat), 1) # sigmoid because we need the probability distributions
        rho = torch.tensor([rho] * len(rho_hat)).to(self.device)
        return torch.sum(rho * torch.log(rho/rho_hat) + (1 - rho) * torch.log((1 - rho)/(1 - rho_hat)))
    
    # define the sparse loss function
    def sparse_loss(self, rho, X, model):
        values = X
        loss = 0
        model_children = list(model.children())
        for i in range(len(model_children)):
            values = model_children[i](values)
            loss += self.kl_divergence(rho, values)
        return loss / X.shape[0]
    


    def batchwise_entropy_loss(self, cluster_outputs):
        """
        Calculate batch wise entropy loss
        """
        X = torch.mean(cluster_outputs, axis = 0)
        return torch.special.entr(X).sum()
        


    def loss_function(self, expert_outputs, cluster_net_outputs, X, psedo_label):
        """
        Compute loss function in a batch
        Loss = L - Beta * Entropy(cluster_net_outputs) 
        L = -log [p_i * exp (-(xhat_i - x_i) ** 2)]
        """
    
        # Create one-hot psedo label 
        # print("Expert output" , expert_outputs)
        encoded_arr = np.zeros((len(psedo_label), self.n_clusters), dtype=float)
        for i in range(len(psedo_label)):
            encoded_arr[i][psedo_label[i]] = 1
        # print("Cluster network output", cluster_net_outputs)
        # print("Encoded arr:", encoded_arr)
        # Cross entropy loss
        entropy_criterion = nn.CrossEntropyLoss()
        entropy_loss = entropy_criterion(cluster_net_outputs, torch.tensor(psedo_label, dtype=torch.long))
        # print("Entropy loss", entropy_loss)


        # MOE reconstruction loss
        loss = 0
        for i in range(self.n_clusters):
            mse = -((expert_outputs[i] - X)**2).mean(axis=1)
            loss += cluster_net_outputs[:, i] * torch.exp(mse)

        moe_loss = -torch.log(loss).sum() 
        # print('MOE loss', moe_loss)
        return  moe_loss - self.cl_loss_param * entropy_loss


    def train_one_autoencoder(self, autoencoder, optimizer, criterion, data_loader, number_of_epochs, sparsity, rho, name='main', verbose=False):
        """
        Training one autoencoder
        """
        print('Training %s ...'%(name))
        for epoch in range(number_of_epochs):

            running_loss = 0.0
            autoencoder.to(self.device)
            autoencoder.train()
            for batch_index, (data) in enumerate(data_loader):
                batch_size = data.size()[0]
                #  Duplicate if batch has one sample (handle one-sample err)
                if batch_size == 1:
                    data = torch.cat([data, data], dim=0)
                    batch_size = 2 

                data = data.to(self.device).view(batch_size, -1)
                # Get output decoder
                rec_X = autoencoder(data)

                if sparsity:
                    # Get latent
                    reg_loss = criterion(data, rec_X)
                    sparse_loss = self.sparse_loss(rho=rho, X = data, model=autoencoder)
                    loss = reg_loss + self.BETA * sparse_loss
                    # if batch_index & 100 == 0:
                    #     print("Reg-loss: {} , Sparse-loss: {}".format(reg_loss, sparse_loss))
                else:
                    loss = criterion(data, rec_X)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                running_loss += loss.data.numpy()
                if batch_index % 200 ==0 and verbose:
                    print('epoch %d loss: %.5f batch: %d' % (epoch, running_loss/((batch_index + 1)), (batch_index + 1)*batch_size))
                if batch_index != 0 and batch_index % 1000 == 0:
                    break
        print('Done training %s'%(name))


    def pretraining(self, dataloader):
        """
        Pretraining step
        1) Train a single main_autoencoder for the entire dataset
        2) Apply k-means for the embedding space after training to get label for cluster net
        3) Training i-th autoencoder using i-th assigned samples by K-means from the entire dataset
        """
        #---------Training main_autoencoder---------------
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.main_autoencoder.parameters(), lr=args.lr, weight_decay = args.wd)

        self.train_one_autoencoder(autoencoder=self.main_autoencoder, optimizer=optimizer,
                                   criterion=criterion, data_loader= dataloader, 
                                   number_of_epochs=self.pretrain_epochs_main, name= "main_autoencoder", 
                                   verbose=True,
                                   sparsity=False,
                                   rho= self.RHO
                                   )

        # ----------K-means clustering --------------------------
        print("------Clustering---------")
        
        # Get latent X
        batch_X = []
        for batch_idx, (data) in enumerate(dataloader):
            batch_size = data.size()[0]
            # Duplicate if batch has one sample
            if batch_size == 1:
                data = torch.cat([data, data], dim=0)
                batch_size = 2 
            data = data.to(self.device).view(batch_size, -1)
            latent_X = self.main_autoencoder(data, latent=True)
            # print("BATCH LATENT X", latent_X)
            batch_X.append(latent_X.detach().cpu().numpy())
        full_latent_X = np.vstack(batch_X)
        
        
        self.cluster_algo = KMeans(n_clusters=self.n_clusters, n_init= self.n_clusters, init="k-means++", random_state=42).fit(full_latent_X)
        self.psedo_label = self.cluster_algo.labels_
        print("Done clustering!")
        print("Original label:", self.psedo_label)

        # ---------Training each autoencoder expert with predefined label from K-means---------------
        for i in range(self.n_clusters):
            # Get full dataset through batch loop
            dataset = []
            for batch_idx, (data) in enumerate(dataloader):
                batch_size = data.size()[0]
                # # Duplicate if batch has one sample
                if batch_size == 1:
                    data = torch.cat([data, data], dim=0)
                    batch_size = 2 
                dataset.append(data.detach().cpu().numpy())
            dataset = np.vstack(dataset)

            # Extract data for specific expert i
            data_expert_i = dataset[self.cluster_algo.labels_ == i]
            data_expert_i = AutoEncoderDataset(data = data_expert_i)
            dataset_expert_i = DataLoader(data_expert_i, batch_size = args.batch_size, shuffle = False)
            optimizer = torch.optim.Adam(self.moe[i].parameters(), lr=args.lr, weight_decay = args.wd)
            criterion = nn.MSELoss()
            # Train expert_i
            self.train_one_autoencoder(autoencoder=self.moe[i], optimizer=optimizer,
                                       criterion=criterion, data_loader=dataset_expert_i, 
                                       number_of_epochs=self.pretrain_epochs, name="Expert {}".format(i), 
                                       verbose=True, sparsity=True, rho = self.RHO)

        print("Done Pretraining step !")

        return self.moe, full_latent_X
    

    def get_expert_outputs(self, X, latent = False):
        """
        Get output of experts in a batch
        Return: List of output of each expert
        """
        output = []
        for i in range(self.n_clusters):
            if latent:
                output_expert_i = self.moe[i](X, latent = True)
            else:
                output_expert_i = self.moe[i](X)
            output.append(output_expert_i)
        return output
    

    def main_training(self, dataloader, name = "MOE", verbose = True):
        """
        Main training to optimize loss function L = -log [p_i * exp (-(xhat_i - x_i) ** 2)]
        """
        print('Training %s ...'%(name))

        # Add parameters 
        params = list(self.moe['cluster_net'].parameters())
        for i in range(self.n_clusters):
            params += list(self.moe[i].parameters())
            self.moe[i].to(self.device)        
            self.moe[i].train()
            
        optimizer = torch.optim.Adam(params, lr=args.lr, weight_decay = args.wd)

        for epoch in range(self.main_train_epochs):
            running_loss = 0.0
            self.moe['cluster_net'].train()
            
            for batch_index, (data) in enumerate(dataloader):
                batch_size = data.size()[0]
                # Duplicate if batch has one sample (the last batch)
                if batch_size == 1:
                    data = torch.cat([data, data], dim=0)
                    batch_size = 2 
                # Get psedo-label
                psedo_label = self.psedo_label[batch_index: batch_index + batch_size]

                # Get decoder output
                expert_outputs = self.get_expert_outputs(data)
                # Get latent output 
                # latent_outputs = self.get_expert_outputs(data, latent=True)
                
                # # Concate k-latent outputs
                # latent_tensor = latent_outputs[0]
                # for i in range(1, len(latent_outputs)):
                #     latent_tensor = torch.hstack((latent_tensor, latent_outputs[i]))
                # # if batch_index % 100 == 0:
                # #     # print("Latent tensor", latent_tensor)

                clustering_net_outputs = self.moe['cluster_net'](data)

                # print("Cluster net output", clustering_net_outputs)
                loss = self.loss_function(expert_outputs=expert_outputs, 
                                          cluster_net_outputs=clustering_net_outputs, X = data,
                                          psedo_label=psedo_label)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                running_loss += loss.data.numpy()
                if batch_index % 100 ==0 and verbose:
                    print('epoch %d loss: %.5f batch: %d' % (epoch, running_loss/((batch_index + 1)), (batch_index + 1)*batch_size))
                if batch_index != 0 and batch_index % 1000 == 0:
                    break
            
            # Update psedolabel
            if epoch != 0  and epoch % 10 == 0:
                self.psedo_label = self.get_final_cluster(dataloader)
                print("Updated pseudo label!")
                print("######################################")        
        print("Done main training!")
        return self.moe


    def get_final_cluster(self, test_loader):
        """
        Assign final cluster for clustering based on cluster_net
        """
        # Convert to eval mode
        for i in range(self.n_clusters):
            self.moe[i].eval()
        self.moe['cluster_net'].eval()
        total_pred = []
        for batch_idx, (data) in enumerate(test_loader):
            batch_size = data.size()[0]
            data = data.view(batch_size, -1).to(self.device)
            # Get the hard assignment label
            with torch.no_grad():
                # # Get latent output 
                # latent_outputs = self.get_expert_outputs(data, latent=True)

                # # Concate k-latent outputs
                # latent_tensor = latent_outputs[0]
                # for i in range(1, len(latent_outputs)):
                #     latent_tensor = torch.hstack((latent_tensor, latent_outputs[i]))
                cluster_pred = self.moe['cluster_net'](data)
                cluster_pred = cluster_pred.cpu().numpy()
                batch_pred = np.argmax(cluster_pred, axis = 1)
                total_pred.append(batch_pred)
        total_pred = np.concatenate(total_pred, axis=0)
        return total_pred

    
    

    