import torch.nn as nn
import torch.nn.init as init
import torch
from torch.nn.parameter import Parameter
import torch.nn.functional as F
from torch.nn.functional import normalize
class Encoder(nn.Module):
    def __init__(self, input_dim, feature_dim):
        super(Encoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 500),
            nn.ReLU(),
            nn.Linear(500, 500),
            nn.ReLU(),
            nn.Linear(500, 2000),
            nn.ReLU(),
            nn.Linear(2000, feature_dim),
        )

    def forward(self, x):
        return self.encoder(x)


class Decoder(nn.Module):
    def __init__(self, input_dim, feature_dim):
        super(Decoder, self).__init__()
        self.decoder = nn.Sequential(
            nn.Linear(feature_dim, 2000),
            nn.ReLU(),
            nn.Linear(2000, 500),
            nn.ReLU(),
            nn.Linear(500, 500),
            nn.ReLU(),
            nn.Linear(500, input_dim)
        )

    def forward(self, x):
        return self.decoder(x)


class ClusteringLayer(nn.Module):

    def __init__(self, n_clusters, n_z):
        super(ClusteringLayer, self).__init__()
        self.centroids = Parameter(torch.Tensor(n_clusters, n_z)) #大小为10*10

    def forward(self, x):
        q = 1.0 / (1.0 + torch.sum(torch.pow(x.unsqueeze(1) - self.centroids, 2), 2))
        q = (q.t() / torch.sum(q, 1)).t()
        return q



class Network(nn.Module):

    def __init__(self, view, input_size, feature_dim, class_num, device):
        super(Network, self).__init__()
        self.encoders = []
        self.decoders = []
        self.cluster_layer = []
        for v in range(view):
            self.encoders.append(Encoder(input_size[v], feature_dim).to(device))
            self.decoders.append(Decoder(input_size[v], feature_dim).to(device))
            self.cluster_layer.append(ClusteringLayer(class_num, feature_dim).to(device))

        self.encoders = nn.ModuleList(self.encoders)
        self.decoders = nn.ModuleList(self.decoders)
        self.cluster_layer = nn.ModuleList(self.cluster_layer)

        self.view = view

    def forward(self, xs):

        xrs = []
        zs = []
        qs = []
        for v in range(self.view):
            x = xs[v]
            z = self.encoders[v](x)
            xr = self.decoders[v](z)
            q = self.cluster_layer[v](z)

            zs.append(z)
            xrs.append(xr)
            qs.append(q)
        return  xrs, zs, qs

    def forward_plot(self, xs):
        zs = []
        hs = []
        for v in range(self.view):
            x = xs[v]
            z = self.encoders[v](x)
            zs.append(z)
            h = self.feature_contrastive_module(z)
            hs.append(h)
        return zs, hs

    def forward_cluster(self, xs):
        qs = []
        preds = []
        for v in range(self.view):
            x = xs[v]
            z = self.encoders[v](x)
            q = self.label_contrastive_module(z)
            pred = torch.argmax(q, dim=1)
            qs.append(q)
            preds.append(pred)
        return qs, preds


class myNet(nn.Module):
    def __init__(self, class_num, feature_dim):
        super(myNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, feature_dim)


        self.reverse_fc1 = nn.Linear(feature_dim, 50)
        self.reverse_fc2 = nn.Linear(50, 320)
        self.deconv2 = nn.ConvTranspose2d(20, 10, kernel_size=6, stride=6)  # Inverse of conv1
        self.deconv1 = nn.ConvTranspose2d(10, 1, kernel_size=5, stride=5, padding=2)

        self.centroids = Parameter(torch.Tensor(class_num, feature_dim))  # 大小为10*10

    def forward(self, x):
        # Original CNNMnist forward pass
        x1 = F.relu(F.max_pool2d(self.conv1(x), 2)) # torch.Size([128, 10, 12, 12])
        x1 = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x1)), 2))# torch.Size([128, 20, 4, 4])
        x1 = x1.view(-1, x1.shape[1] * x1.shape[2] * x1.shape[3]) # torch.Size([128, 320])
        x1 = F.relu(self.fc1(x1)) # torch.Size([128, 50])
        x1 = F.dropout(x1, training=self.training) # torch.Size([128, 50])
        x1 = self.fc2(x1) # torch.Size([128, 10])
        zs = x1

        # Reverse CNN forward pass
        x2 = self.reverse_fc1(x1) # torch.Size([128, 50])
        x2 = F.dropout(x2, training=self.training)
        x2 = F.relu(self.reverse_fc2(x2)) # 320
        x2 = x2.view(-1, 20, 4, 4)
        x2 = F.relu(F.max_pool2d(self.conv2_drop(self.deconv2(x2)), 2)) # torch.Size([128, 10, 12, 12])
        x2 = F.relu(F.max_pool2d(self.deconv1(x2), 2))  #  torch.Size([128, 1, 28, 28])
        xrs = x2


        init.uniform_(self.centroids)
        q = 1.0 / (1.0 + torch.sum(torch.pow(zs.unsqueeze(1) - self.centroids, 2), 2))
        q = (q.t() / torch.sum(q, 1)).t()
        return zs, xrs, q




class Net_linear(nn.Module):
    def __init__(self, class_num, feature_dim, dim):
        super(Net_linear, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(dim, 500),
            nn.ReLU(),
            nn.Linear(500, 500),
            nn.ReLU(),
            nn.Linear(500, 2000),
            nn.ReLU(),
            nn.Linear(2000, feature_dim),
        )

        self.decoder = nn.Sequential(
            nn.Linear(feature_dim, 2000),
            nn.ReLU(),
            nn.Linear(2000, 500),
            nn.ReLU(),
            nn.Linear(500, 500),
            nn.ReLU(),
            nn.Linear(500, dim)
        )
        self.centroids = Parameter(torch.Tensor(class_num, feature_dim))


        init.xavier_normal_(self.centroids)

    def forward(self, x):
        zs = self.encoder(x)
        xrs = self.decoder(zs)
        q = 1.0 / (1.0 + torch.sum(torch.pow(zs.unsqueeze(1) - self.centroids, 2), 2))
        q = (q.t() / torch.sum(q, 1)).t()
        return zs, xrs, q



class test_linear(nn.Module):
    def __init__(self,view , class_num, feature_dim, high_feature_dim, input_dim, batch_size, temperature_f = 0.5):
        super(test_linear, self).__init__()
        self.batch_size = batch_size
        self.temperature_f = temperature_f
        self.cos_sim = nn.CosineSimilarity(dim=-1)
        self.criterion = nn.CrossEntropyLoss(reduction="sum")
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 500),
            nn.ReLU(),
            nn.Linear(500, 500),
            nn.ReLU(),
            nn.Linear(500, 2000),
            nn.ReLU(),
            nn.Linear(2000, feature_dim),
        )

        self.decoder = nn.Sequential(
            nn.Linear(feature_dim, 2000),
            nn.ReLU(),
            nn.Linear(2000, 500),
            nn.ReLU(),
            nn.Linear(500, 500),
            nn.ReLU(),
            nn.Linear(500, input_dim)
        )

        self.feature_contrastive_module = nn.Sequential(
            nn.Linear(feature_dim, high_feature_dim),
        )

        self.centroids = Parameter(torch.Tensor(class_num, feature_dim))


        init.xavier_normal_(self.centroids)

    def forward(self, x):
        zs = self.encoder(x)
        xrs = self.decoder(zs)
        hs = normalize(self.feature_contrastive_module(zs), dim=1)

        q = 1.0 / (1.0 + torch.sum(torch.pow(zs.unsqueeze(1) - self.centroids, 2), 2))
        q = (q.t() / torch.sum(q, 1)).t()

        return zs, xrs, hs, q


    def mask_correlated_samples(self, N):
        mask = torch.ones((N, N))
        mask = mask.fill_diagonal_(0)
        for i in range(N//2):
            mask[i, N//2 + i] = 0
            mask[N//2 + i, i] = 0
        mask = mask.bool()
        return mask

    def forward_feature(self, hs, pair_hs):
        with torch.no_grad():
            h_j = pair_hs
        h_i = hs
        N = 2 * len(h_i)
        h = torch.cat((h_i, h_j), dim=0)
        sim = torch.matmul(h, h.T) / self.temperature_f
        sim_i_j = torch.diag(sim, len(h_i))
        sim_j_i = torch.diag(sim, -len(h_i))
        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        mask = self.mask_correlated_samples(N)
        negative_samples = sim[mask].reshape(N, -1)
        labels = torch.zeros(N).to(positive_samples.device).long()
        logits = torch.cat((positive_samples, negative_samples), dim=1)
        loss = self.criterion(logits, labels)
        loss /= N
        return loss

    def forward_label(self, anchor, positive, negative, update_lbda):
        if update_lbda > 0:
            pos_similarity = self.cos_sim(anchor, positive)
            neg_similarity = self.cos_sim(anchor, negative)
        else:
            pos_similarity = self.cos_sim(anchor, negative)
            neg_similarity = self.cos_sim(anchor, positive)
        logits = pos_similarity.reshape(-1, 1)
        logits = torch.cat((logits, neg_similarity.reshape(-1, 1)), dim=1)
        logits /= self.temperature_f
        labels = torch.zeros(logits.size(0), dtype=torch.long).to(pos_similarity.device)
        loss = self.criterion(logits, labels)
        loss /= logits.size(0)
        return loss
