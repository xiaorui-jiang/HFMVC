from sklearn.preprocessing import MinMaxScaler
import numpy as np
from torch.utils.data import Dataset
import scipy.io
import torch
import random
import math

class BDGP(Dataset):
    def __init__(self, path, num_user, Dirichlet_alpha):
        self.num_user = num_user
        self.Dirichlet_alpha = Dirichlet_alpha
        data1 = scipy.io.loadmat(path+'BDGP.mat')['X1'].astype(np.float32)
        data2 = scipy.io.loadmat(path+'BDGP.mat')['X2'].astype(np.float32)
        labels = scipy.io.loadmat(path+'BDGP.mat')['Y'].astype(np.int32).reshape(2500,)
        self.V1 = data1
        self.V2 = data2
        self.Y = labels
        self.user_data = self.split_data()

    def __len__(self):
        return self.V1.shape[0]

    def split_data(self):
        dict_users = {i: np.array([]) for i in range(self.num_user // 2 )}
        N = len(self.Y)
        n_classes = max(self.Y) + 1

        min_size = 0
        min_require_size = 10

        while min_size < min_require_size:
            idx_batch = [[] for _ in range(self.num_user // 2 )]
            for k in range(n_classes):
                idx_k = np.where(self.Y == k)[0]
                np.random.shuffle(idx_k)
                proportions = np.random.dirichlet(np.repeat(self.Dirichlet_alpha, self.num_user // 2))
                proportions = np.array([p * (len(idx_j) < (N / self.num_user * 2)) for p, idx_j in zip(proportions, idx_batch)])
                proportions = proportions / proportions.sum()
                proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
                idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
                min_size = min([len(idx_j) for idx_j in idx_batch])
            for j in range(self.num_user // 2):
                np.random.shuffle(idx_batch[j])
                dict_users[j] = idx_batch[j]
        return  dict_users




class MNIST_USPS(Dataset):
    def \
            __init__(self, path, num_user, Dirichlet_alpha):
        self.num_user = num_user
        self.Dirichlet_alpha = Dirichlet_alpha
        self.Y = scipy.io.loadmat(path + 'MNIST_USPS.mat')['Y'].astype(np.int32).reshape(5000,)
        self.V1 = scipy.io.loadmat(path + 'MNIST_USPS.mat')['X1'].astype(np.float32)
        self.V2 = scipy.io.loadmat(path + 'MNIST_USPS.mat')['X2'].astype(np.float32)
        self.user_data = self.split_data()
    def __len__(self):
        return 5000

    def split_data(self):
        dict_users = {i: np.array([]) for i in range(self.num_user // 2 )}
        N = len(self.Y)
        n_classes = max(self.Y) + 1

        min_size = 0
        min_require_size = 10

        while min_size < min_require_size:
            idx_batch = [[] for _ in range(self.num_user // 2 )]
            for k in range(n_classes):
                idx_k = np.where(self.Y == k)[0]
                np.random.shuffle(idx_k)
                proportions = np.random.dirichlet(np.repeat(self.Dirichlet_alpha, self.num_user // 2))
                proportions = np.array([p * (len(idx_j) < (N / self.num_user * 2)) for p, idx_j in zip(proportions, idx_batch)])
                proportions = proportions / proportions.sum()
                proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
                idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
                min_size = min([len(idx_j) for idx_j in idx_batch])
            for j in range(self.num_user // 2):
                np.random.shuffle(idx_batch[j])
                dict_users[j] = idx_batch[j]
        return  dict_users





class Fashion(Dataset):
    def __init__(self, path, num_user, Dirichlet_alpha):
        self.Y = scipy.io.loadmat(path + 'Fashion.mat')['Y'].astype(np.int32).reshape(10000,)
        self.V1 = scipy.io.loadmat(path + 'Fashion.mat')['X1'].astype(np.float32)
        self.V2 = scipy.io.loadmat(path + 'Fashion.mat')['X2'].astype(np.float32)
        self.V3 = scipy.io.loadmat(path + 'Fashion.mat')['X3'].astype(np.float32)
        self.num_user = num_user
        self.view = 3
        self.Dirichlet_alpha = Dirichlet_alpha
        self.user_data = self.split_data()
    def __len__(self):
        return 10000

    def split_data(self):
        dict_users = {i: np.array([]) for i in range(self.num_user // self.view)}
        N = len(self.Y)
        n_classes = max(self.Y) + 1

        min_size = 0
        min_require_size = 10

        while min_size < min_require_size:
            idx_batch = [[] for _ in range(self.num_user // self.view)]
            for k in range(n_classes):
                idx_k = np.where(self.Y == k)[0]
                np.random.shuffle(idx_k)
                proportions = np.random.dirichlet(np.repeat(self.Dirichlet_alpha, self.num_user // self.view))
                proportions = np.array(
                    [p * (len(idx_j) < (N / self.num_user * self.view)) for p, idx_j in zip(proportions, idx_batch)])
                proportions = proportions / proportions.sum()
                proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
                idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
                min_size = min([len(idx_j) for idx_j in idx_batch])
            for j in range(self.num_user // self.view):
                np.random.shuffle(idx_batch[j])
                dict_users[j] = idx_batch[j]
        return dict_users

class Caltech_2(Dataset):
    def __init__(self, path,num_user, Dirichlet_alpha):
        data = scipy.io.loadmat(path)
        scaler = MinMaxScaler()
        self.V1 = scaler.fit_transform(data['X1'].astype(np.float32))
        self.V2 = scaler.fit_transform(data['X2'].astype(np.float32))
        self.V3 = scaler.fit_transform(data['X3'].astype(np.float32))
        self.V4 = scaler.fit_transform(data['X4'].astype(np.float32))
        self.V5 = scaler.fit_transform(data['X5'].astype(np.float32))
        self.Y = scipy.io.loadmat(path)['Y'].transpose().astype(np.int32).reshape(1400,)
        self.num_user = num_user
        self.view = 2
        self.Dirichlet_alpha = Dirichlet_alpha
        self.user_data = self.split_data()

    def __len__(self):
        return 1400

    def split_data(self):
        dict_users = {i: np.array([]) for i in range(self.num_user // self.view)}
        N = len(self.Y)
        n_classes = max(self.Y) + 1

        min_size = 0  # 所有客户端中，[所分到最少样本的客户端]  的样本数
        min_require_size = 10  # 每个客户端最少需要分到这么多样本

        while min_size < min_require_size:
            idx_batch = [[] for _ in range(self.num_user // self.view)]
            for k in range(n_classes):  # 10类
                idx_k = np.where(self.Y == k)[0]
                np.random.shuffle(idx_k)
                proportions = np.random.dirichlet(np.repeat(self.Dirichlet_alpha, self.num_user // self.view))
                proportions = np.array(
                    [p * (len(idx_j) < (N / self.num_user * self.view)) for p, idx_j in zip(proportions, idx_batch)])
                proportions = proportions / proportions.sum()
                proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
                idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
                min_size = min([len(idx_j) for idx_j in idx_batch])
            for j in range(self.num_user // self.view):
                np.random.shuffle(idx_batch[j])
                dict_users[j] = idx_batch[j]

        return dict_users


class Caltech_3(Dataset):
    def __init__(self, path,num_user, Dirichlet_alpha):
        data = scipy.io.loadmat(path)
        scaler = MinMaxScaler()
        self.V1 = scaler.fit_transform(data['X1'].astype(np.float32))
        self.V2 = scaler.fit_transform(data['X2'].astype(np.float32))
        self.V3 = scaler.fit_transform(data['X5'].astype(np.float32))

        self.Y = scipy.io.loadmat(path)['Y'].transpose().astype(np.int32).reshape(1400,)
        self.num_user = num_user
        self.view = 3
        self.Dirichlet_alpha = Dirichlet_alpha
        self.user_data = self.split_data()

    def __len__(self):
        return 1400

    def split_data(self):
        dict_users = {i: np.array([]) for i in range(self.num_user // self.view)}
        N = len(self.Y)
        n_classes = max(self.Y) + 1

        min_size = 0
        min_require_size = 10

        while min_size < min_require_size:
            idx_batch = [[] for _ in range(self.num_user // self.view)]
            for k in range(n_classes):
                idx_k = np.where(self.Y == k)[0]
                np.random.shuffle(idx_k)
                proportions = np.random.dirichlet(np.repeat(self.Dirichlet_alpha, self.num_user // self.view))
                proportions = np.array(
                    [p * (len(idx_j) < (N / self.num_user * self.view)) for p, idx_j in zip(proportions, idx_batch)])
                proportions = proportions / proportions.sum()
                proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
                idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
                min_size = min([len(idx_j) for idx_j in idx_batch])
            for j in range(self.num_user // self.view):
                np.random.shuffle(idx_batch[j])
                dict_users[j] = idx_batch[j]
        return dict_users



class Caltech_4(Dataset):
    def __init__(self, path,num_user, Dirichlet_alpha):
        data = scipy.io.loadmat(path)
        scaler = MinMaxScaler()
        self.V1 = scaler.fit_transform(data['X1'].astype(np.float32))
        self.V2 = scaler.fit_transform(data['X2'].astype(np.float32))
        self.V3 = scaler.fit_transform(data['X5'].astype(np.float32))
        self.V4 = scaler.fit_transform(data['X4'].astype(np.float32))
        self.Y = scipy.io.loadmat(path)['Y'].transpose().astype(np.int32).reshape(1400,)
        self.num_user = num_user
        self.view = 4
        self.Dirichlet_alpha = Dirichlet_alpha
        self.user_data = self.split_data()

    def __len__(self):
        return 1400


    def split_data(self):
        dict_users = {i: np.array([]) for i in range(self.num_user // self.view)}
        N = len(self.Y)
        n_classes = max(self.Y) + 1

        min_size = 0
        min_require_size = 10

        while min_size < min_require_size:
            idx_batch = [[] for _ in range(self.num_user // self.view)]
            for k in range(n_classes):  # 10类
                idx_k = np.where(self.Y == k)[0]
                np.random.shuffle(idx_k)
                proportions = np.random.dirichlet(np.repeat(self.Dirichlet_alpha, self.num_user // self.view))
                proportions = np.array(
                    [p * (len(idx_j) < (N / self.num_user * self.view)) for p, idx_j in zip(proportions, idx_batch)])
                proportions = proportions / proportions.sum()
                proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
                idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
                min_size = min([len(idx_j) for idx_j in idx_batch])
            for j in range(self.num_user // self.view):
                np.random.shuffle(idx_batch[j])
                dict_users[j] = idx_batch[j]
        return dict_users


class Caltech_5(Dataset):
    def __init__(self, path,num_user, Dirichlet_alpha):
        data = scipy.io.loadmat(path)
        scaler = MinMaxScaler()
        self.V1 = scaler.fit_transform(data['X1'].astype(np.float32))
        self.V2 = scaler.fit_transform(data['X2'].astype(np.float32))
        self.V3 = scaler.fit_transform(data['X5'].astype(np.float32))
        self.V4 = scaler.fit_transform(data['X4'].astype(np.float32))
        self.V5 = scaler.fit_transform(data['X3'].astype(np.float32))
        self.Y = scipy.io.loadmat(path)['Y'].transpose().astype(np.int32).reshape(1400,)
        self.num_user = num_user
        self.view = 5
        self.Dirichlet_alpha = Dirichlet_alpha
        self.user_data = self.split_data()

    def __len__(self):
        return 1400

    def split_data(self):
        dict_users = {i: np.array([]) for i in range(self.num_user // self.view)}
        N = len(self.Y)
        n_classes = max(self.Y) + 1

        min_size = 0
        min_require_size = 10

        while min_size < min_require_size:
            idx_batch = [[] for _ in range(self.num_user // self.view)]
            for k in range(n_classes):  # 10类
                idx_k = np.where(self.Y == k)[0]
                np.random.shuffle(idx_k)
                proportions = np.random.dirichlet(np.repeat(self.Dirichlet_alpha, self.num_user // self.view))
                proportions = np.array(
                    [p * (len(idx_j) < (N / self.num_user * self.view)) for p, idx_j in zip(proportions, idx_batch)])
                proportions = proportions / proportions.sum()
                proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
                idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
                min_size = min([len(idx_j) for idx_j in idx_batch])
            for j in range(self.num_user // self.view):
                np.random.shuffle(idx_batch[j])
                dict_users[j] = idx_batch[j]

        return dict_users



class DatasetSplit(Dataset):

  def __init__(self, dataset_x, dataset_y, idxs, dim):
    self.dataset_x = dataset_x[idxs]
    self.dataset_y = dataset_y[idxs]
    self.idxs = [int(i) for i in idxs]
    self.dim = dim

  def __len__(self):
    return len(self.idxs)

  def __getitem__(self, item):

    image, label = self.dataset_x[item], self.dataset_y[item]
    image = image.reshape(self.dim)
    return torch.tensor(image), torch.tensor(label)



def load_data(dataset,num_user, Dirichlet_alpha):
    if dataset == "BDGP":
        dataset = BDGP('./data/',num_user, Dirichlet_alpha)
        dims = [1750, 79]
        view = 2
        data_size = 2500
        class_num = 5
    elif dataset == "MNIST-USPS":
        dataset = MNIST_USPS('./data/',num_user, Dirichlet_alpha)
        dims = [784, 784]
        view = 2
        data_size = 5000
        class_num = 10

    elif dataset == "Fashion":
        dataset = Fashion('./data/',num_user, Dirichlet_alpha)
        dims = [784, 784, 784]
        view = 3
        data_size = 10000
        class_num = 10
    elif dataset == "Caltech-2V":
        dataset = Caltech_2('data/Caltech-5V.mat',num_user, Dirichlet_alpha)
        dims = [40, 254]
        view = 2
        data_size = 1400
        class_num = 7
    elif dataset == "Caltech-3V":
        dataset = Caltech_3('data/Caltech-5V.mat',num_user, Dirichlet_alpha)
        dims = [40, 254, 928]
        view = 3
        data_size = 1400
        class_num = 7
    elif dataset == "Caltech-4V":
        dataset = Caltech_4('data/Caltech-5V.mat', num_user, Dirichlet_alpha)
        dims = [40, 254, 928, 512]
        view = 4
        data_size = 1400
        class_num = 7
    elif dataset == "Caltech-5V":
        dataset = Caltech_5('data/Caltech-5V.mat', num_user, Dirichlet_alpha)
        dims = [40, 254, 928, 512, 1984]
        view = 5
        data_size = 1400
        class_num = 7

    else:
        raise NotImplementedError
    return dataset, dims, view, data_size, class_num