# -*- coding:gbk -*-
import torch
from my_network import Network, test_linear
from torch.utils.data import Dataset, DataLoader
import argparse
import random
from sklearn.metrics.pairwise import euclidean_distances
import copy
from dataloader import load_data, DatasetSplit
from sklearn.cluster import KMeans
from scipy.optimize import linear_sum_assignment
import time
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score, accuracy_score
import matplotlib
import numpy as np
matplotlib.use('Agg')
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
plt.rcParams["font.family"] = "Times New Roman"

parser = argparse.ArgumentParser(description='train')
parser.add_argument('--dataset', default='BDGP')
parser.add_argument("--learning_rate", default=0.001)
parser.add_argument("--mse_epochs", default=500)
parser.add_argument("--main_epochs", default=5000)
parser.add_argument("--alpha", default=0.1)
parser.add_argument("--beta", default=0.1)
parser.add_argument("--feature_dim", default=512)
parser.add_argument("--high_feature_dim", default=128)
parser.add_argument("--num_users", default=10)
parser.add_argument("--Dirichlet_alpha", default=99999)
parser.add_argument("--interval_epoch", default=100)
parser.add_argument("--batch_size", default=2500)


args = parser.parse_args()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.cuda.set_device(0)

if args.dataset == "MNIST-USPS":
    seed = 10
    args.lbda = 5.0
if args.dataset == "BDGP":
    seed = 10
    args.lbda = 4.7
if args.dataset == "Fashion":
    seed = 10
    args.lbda = 5.2
if args.dataset == "Caltech-5V":
    seed = 10
    args.lbda = 5.5


def plot_embedding(data, label, title):
    x_min, x_max = np.min(data, 0), np.max(data, 0)
    data = (data - x_min) / (x_max - x_min)
    fig = plt.figure()
    ax = plt.subplot(111)
    # RGB
    color0 = (0.8941176470588236, 0.10196078431372549, 0.10980392156862745, 1.0)
    color1 = (0.21568627450980393, 0.49411764705882355, 0.7215686274509804, 1.0)
    color2 = (0.30196078431372547, 0.6862745098039216, 0.2901960784313726, 1.0)
    color3 = (0.596078431372549, 0.3058823529411765, 0.6392156862745098, 1.0)
    color4 = (1.0, 0.4980392156862745, 0.0, 1.0)
    color5 = (1.0, 1.0, 0.2, 1.0)
    color6 = (0.6509803921568628, 0.33725490196078434, 0.1568627450980392, 1.0)
    color7 = (0.9686274509803922, 0.5058823529411764, 0.7490196078431373, 1.0)
    color8 = (0.6, 0.6, 0.6, 1.0)
    color9 = (0.3, 0.5, 0.4, 0.7)
    colorcenters = (0.1, 0.1, 0.1, 1.0)
    c = [color0, color1, color2, color3, color4, color5, color6, color7, color8, color9]
    for i in range(label.shape[0]):
        if label[i] >= 10:
            color = c[label[i] - 10]
            plt.text(data[i, 0], data[i, 1], 'O', color=color,
                     fontdict={'weight': 'bold', 'size': 9})
        else:
            color = c[label[i]]
            plt.text(data[i, 0], data[i, 1], str(label[i]), color=color,
                     fontdict={'weight': 'bold', 'size': 9})
    # plt.legend()
    plt.xlim(-0.005, 1.02)
    plt.ylim(-0.005, 1.025)
    plt.xticks([])
    plt.yticks([])
    plt.title(title, fontsize=20, fontproperties=FontProperties())
    return fig


def TSNE_PLOT(Z, Y, name="xxx"):
    tsne = TSNE(n_components=2, init='pca', random_state=0)
    F = tsne.fit_transform(Z)  # TSNE features——>2D
    fig1 = plot_embedding(F, Y, name)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.margins(0, 0)
    now_time = time.strftime('%Y%m%d%H%M%S', time.localtime(time.time()))
    plt.subplots_adjust(wspace=0.25)
    plt.tight_layout()
    plt.savefig('./save/{}.png'.format(now_time), dpi=500)


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def local_train(nu, model, pretrain=False):
    model.train()
    local_epochs = args.mse_epochs if pretrain else 1
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    for pre_epoch in range(local_epochs):
        tot_loss = 0.
        c = 0
        hs_list = []
        for batch_idx, (xs, ys) in enumerate(data_loader_list[nu]):
            xs = xs.to(device)
            optimizer.zero_grad()
            _, xrs, hs, qs = model(xs)
            lh = len(hs)
            if pre_epoch == local_epochs - 1:
                hs_list.append(hs)
            mseloss = criterion(xs, xrs)
            if not pretrain:
                con_losses = []
                adv_losses = []
                for vi_tmp in range(1, view + 1):
                    pair_hs = total_hs[(nu + vi_tmp * step) % args.num_users][c:c + lh]
                    con_losses.append(model.forward_feature(hs, pair_hs))
                    _, _, pre_hs, pre_qs = pre_mdls[nu](xs)
                    _, _, glo_hs, glo_qs = agg_mdls[nu](xs)
                    pre_hs = pre_hs.to(device)
                    glo_hs = glo_hs.to(device)
                    hs = hs.to(device)
                    adv_losses.append(model.forward_label(hs, glo_hs, pre_hs, update_lbda))
                con_loss = sum(con_losses)
                adv_loss = sum(adv_losses)
                loss = mseloss + args.alpha * con_loss + trade_off * adv_loss
                c += lh
            else:
                loss = mseloss
            loss.backward()
            optimizer.step()
            tot_loss += loss.item()

    with torch.no_grad():
        all_hs = torch.cat(hs_list, dim=0)
    return all_hs


# regulate value to [1, 10)
def regulate_value(data):
    sign_data = None
    if data < 0:
        sign_data = -1
    elif data > 0:
        sign_data = 1
    elif data == 0:
        return 0

    abs_data = abs(data)
    if abs_data >= 10:
        return regulate_value(sign_data * abs_data / 10)
    elif 0 < abs_data < 1:
        return regulate_value(sign_data * abs_data * 10)
    else:
        return sign_data * abs_data


def valid(valid_model_list, valid_dataset_list, HeterogeneityAware=False):
    local_zs, local_ys, local_hs = [], [], []
    for an in range(args.num_users):
        zs_list, ys_list, hs_list = [], [], []
        for batch_idx, (xs, ys) in enumerate(valid_dataset_list[an]):
            xs = xs.to(device)
            zs, xrs, hs, _ = valid_model_list[an](xs)
            zs_list.append(zs)
            ys_list.append(ys)
            hs_list.append(hs)
        local_zs.append(torch.cat(zs_list, dim=0))
        local_ys.append(torch.cat(ys_list, dim=0))
        local_hs.append(torch.cat(hs_list, dim=0))

    for st in range(step):
        for v1 in range(1, view):
            local_hs[st] = torch.cat((local_hs[st], local_hs[st + v1 * step]), dim=1)

    global_ys = torch.cat(local_ys[0:step], dim=0)
    global_hs = torch.cat(local_hs[0:step], dim=0)

    kmeans = KMeans(n_clusters=class_num, n_init=100)
    kmeans.fit(global_hs.detach().cpu().numpy())
    labels = kmeans.predict(global_hs.detach().cpu().numpy())
    nmi, ari, acc, pur = evaluate(labels, global_ys.detach().cpu().numpy())
    # TSNE_PLOT(global_hs.detach().cpu().numpy(), global_ys.numpy(), 'Dirichlet ($\infty$)')

    accs.append(acc)
    nmis.append(nmi)
    aris.append(ari)

    print('acc', acc)
    print('nmi', nmi)
    print('ari', ari)
    print('pur', pur)
    print('*' * 50)

    if HeterogeneityAware:
        kms = KMeans(n_clusters=class_num, n_init=100)
        kms.fit(global_hs.detach().cpu().numpy())
        WCSS = kms.inertia_
        cluster_centers = kms.cluster_centers_
        inter_cluster_distances = euclidean_distances(cluster_centers)
        AICD = np.mean(inter_cluster_distances)
        TSNE_PLOT(global_hs.detach().cpu().numpy(), global_ys.numpy(), '')
        return WCSS / AICD


def cluster_acc(y_true, y_pred):
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    u = linear_sum_assignment(w.max() - w)
    ind = np.concatenate([u[0].reshape(u[0].shape[0], 1), u[1].reshape([u[0].shape[0], 1])], axis=1)
    return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size


def evaluate(label, pred):
    nmi = normalized_mutual_info_score(label, pred)
    ari = adjusted_rand_score(label, pred)
    acc = cluster_acc(label, pred)
    pur = purity(label, pred)
    return nmi, ari, acc, pur


def match(y_pred, y_true):
    assert y_pred.size == y_true.size
    D = class_num
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):  # 5000
        w[y_pred[i], y_true[i]] += 1
    ind = linear_sum_assignment(w.max() - w)
    return ind[1]


def purity(y_true, y_pred):
    y_voted_labels = np.zeros(y_true.shape)
    labels = np.unique(y_true)
    ordered_labels = np.arange(labels.shape[0])
    for k in range(labels.shape[0]):
        y_true[y_true == labels[k]] = ordered_labels[k]
    labels = np.unique(y_true)
    bins = np.concatenate((labels, [np.max(labels) + 1]), axis=0)
    for cluster in np.unique(y_pred):
        hist, _ = np.histogram(y_true[y_pred == cluster], bins=bins)
        winner = np.argmax(hist)
        y_voted_labels[y_pred == cluster] = winner
    return accuracy_score(y_true, y_voted_labels)


def aggregate_models(model_list, weights=None):
    if weights is None:
        weights = [1.0 / len(model_list)] * len(model_list)
    agg_model = copy.deepcopy(model_list[0])
    agg_state_dict = agg_model.state_dict()
    for key in agg_state_dict:
        agg_state_dict[key].zero_()
    for model, weight in zip(model_list, weights):
        model_state_dict = model.state_dict()
        for key in agg_state_dict:
            agg_state_dict[key] += model_state_dict[key] * weight
    agg_model.load_state_dict(agg_state_dict)
    return agg_model


if __name__ == '__main__':

    T = 5

    for i in range(T):

        setup_seed(i+10)


        dataset, dims, view, data_size, class_num = load_data(args.dataset, args.num_users, args.Dirichlet_alpha)

        data_loader_list = []

        assert args.num_users % view == 0, "param error"

        step = args.num_users // view


        for i in range(1, view + 1):
            for j in range(step):
                data_loader = DataLoader(DatasetSplit(getattr(dataset, 'V' + str(i)), dataset.Y, dataset.user_data[j], dims[i - 1]),batch_size=args.batch_size, shuffle=False)
                data_loader_list.append(copy.deepcopy(data_loader))

        update_lbda = None
        local_models = []
        agg_mdls = []
        pre_mdls = []
        accs, nmis, aris = [], [], []
        total_hs = [torch.tensor(0) for _ in range(args.num_users)]
        print('Start Training')

        for vi in range(view):
            for _ in range(step):
                local_models.append(copy.deepcopy(
                    test_linear(view, class_num, args.feature_dim, args.high_feature_dim, dims[vi], args.batch_size).to(device)))


        for nu in range(args.num_users):
            total_hs[nu] = local_train(nu, local_models[nu], pretrain=True)


        for me in range(args.main_epochs):

            if me % args.interval_epoch == 0:

                if me == 0:
                    # Heterogeneity-Aware (HA)
                    tmp_value = valid(local_models, data_loader_list, HeterogeneityAware=True)
                    update_lbda = (regulate_value(tmp_value) - args.lbda)
                    print('@',update_lbda)
                    trade_off = min(args.beta, abs(update_lbda))


                pre_mdls = copy.deepcopy(local_models)

                agg_mdls = []
                for vi in range(view):
                    for st in range(step):
                        agg_mdls.append(copy.deepcopy(aggregate_models(local_models[vi * step: (vi + 1) * step])))

                local_models = copy.deepcopy(agg_mdls)

                valid(local_models, data_loader_list, HeterogeneityAware=False)

                for nu in range(args.num_users):
                    total_hs[nu] = copy.deepcopy(local_train(nu, local_models[nu], pretrain=False))

            else:
                for nu in range(args.num_users):
                    local_train(nu, local_models[nu], pretrain=False)


        now_time = time.strftime('%Y%m%d%H%M%S', time.localtime(time.time()))
        acc_name = "save/acc/{}_acc.txt".format(now_time)
        with open(acc_name, "a+") as f:
            f.write("Arguments:\n")
            f.write(f"  --dataset: {args.dataset}\n")
            f.write(f"  --learning_rate: {args.learning_rate}\n")
            f.write(f"  --mse_epochs: {args.mse_epochs}\n")
            f.write(f"  --main_epochs: {args.main_epochs}\n")
            f.write(f"  --feature_dim: {args.feature_dim}\n")
            f.write(f"  --high_feature_dim: {args.high_feature_dim}\n")
            f.write(f"  --num_users: {args.num_users}\n")
            f.write(f"  --Dirichlet_alpha: {args.Dirichlet_alpha}\n")
            f.write(f"  --alpha: {args.alpha}\n")
            f.write(f"  --beta: {args.beta}\n")
            f.write(f"  --lbda: {args.lbda}\n")
            f.write(f"  --interval_epoch: {args.interval_epoch}\n")
            f.write(f"  --batch_size: {args.batch_size}\n")
            f.write(f"  --accs: {str(accs)}\n")
            f.write(f"  --nmis: {str(nmis)}\n")
            f.write(f"  --aris: {str(aris)}\n")
            f.write('\r\n********************************************************************************')



