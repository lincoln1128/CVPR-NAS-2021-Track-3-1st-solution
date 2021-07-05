import os, sys

sys.path.append('EfficientNet-PyTorch-master/')

import torch.nn as nn
import torchvision

from efficientnet_pytorch import EfficientNetv2, EfficientNet
from torch.nn import functional as F

from scipy import interpolate

from resnet import resnet20, resnet32

from torchvision_resnet import resnet18_3w, resnet18, resnet34_3w, resnet34_2w_d

from resnet_valid import ResNet_valid

from resnet_search import  ResNet_search, CHANNELS, LAYERS, RESOLUTION, AUG

from arch_model import arch_model
import time

os.environ["CUDA_VISIBLE_DEVICES"] = '3'

def normalization(data, dim):
    _range = torch.max(data, dim)[0] - torch.min(data, dim)[0]
    _range = _range.unsqueeze(-1).expand(-1, -1, data.size(-1))
    return (data - torch.min(data, dim)[0].unsqueeze(-1).expand(-1, -1, data.size(-1))) / _range

def show_time(seconds):
    # show amount of time as human readable
    if seconds < 60:
        return "{:.2f}s".format(seconds)
    elif seconds < (60 * 60):
        minutes, seconds = div_remainder(seconds, 60)
        return "{}m,{}s".format(minutes, seconds)
    else:
        hours, seconds = div_remainder(seconds, 60 * 60)
        minutes, seconds = div_remainder(seconds, 60)
        return "{}h,{}m,{}s".format(hours, minutes, seconds)

def top_k_accuracy(output, target, top_k):
    if len(output.shape) == 2:
        output = output.reshape(list(output.shape) + [1, 1])
        target = target.reshape(list(target.shape) + [1, 1])
    correct = np.zeros(len(top_k))
    _, pred = output.topk(max(top_k), 1, True, True)
    for i, k in enumerate(top_k):
        target_expand = target.unsqueeze(1).repeat(1, k, 1, 1)
        equal = torch.max(pred[:, :k, :, :].eq(target_expand), 1)[0]
        correct[i] = torch.sum(equal)
    return correct, len(target.view(-1)), equal.cpu().numpy()

def paths2configs(paths, expansion = 1):
    configs = []
    for i,path in enumerate(paths):
        if i < 4:
            configs.append(LAYERS[i][path])
        elif i < 8:
            configs.append(CHANNELS[path] * expansion)
        elif i < 9:
            configs.append(RESOLUTION[path])
        else:
            configs.append(AUG[path])
    return configs

class NAS:
    def __init__(self):
        pass

    # given some input data, return the "best possible" architecture
    def search(self, train_x, train_y, valid_x, valid_y, metadata):
        n_classes = metadata['n_classes']

        # load resnet18 model (definitely not the best possible architecture, but it'll work as an example!)
        # model = torchvision.models.resnet18()
        #
        # model = EfficientNetv2.from_name('efnetv2-b0')
        # model._conv_stem = nn.Conv2d(train_x.shape[1], 24, kernel_size=3, stride=2, bias=False)
        # model._fc = nn.Linear(model._fc.in_features, n_classes, bias=True)
        #
        # model = EfficientNet.from_name('efficientnet-b0')
        # # model._conv_stem = nn.Conv2d(train_x.shape[1], 24, kernel_size=3, stride=2, bias=False)
        # model._conv_stem = nn.Conv2d(train_x.shape[1], 32, kernel_size=3, stride=2, bias=False)
        # model._fc = nn.Linear(model._fc.in_features, n_classes, bias=True)
        #
        # reshape it to this dataset
        # model = torchvision.models.resnet50()
        # model.conv1 = nn.Conv2d(train_x.shape[1], 64, kernel_size=(7, 7), stride=2, padding=3)
        # model.fc = nn.Linear(model.fc.in_features, n_classes, bias=True)
        #
        # model = ResNet_valid()
        # model.conv1 = nn.Conv2d(train_x.shape[1], 64, kernel_size=(7, 7), stride=2, padding=3)
        # model.fc = nn.Linear(model.fc.in_features, n_classes, bias=True)
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device("cpu")
        model = ResNet_search(in_planes=3, num_classes = n_classes).to(device)

        model_arch = arch_model().to(device)

        batch_size = metadata['batch_size']
        # batch_size = 16
        lr = metadata['lr']

        train_pack = list(zip(train_x, train_y))
        valid_pack = list(zip(valid_x, valid_y))

        train_loader = torch.utils.data.DataLoader(train_pack, int(batch_size), shuffle=False)
        valid_loader = torch.utils.data.DataLoader(valid_pack, int(batch_size))
        # test_loader = torch.utils.data.DataLoader(
        # test_x, int(batch_size))

        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=.9, weight_decay=3e-4)
        epochs = 50
        num_sample = 1

        # sum reduction to match tensorflow
        criterion = nn.CrossEntropyLoss()
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

        # best_val_acc = 0
        # best_epoch = 0
        # best_test = None
        #
        # train_results, valid_results, test_results = [], [], []



        train_start = time.time()
        for epoch in range(epochs):
            print("=== EPOCH {} ===".format(epoch))
            model.train()
            # train_start = time.time()
            corrects, divisor, cumulative_loss = 0, 0, 0
            for batch_idx, (data, target) in enumerate(train_loader):
                # pass data ===========================
                # data, target = data.cuda(), target.cuda()
                data, target = data.to(device), target.to(device)
                data_v, target_v = next(iter(valid_loader))
                data_v, target_v = data_v.to(device), target_v.to(device)
                # data_v, target_v = data_v.cuda(), target_v.cuda()

                weights = F.softmax(model_arch.arch_weights, dim=1)
                if batch_idx == 0:
                    print(weights)
                    paths = weights.argmax(dim=1)
                    print(paths2configs(paths))



                optimizer.zero_grad()
                loss = 0


                for i in range(num_sample):
                    paths = torch.multinomial(weights, 1)
                    output = model(data, paths)
                    loss1 = criterion(output, target)
                    loss = loss + loss1

                loss.backward()
                optimizer.step()

                if epoch >= 5:
                    model.eval()
                    rewards = []
                    log_probs = []
                    for i in range(num_sample):
                        paths = torch.multinomial(weights, 1)
                        p = torch.gather(weights, 1, paths).log().sum()
                        output_v = model(data_v, paths)
                        loss1 = criterion(output_v, target_v)
                        rewards.append(-loss1)
                        log_probs.append(p)
                    model_arch.step(rewards, log_probs)
                    model.train()

                cumulative_loss += loss.item() / num_sample
                corr, div, _ = top_k_accuracy(output, target, top_k=[1])
                corrects = corrects + corr
                divisor += div

            average_epoch_t = (time.time() - train_start) / (epoch + 1)

            prog_str = "T Remaining Est: {}".format(
                show_time(average_epoch_t * (epochs - epoch)))

            acc = 100. * corrects / float(divisor)
            print(cumulative_loss/len(train_loader), acc, time.time() - train_start)
            print(prog_str)

            # scheduler.step()




        # results = torch_evaluator(model, data, metadata, n_epochs=64, full_train=True)

        # model = resnet34_2w_d()
        # model.conv1 = nn.Conv2d(train_x.shape[1], 128, kernel_size=(7, 7), stride=2, padding=3)
        # model.fc = nn.Linear(model.fc.in_features, n_classes, bias=True)

        # model = torchvision.models.mobilenet_v2()
        # model.features[0][0] = nn.Conv2d(train_x.shape[1], 32, kernel_size=(3, 3), stride=2, padding=1)
        # model.classifier[1] = nn.Linear(model.classifier[1].in_features, n_classes, bias=True)

        # model = resnet32()
        # model.conv1 = nn.Conv2d(train_x.shape[1], 16, kernel_size=(3, 3), stride=1, padding=1)
        # model.linear = nn.Linear(model.linear.in_features, n_classes, bias=True)
        weights = F.softmax(model_arch.arch_weights, dim=1)

        paths = weights.argmax(dim=1)
        configs = paths2configs(paths, expansion = 8)
        print(weights)
        print(configs)
        model = ResNet_valid(configs = configs, in_planes = 3, num_classes = n_classes)


        # configs = [2, 4, 8, 4, 320, 384, 384, 512, 1]


        # if train_x.shape[1] == 3:
        #     configs = [3, 4, 7, 2, 320, 448, 448, 256, 1]
        # elif train_x.shape[2] == 28:
        #     configs = [4, 8, 2, 3, 320, 512, 448, 384, 2]
        # else:
        #     configs = [4, 2, 6, 2, 384, 512, 448, 384, 2]
        #
        # # configs = [3, 4, 7, 2, 320, 448, 448, 256, 1]
        #
        #
        #
        # train_x = torch.from_numpy(train_x)
        #
        # x1 = normalization(train_x.view(train_x.size(0), train_x.size(1), -1), 2)
        # MEAN = x1.mean(0).mean(-1)
        # STD = x1.std(0).std(-1)
        #
        # if len(MEAN) == 1:
        #     MEAN = MEAN.expand(3)
        #     STD = STD.expand(3)
        # print(MEAN, STD)
        # # DDD
        #
        #
        # model = ResNet_valid(configs=configs, in_planes=3, num_classes=n_classes, MEAN = MEAN, STD = STD)


        return model



# load the exact data loaders that we'll use to load the data
from ingestion_program.nascomp.helpers import *

# load the exact retraining script we'll use to evaluate the found models
from ingestion_program.nascomp.torch_evaluator import *

# if you want to use the real development data, download the public data and set data_dir appropriately
data_dir = 'public_data'

# find all the datasets in the given directory:
dataset_paths = get_dataset_paths(data_dir)
# dataset_paths = [dataset_paths[1], dataset_paths[2], dataset_paths[0]]
dataset_predictions = []
for path in dataset_paths:
    (train_x, train_y), (valid_x, valid_y), (test_x), metadata = load_datasets(path)

    # train_x = F.interpolate(torch.from_numpy(train_x), size=[112, 112], mode='bilinear', align_corners=True).numpy()
    # valid_x = F.interpolate(torch.from_numpy(valid_x), size=[112, 112], mode='bilinear', align_corners=True).numpy()
    # test_x = F.interpolate(torch.from_numpy(test_x), size=[112, 112], mode='bilinear', align_corners=True).numpy()

    print("=== {} {}".format(metadata['name'], "=" * 50))
    print("Train X shape:", train_x.shape)
    print("Train Y shape:", train_y.shape)
    print("Valid X shape:", valid_x.shape)
    print("Valid Y shape:", valid_y.shape)
    print("Test X shape:", test_x.shape)
    print("Metadata:", metadata)

    # initialize our NAS class
    nas = NAS()

    # search for a model
    model = nas.search(train_x, train_y, valid_x, valid_y, metadata)

    # package data for the evaluator
    data = (train_x, train_y), (valid_x, valid_y), test_x

    # retrain the model from scratch
    results = torch_evaluator(model, data, metadata, n_epochs=64, full_train=True)

    # clean up the NAS class
    del nas

    # save our predictions
    dataset_predictions.append(results['test_predictions'])
    print()

# %% md

# Score the Predictions


# %%

overall_score = 0
out = []
for i, path in enumerate(dataset_paths):
    # load the reference values
    ref_y = np.load(os.path.join(path, 'test_y.npy'))

    # load the dataset_metadata for this dataset
    metadata = load_dataset_metadata(path)

    print("=== Scoring {} ===".format(metadata['name']))
    index = metadata['name'][-1]

    # load the model predictions
    pred_y = dataset_predictions[i]

    # compute accuracy
    score = sum(ref_y == pred_y) / float(len(ref_y)) * 100
    print("  Raw score:", score)
    print("  Benchmark:", metadata['benchmark'])

    # adjust score according to benchmark
    point_weighting = 10 / (100 - metadata['benchmark'])
    score -= metadata['benchmark']
    score *= point_weighting
    print("  Adjusted:  ", score)

    # add per-dataset score to overall
    overall_score += score

    # add to scoring stringg
    out.append("Dataset_{}_Score: {:.3f}".format(index, score))
out.append("Overall_Score: {:.3f}".format(overall_score))

# print score
print(out)
