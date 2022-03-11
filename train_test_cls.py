'''Train CIFAR10 with PyTorch.'''
import torch.backends.cudnn as cudnn
import argparse
from tqdm import tqdm
import torch
import csv
import os
import torchvision
import numpy as np
from torch.utils import data
import torchvision.transforms.functional as tf
from my_dataset.my_datasets_ori import MyValDataSet_cls,MyDataSet_cls
from sklearn import metrics
import random
import models.my_H_Net_model.H_Net_Efficient as CLS_Efficient_b3_DAC_CAC_input_CAM
import models.Incorp_2017.Model as Model
import torch.nn as nn
import utils as utils
from tensorboardX import SummaryWriter
from torch.cuda.amp import autocast, GradScaler
import warnings

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--imgs_train_path', '-it', type=str,
                    default='/mnt/ai2020/orton/dataset/hua_lu_fu_sai/test20210811/test_image/',
                    help='imgs train data path.')
parser.add_argument('--resize', default=224, type=int, help='resize shape')
parser.add_argument('--batch_size', default=1, type=int, help='batchsize')
parser.add_argument('--device', default='cuda', type=str, help='use cuda')
parser.add_argument('--start_epoch', '-s', default=0, type=int, help='start epoch')
parser.add_argument('--end_epoch', '-e', default=300, type=int, help='end epoch')
parser.add_argument('--devicenum', default='2', type=str, help='use devicenum')
args = parser.parse_args()

RANDOM_SEED = 6666

np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed_all(RANDOM_SEED)
random.seed(RANDOM_SEED)

checkpoint_dir = "../../checkpoint/my_cls_checkpoint/skin_my_cls/"
# the learning rate para
device = args.device # 是否使用cuda
os.environ["CUDA_VISIBLE_DEVICES"] = '2'
warnings.filterwarnings("ignore")

batch_size = 16
lr_decay = 2
stage = 0
start_epoch = 0
stage_epochs = [50, 50, 50, 50]
total_epochs = sum(stage_epochs)
writer_dir = os.path.join(checkpoint_dir, "skin_my_cls_DAC_CAC_Input_CAT_CAM")
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)

if not os.path.exists(writer_dir):
    os.makedirs(writer_dir)

writer = SummaryWriter(writer_dir)
result_path = '/mnt/ai2019/zxg_FZU/dataset/Nn-Net_result/cls/skin/'
if not os.path.isdir(result_path):
    os.mkdir(result_path)

# device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
print("using {} device.".format(device))
torch.backends.cudnn.enabled = True

net = CLS_Efficient_b3_DAC_CAC_input_CAM.CLS_Efficient_b3_DAC_CAC_input_cat_CAM(4,3)

net = net.to(device)
criterion = nn.CrossEntropyLoss().to('cuda')
criterion = criterion.to(device)

# Data
print("param size = %fMB", utils.count_parameters_in_MB(net))
# training dataset
############# Load training and validation data
data_train_root = '/mnt/ai2019/zxg_FZU/dataset/skin_bing_zao/Classify/train_resize_512_augu/'
data_train_root_mask = '/mnt/ai2019/zxg_FZU/dataset/skin_bing_zao/Classify/Nn_Net/train/'
data_train_list = '/mnt/ai2019/zxg_FZU/dataset/skin_bing_zao/train_img_labels_aug_cls.txt'
trainloader = data.DataLoader(
    MyDataSet_cls(data_train_root, data_train_root_mask, data_train_list),
    batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=False, drop_last=True)

data_val_root = '/mnt/ai2019/zxg_FZU/dataset/skin_bing_zao/Classify/val_resize_512_augu/'
data_val_root_mask = '/mnt/ai2019/zxg_FZU/dataset/skin_bing_zao/Classify/Nn_Net/val/'
data_val_list = '/mnt/ai2019/zxg_FZU/dataset/skin_bing_zao/val_img_labels_aug_cls.txt'
valloader = data.DataLoader(MyValDataSet_cls(data_val_root, data_val_root_mask, data_val_list, crop_size=224),
                            batch_size=batch_size,
                            shuffle=True,
                            num_workers=4,
                            pin_memory=False)

############# Load testing data
data_train_root = '/mnt/ai2020/orton/dataset/skin_bing_zao/test_resize_512/Images/'
data_train_root_mask = '/mnt/ai2019/zxg_FZU/dataset/skin_bing_zao/Classify/Nn_Net/test/'
data_train_list = '/mnt/ai2019/zxg_FZU/dataset/skin_bing_zao/test_img_labels_aug_cls.txt'
testloader = data.DataLoader(MyValDataSet_cls(data_train_root, data_train_root_mask, data_train_list,crop_size=224), batch_size=1,
                             shuffle=False,
                             num_workers=8,
                             pin_memory=True)

# Model
print('==> Building model..')

def cla_evaluate(label, binary_score, pro_score):
    acc = metrics.accuracy_score(label, binary_score)
    AP = metrics.average_precision_score(label, pro_score)
    auc = metrics.roc_auc_score(label, pro_score)
    f1_score = metrics.f1_score(label, binary_score, average='macro')
    precision = metrics.precision_score(label, binary_score)
    recall = metrics.recall_score(label, binary_score, average='macro')
    jaccard = metrics.jaccard_score(label, binary_score, average='macro')
    CM = metrics.confusion_matrix(label, binary_score)
    sens = float(CM[1, 1]) / float(CM[1, 1] + CM[1, 0])
    spec = float(CM[0, 0]) / float(CM[0, 0] + CM[0, 1])
    return acc, AP, auc, f1_score, precision, recall, jaccard, sens, spec

RANDOM_SEED = 6666
def train_main():
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    torch.cuda.manual_seed_all(RANDOM_SEED)
    random.seed(RANDOM_SEED)
    stage = 0

    def train_val(epoch,net, trainloader,valloader, criterion, optimizer):
        with tqdm(total=len(trainloader), ncols=120, ascii=True) as t:
            scaler = GradScaler()
            if torch.cuda.is_available():
                net.to('cuda')
            net.train()
            losses = []
            acc = 0
            total = 0
            running_loss = 0
            for batch_idx, (inputs, coarsemask, labels, name) in enumerate(trainloader):
                inputs, labels, coarsemask = inputs.to(device), labels.to(device), coarsemask.to(device)
                t.set_description("Train(Epoch{}/{})".format(epoch, total_epochs))
                coarsemask = coarsemask.unsqueeze(1).cuda()
                with autocast():
                    inputs = torch.cat([inputs,coarsemask],dim=1)
                    predictions = net(inputs,coarsemask)
                    loss = criterion(predictions, labels.squeeze())
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                with torch.no_grad():
                    # print(torch.max(predictions, 1)[1])
                    # print(torch.max(predictions, 1))
                    acc += (torch.max(predictions, 1)[1].view(labels.size()).data == labels.data).sum()
                    total += trainloader.batch_size
                    running_loss += loss.item()
                epoch_acc = 100 * ((acc.item()) / total)
                t.set_postfix(train_loss='{:.3f}'.format(running_loss / (batch_idx + 1)),
                              train_acc='{:.2f}%'.format(epoch_acc))
                t.update(1)
            epoch_loss = running_loss / len(trainloader.dataset)
        with tqdm(total=len(valloader), ncols=120, ascii=True) as t:
            net.eval()
            correct_val = 0
            total_val = 0
            val_running_loss = 0.0
            with torch.no_grad():
                for batch_idx, (inputs, coarsemask, labels, name) in enumerate(valloader):
                    t.set_description("val(Epoch{}/{})".format(epoch, total_epochs))
                    inputs, labels, coarsemask = inputs.to(device), labels.to(device), coarsemask.to(device)
                    coarsemask = coarsemask.unsqueeze(1).cuda()
                    inputs = torch.cat([inputs,coarsemask],dim=1)

                    out = net(inputs,coarsemask)
                    correct_val += (torch.max(out, 1)[1].view(labels.size()).data == labels.data).sum()
                    total_val += valloader.batch_size
                    epoch_val_acc = 100 * ((correct_val.item()) / total_val)
                    loss = criterion(out, labels.squeeze())
                    val_running_loss += loss.item()

                    t.set_postfix(val_loss='{:.3f}'.format(val_running_loss / (batch_idx + 1)),
                                  val_acc='{:.2f}%'.format(epoch_val_acc))
                    t.update(1)
            epoch_val_loss = val_running_loss / len(valloader.dataset)
            return epoch_acc, epoch_loss, epoch_val_acc, epoch_val_loss

    def save_checkpoint():
        filename = os.path.join(checkpoint_dir, "skin_my_cls_DAC_CAC_Input_CAT_CAM.pth")
        torch.save(net.state_dict(), filename)

    def adjust_learning_rate():
        nonlocal lr
        lr = lr / lr_decay
        return torch.optim.AdamW(net.parameters(), lr, weight_decay=1e-1)

    lr = 1e-4
    optimizer = torch.optim.AdamW(net.parameters(), lr, weight_decay=1e-1)

    # initialize the accuracy
    acc = 0.0
    for epoch in range(start_epoch, total_epochs):

        train_acc, epoch_loss, epoch_val_acc, epoch_val_loss = train_val(epoch,net,trainloader,valloader, criterion, optimizer)
        writer.add_scalar("train acc", train_acc, epoch)
        writer.add_scalar("train loss", epoch_loss, epoch)
        writer.add_scalar("val accuracy", epoch_val_acc, epoch)
        writer.add_scalar("val loss", epoch_val_loss, epoch)

        if (epoch + 1) in np.cumsum(stage_epochs)[:-1]:
            stage += 1
            optimizer = adjust_learning_rate()

            print('Step into next stage')
        if epoch_val_acc > acc or epoch_val_acc == acc:
            acc = epoch_val_acc
            print("save the checkpoint, the accuracy of validation is {}".format(acc))
            save_checkpoint()

def test_data():
    net.eval()
    net.load_state_dict(torch.load(
        '/mnt/ai2019/zxg_FZU/seg_and_cls_projects/my_secode_paper_source_code/checkpoint/my_cls_checkpoint/skin_my_cls/skin_my_cls_DAC_CAC_Input_CAT_CAM.pth',
        map_location=device), strict=False)
    net = net.to(device)
    test_for_csv = []
    correct_val = 0
    total_val = 0
    pred_score = []
    label_val = []
    metrics_for_csv = []
    with torch.no_grad():
        with tqdm(total=len(testloader), ncols=70, ascii=True) as t:
            for batch_idx, (inputs, coarsemask, label, name) in enumerate(testloader):
                t.set_description("Val(Epoch {}/{})".format(1, 1))
                net.eval()

                image, label, coarsemask = inputs.to(device), label.to(device), coarsemask.to(device)
                coarsemask = coarsemask.unsqueeze(1).cuda()
                inputs = torch.cat((image, coarsemask), dim=1)
                out = net(inputs,coarsemask)
                correct_val += (torch.max(out, 1)[1].view(label.size()).data == label.data).sum()
                total_val += testloader.batch_size
                epoch_val_acc = ((correct_val.item()) / total_val)

                pred_label = torch.max(out, dim=1)[1].cpu().data.numpy().item()

                pred_score.append(torch.softmax(out[0], dim=0).cpu().data.numpy())
                label_val.append(label[0].cpu().data.numpy())

                img_name = "".join(name)
                test_for_csv.append([img_name, pred_label])
                t.set_postfix(batch_idx='{:.3f}'.format(batch_idx), )
                t.update(1)

            pro_score = np.array(pred_score)
            label_val = np.array(label_val)
            num = len(pro_score[1])

            pro_score_all = np.array(pro_score)
            binary_score_all = np.eye(num)[np.argmax(np.array(pro_score), axis=-1)]
            label_val_all = np.eye(num)[np.int64(np.array(label_val))]
            if num == 3:
                metrics_for_csv.append(['melanoma', 'seborrheic_keratosis','normal'])
            else:
                metrics_for_csv.append(['baso', 'eosi','lymp','mono','mono'])

            metrics_for_csv.append(['acc', 'AP', 'auc', 'f1_score', 'precision', 'recall','jaccard', 'sens', 'spec'])
            for i in range(num):
                label_val_cls0 = label_val_all[:, i-1]
                pred_prob_cls0 = pro_score_all[:, i-1]
                binary_score_cls0 = binary_score_all[:, i-1]
                acc, AP, auc, f1_score, precision, recall, jaccard, sens, spec = cla_evaluate(label_val_cls0,
                                                                                              binary_score_cls0,
                                                                                              pred_prob_cls0)
                line_test_cls0 = "test:acc=%f,AP=%f,auc=%f,f1_score=%f,precision=%f,recall=%f,sens=%f,spec=%f\n" % (
                    acc, AP, auc, f1_score, precision, recall, sens, spec)
                print(line_test_cls0)
                metrics_for_csv.append([acc, AP, auc, f1_score, precision, recall, jaccard, sens, spec])

            results_file = open(result_path + '/skin_my_cls_DAC_CAC_Input_CAT_CAM.csv', 'w', newline='')
            csv_writer = csv.writer(results_file, dialect='excel')
            for row in metrics_for_csv:
                csv_writer.writerow(row)

if __name__ == '__main__':
    train_main()
    test_data()