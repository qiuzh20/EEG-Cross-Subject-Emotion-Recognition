import torch
import torch.optim as optim
from torch.nn import functional as F
from model import EEGNet, EEGNet2D
from dataset import Onesec_Dataset, Contrastive_Dataset
import random

import numpy as np
import argparse
import os
import time
import json
import utils

from tqdm import tqdm
from torch.utils.data import DataLoader, random_split

def str2bool(v):
    """Method to map string to bool for argument parser"""
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    if v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    raise argparse.ArgumentTypeError('Boolean value expected.')

def train(trainloader, epoch, net, optimizer, logger, args, device='cpu'):
    print('\nEpoch: %d' % epoch)
    net.train()
    correct = 0
    total = 0
    total_loss = 0
    total_len = len(trainloader)
    pbar = tqdm(enumerate(trainloader), total=total_len)
    steps = epoch * total_len
    for batch_idx, (inputs, targets) in pbar:
        # inputs, targets = inputs.to(device), targets.to(device)
        # targets = targets.to(device)
        if args.total_shuffle:
            if args.model == 'EEG1D':
                inputs = inputs.transpose(1,2)
            else:
                inputs = inputs.unsqueeze(1)
            outputs = net(inputs)
            loss = F.cross_entropy(outputs, targets) #/ grad_acc
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1)
            optimizer.step()
            optimizer.zero_grad()
            total_loss += loss.item()
            # 检验训练准确率
            predicted = torch.argmax(outputs, dim=-1)
            correct += predicted.eq(targets).sum().item()
            total += targets.size(0)
        else:
            for i in range(30):
                temp_inputs = inputs[:, i*125:(i+1)*125]
                if args.model == 'EEG1D':
                    temp_inputs = temp_inputs.transpose(1,2)
                else:
                    temp_inputs = temp_inputs.unsqueeze(1)
                outputs = net(temp_inputs)
                loss = F.cross_entropy(outputs, targets) #/ grad_acc
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 1)
                optimizer.step()
                optimizer.zero_grad()
                total_loss += loss.item()
                # 检验训练准确率
                predicted = torch.argmax(outputs, dim=-1)
                correct += predicted.eq(targets).sum().item()
                total += targets.size(0)
            
        steps += 1
        train_acc = 100.*correct/total
        # 记录
        if steps % 10 == 0:
            logger.log('train/loss', total_loss/30, steps)
            logger.log('train/acc', train_acc, steps)
            pbar.set_description(f"epoch {epoch+1} iter {steps}: train loss {loss.item():.5f}, acc {train_acc:.3}")
        

def contrastive_train(trainloader, epoch, net, optimizer, logger, args, device='cpu'):
    print('\nEpoch: %d' % epoch)
    net.train()
    total_loss = 0
    total_len = len(trainloader)
    pbar = tqdm(enumerate(trainloader), total=total_len)
    temp_step = 0
    steps = epoch * total_len
    for batch_idx, batch in pbar:
        # batch = (batch[0].to(device), batch[1].to(device), batch[2].to(device))
        if args.model == 'EEG1D':
            # 使用1D组合卷积的情况
            bs, trials = batch[2].shape[0], batch[2].shape[1]
            # print(batch[0].shape, batch[1].shape, batch[2].shape) 
            # 计算样本A中的用于比较的特征
            test_feature = net(batch[0].float().transpose(1,2), output_feature=True) # (bs, feature dim)
            # 计算样本B中的pos参考特征
            pos_ref = net(batch[1].float().transpose(1,2), output_feature=True) # (bs, feature dim)
            # 计算负向的参考特征
            # 此处为了卷积计算，需要先将 (bs, 2*neg_number+1, sample_interval*frequency, channel) 改为 ((bs*2*neg_number+1), sample_interval*frequency, channel)
            neg_ref = net(batch[2].reshape(-1, batch[2].shape[2], batch[2].shape[3]).float().transpose(1,2), output_feature=True).reshape(bs, trials, -1) # (bs, 2*neg_number+1, feature dim)
        else:
            # 使用2D卷积的情况
            bs, trials = batch[2].shape[0], batch[2].shape[1]
            # (bs, sample_interval*frequency, channel),  (bs, sample_interval*frequency, channel),  (bs, 2*neg_number+1, sample_interval*frequency, channel)
            # 计算样本A中的用于比较的特征
            test_feature = net(batch[0].float().unsqueeze(1), output_feature=True) # (bs, feature dim)
            # 计算样本B中的pos参考特征
            pos_ref = net(batch[1].float().unsqueeze(1), output_feature=True) # (bs, feature dim)
            # 计算负向的参考特征
            # 此处为了卷积计算，需要先将 (bs, 2*neg_number+1, sample_interval*frequency, channel) 改为 ((bs*2*neg_number+1), sample_interval*frequency, channel)
            neg_ref = net(batch[2].reshape(-1, batch[2].shape[2], batch[2].shape[3]).float().unsqueeze(1), output_feature=True).reshape(bs, trials, -1) # (bs, 2*neg_number+1, feature dim)
        # 计算contrastive loss，具体请参考utils中的函数
        loss = utils.contrastive_loss(test_feature, pos_ref, neg_ref) #/ grad_acc
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1)
        # if batch_idx % grad_acc == 0:
        #     optimizer.step()
        #     optimizer.zero_grad()
        optimizer.step()
        optimizer.zero_grad()
        total_loss += loss.item()
        steps += 1
        temp_step += 1
        pbar.set_description(f"epoch {epoch+1} step {steps}:  train loss {loss.item():.5f}")
        logger.log('train/loss_contra', loss.item(), steps)
    
    avg_loss = total_loss/temp_step
    
    return avg_loss
    
   
def test(testloader, epoch, net, logger, args, device='cpu'):
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            # 依次检测30s的正确率
            if args.total_shuffle:
                if args.model == 'EEG1D':
                    inputs = inputs.transpose(1,2)
                else:
                    inputs = inputs.unsqueeze(1)
                outputs = net(inputs)
                loss = F.cross_entropy(outputs, targets)
                test_loss += loss.item()
                predicted = torch.argmax(outputs, dim=-1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().sum().item()
            else:
                for i in range(30):
                    if args.model == 'EEG1D':
                        temp_inputs = inputs[:, i*125:(i+1)*125].transpose(1,2)
                    else:
                        temp_inputs = inputs[:, i*125:(i+1)*125].unsqueeze(1)
                    outputs = net(temp_inputs)
                    loss = F.cross_entropy(outputs, targets)
                    test_loss += loss.item()
                    predicted = torch.argmax(outputs, dim=-1)
                    total += targets.size(0)
                    correct += predicted.eq(targets).sum().item()

    acc = 100.*correct/total
    test_loss = test_loss/(batch_idx+1)
    test_loss = test_loss if args.total_shuffle else test_loss/30
    logger.log('eval/loss', test_loss, epoch)
    logger.log('eval/acc', acc, epoch)
    print('Finish eval epoch {}: loss {:.5}, acc {:.3}'.format(epoch, test_loss, acc))
    return acc



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='./data.pth')
    # training parameter
    parser.add_argument("--seed", default=0, type=int)              # Sets PyTorch and Numpy seeds
    parser.add_argument("--save_model", default=False, action="store_true")        # Save model and optimizer parameters
    parser.add_argument('--lr', type=float, default=0.0005, help='initial learning rate')
    parser.add_argument('--epochs', type=int, default=100, help='upper epoch limit')
    parser.add_argument('--batch_size', type=int, default=128, metavar='N',help='batch size')
    parser.add_argument('--num_workers', type=int, default=0)

    parser.add_argument("--contrastive_training",  default=False, action="store_true")
    parser.add_argument('--sample_interval', type=int, default=5)
    parser.add_argument('--neg_number', type=int, default=10)

    parser.add_argument("--normalize",  default=False, action="store_true")
    parser.add_argument('--drop_out', type=float, default=0, help='initial learning rate')
    
    parser.add_argument("--random_label_test",  default=False, action="store_true")
    
    parser.add_argument("--total_shuffle",  default=False, action="store_true")

    parser.add_argument('--weight_decay', default=0, type=float)
    parser.add_argument('--save_freq', type=int, default=50)
    
    parser.add_argument('--ckpt_path', type=str, default=None)
    # model type
    parser.add_argument('--model', type=str, default='EEG1D',help='type of EEGNet')
    # EEG1D 网络使用参数
    parser.add_argument('--F1', type=int, default=16)
    parser.add_argument('--D', type=int, default=4)
    # EEG2D 网络使用参数
    
    # Work dir
    parser.add_argument('--notes', default=None, type=str) # additional experiment name
    parser.add_argument('--work_dir', default='exp3_refnet', type=str)
    args = parser.parse_args()
    
    # 固定随机种子
    utils.fix_seed(args.seed)
    
    # 准备实验路径，区分不同实验
    base_dir = 'runs'
    utils.make_dir(base_dir)
    args.work_dir = os.path.join(base_dir, args.work_dir)
    utils.make_dir(args.work_dir)
    args.work_dir = os.path.join(args.work_dir, 'contrastive' if args.contrastive_training else 'supervised')
    utils.make_dir(args.work_dir)
    args.work_dir = os.path.join(args.work_dir, args.model)
    utils.make_dir(args.work_dir)

    ts = time.gmtime()
    ts = time.strftime("%m-%d-%H:%M", ts)
    exp_name = f'dropout{args.drop_out}-weight_decay{args.weight_decay}' + f'norm{args.normalize}' + 'bs' + str(args.batch_size) + '-s' + str(args.seed)
    if args.model == 'EEG1D':
        exp_name = f'F1{args.F1}-D{args.D}' + exp_name
    if args.contrastive_training:
        exp_name = f'interv{args.sample_interval}-neg{args.neg_number}' + exp_name
    if args.notes is not None:
        exp_name = args.notes + '_' + exp_name
    if args.random_label_test:
        exp_name = 'random_label_test' + exp_name
    if args.total_shuffle:
        exp_name = 'total_shuffle' + exp_name

    exp_name += '-' + ts
    args.work_dir = args.work_dir + '/' + exp_name
    utils.make_dir(args.work_dir)

    # 准备储存模型的路径
    args.model_dir = os.path.join(args.work_dir, 'model')
    utils.make_dir(args.model_dir)
    
    # 记录实验参数
    with open(os.path.join(args.work_dir, 'args.json'), 'w') as f:
        json.dump(vars(args), f, sort_keys=True, indent=4)
    
    raw_data = torch.load(args.data_dir)
    X_total, y_total = raw_data['X'], raw_data['y']
    
    # train_X, train_y = X_total[:70], y_total[:70]
    # test_X, test_y = X_total[70:], y_total[:70]
    
    # train_dataset = Onesec_Dataset(data=train_X.reshape(70*28, 3750, 32), label=train_y.reshape(70*28, 1))
    # train_dataset = Onesec_Dataset(data=test_X.reshape(10*28, 3750, 32), label=test_y.reshape(10*28, 1))

    if args.model == 'EEG1D':
        model = EEGNet(nb_classes=9, Chans=32, Samples=125, F1=args.F1, D=args.D, dropoutRate=args.drop_out)
    else:
        model = EEGNet2D(nb_classes=9, dropout=args.drop_out)
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    if args.ckpt_path is not None:
        print("loading %s", args.ckpt_path)
        model.load_state_dict(torch.load(args.ckpt_path))

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Model Built with Total Number of Trainable Parameters: " + str(total_params)) 

    with open(os.path.join(args.work_dir, 'total_param_{}.txt'.format(total_params)), 'w') as f:
        pass

    logger = utils.Logger(args.work_dir, use_tb=True)
    
    
    
    if args.contrastive_training:
        best_loss = np.inf
        dataset = Contrastive_Dataset(X_total, sample_interval=args.sample_interval, neg_number=args.neg_number, channel_wise_normalize=args.normalize, device=device)
        train_loader = DataLoader(dataset=dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True)
    
        for epoch in range(args.epochs):
            avg_loss = contrastive_train(train_loader, epoch, model, optimizer, logger, args, device)
            if avg_loss < best_loss:
                best_loss = avg_loss
                torch.save(model.state_dict(), args.model_dir + '/ckpt_contrastive.pth')
    else:
        best_acc = 0.1
        if args.random_label_test:
            random.shuffle(y_total)
        train_X, train_y = X_total[:70], y_total[:70]
        test_X, test_y = X_total[70:], y_total[70:]
        
        if args.total_shuffle: # 整体打乱处理
            train_dataset = Onesec_Dataset(data=train_X.reshape(70*28*30, 125, 32), label=train_y.repeat_interleave(30, 1).reshape(70*28*30,), channel_wise_normalize=args.normalize, device=device)
            test_dataset = Onesec_Dataset(data=test_X.reshape(10*28*30, 125, 32), label=test_y.repeat_interleave(30, 1).reshape(10*28*30,), channel_wise_normalize=args.normalize, device=device)
        else: # 最初发现将30s长的片段整体输入会导致训练loss出现周期波折，担心模型学习到错误的信号
            train_dataset = Onesec_Dataset(data=train_X.reshape(70*28, 125*30, 32), label=train_y.reshape(70*28,), channel_wise_normalize=args.normalize, device=device)
            test_dataset = Onesec_Dataset(data=test_X.reshape(10*28, 125*30, 32), label=test_y.reshape(10*28,), channel_wise_normalize=args.normalize, device=device)
            # train_dataset, test_dataset = random_split(dataset=dataset, lengths=[70*28*30, 10*28*30]) 早期采用了直接随机拆分，但是这样会让训练和侧视数据中出现相同的测试被试
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=32)

        for epoch in range(args.epochs):
            train(train_loader, epoch, model, optimizer, logger, args, device)
            acc = test(test_loader, epoch, model, logger, args, device)
            if acc > best_acc:
                if (epoch+1) % args.save_freq == 0 and args.save_model:
                    print('Saving ckpt {} with acc'.format(epoch, acc))
                    torch.save(model.state_dict(), args.model_dir + '/ckpt_{}_acc_{:.3}.pth'.format(epoch+1, acc))
                    best_acc = acc
            # scheduler.step()

    logger._sw.close()

    