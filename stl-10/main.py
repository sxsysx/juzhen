from __future__ import print_function
import os
import random
import numpy as np
import argparse

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import torch.nn.parallel
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import recall_score, f1_score

import model_loader
import dataloader
import matplotlib
matplotlib.use('TkAgg')  # 设置一个支持显示的后端
import matplotlib.pyplot as plt
import sys

def init_params(net):
    for m in net.modules():
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            init.kaiming_normal_(m.weight, mode='fan_in')
            if m.bias is not None:
                init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            init.constant_(m.weight, 1)
            init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            init.normal_(m.weight, std=1e-3)
            if m.bias is not None:
                init.constant_(m.bias, 0)

# Training
def train(trainloader, net, criterion, optimizer, use_cuda=True):
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    all_targets = []
    all_predictions = []

    if isinstance(criterion, nn.CrossEntropyLoss):
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            batch_size = inputs.size(0)
            total += batch_size
            if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()
            optimizer.zero_grad()
            inputs, targets = Variable(inputs), Variable(targets)
            #print(inputs.size())
            #sys.exit(1)
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()*batch_size
            _, predicted = torch.max(outputs.data, 1)
            correct += predicted.eq(targets.data).cpu().sum().item()
            
            # 收集所有目标和预测结果用于计算召回率和F1分数
            all_targets.extend(targets.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

    elif isinstance(criterion, nn.MSELoss):
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            batch_size = inputs.size(0)
            total += batch_size

            one_hot_targets = torch.FloatTensor(batch_size, 10).zero_()
            one_hot_targets = one_hot_targets.scatter_(1, targets.view(batch_size, 1), 1.0)
            one_hot_targets = one_hot_targets.float()
            if use_cuda:
                inputs, one_hot_targets = inputs.cuda(), one_hot_targets.cuda()
            inputs, one_hot_targets = Variable(inputs), Variable(one_hot_targets)
            outputs = F.softmax(net(inputs))
            loss = criterion(outputs, one_hot_targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()*batch_size
            _, predicted = torch.max(outputs.data, 1)
            correct += predicted.cpu().eq(targets).cpu().sum().item()
            
            # 收集所有目标和预测结果用于计算召回率和F1分数
            all_targets.extend(targets.numpy())
            all_predictions.extend(predicted.cpu().numpy())
    
    # 计算召回率和F1分数
    recall = recall_score(all_targets, all_predictions, average='macro')
    f1 = f1_score(all_targets, all_predictions, average='macro')

    return train_loss/total, 100 - 100.*correct/total, recall, f1


def test(testloader, net, criterion, use_cuda=True):
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    all_targets = []
    all_predictions = []

    if isinstance(criterion, nn.CrossEntropyLoss):
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(testloader):
                batch_size = inputs.size(0)
                total += batch_size

                if use_cuda:
                    inputs, targets = inputs.cuda(), targets.cuda()
                inputs, targets = Variable(inputs), Variable(targets)
                outputs = net(inputs)
                loss = criterion(outputs, targets)
                test_loss += loss.item()*batch_size
                _, predicted = torch.max(outputs.data, 1)
                correct += predicted.eq(targets.data).cpu().sum().item()
                
                # 收集所有目标和预测结果用于计算召回率和F1分数
                all_targets.extend(targets.cpu().numpy())
                all_predictions.extend(predicted.cpu().numpy())

    elif isinstance(criterion, nn.MSELoss):
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(testloader):
                batch_size = inputs.size(0)
                total += batch_size

                one_hot_targets = torch.FloatTensor(batch_size, 10).zero_()
                one_hot_targets = one_hot_targets.scatter_(1, targets.view(batch_size, 1), 1.0)
                one_hot_targets = one_hot_targets.float()
                if use_cuda:
                    inputs, one_hot_targets = inputs.cuda(), one_hot_targets.cuda()
                inputs, one_hot_targets = Variable(inputs), Variable(one_hot_targets)
                outputs = F.softmax(net(inputs))
                loss = criterion(outputs, one_hot_targets)
                test_loss += loss.item()*batch_size
                _, predicted = torch.max(outputs.data, 1)
                correct += predicted.cpu().eq(targets).cpu().sum().item()
                
                # 收集所有目标和预测结果用于计算召回率和F1分数
                all_targets.extend(targets.numpy())
                all_predictions.extend(predicted.cpu().numpy())
    
    # 计算召回率和F1分数
    recall = recall_score(all_targets, all_predictions, average='macro')
    f1 = f1_score(all_targets, all_predictions, average='macro')

    return test_loss/total, 100 - 100.*correct/total, recall, f1

def train_semisupervised(labeled_loader, unlabeled_loader, net, criterion, optimizer, 
                        transform_weak, transform_strong, args, use_cuda=True):
    """
    半监督学习训练函数，结合伪标签和一致性正则化
    
    Args:
        labeled_loader: 有标签数据加载器
        unlabeled_loader: 无标签数据加载器
        net: 神经网络模型
        criterion: 损失函数
        optimizer: 优化器
        transform_weak: 弱数据增强变换
        transform_strong: 强数据增强变换
        args: 参数字典，包含半监督学习相关参数
        use_cuda: 是否使用GPU
    """
    net.train()
    train_loss = 0
    supervised_loss = 0
    consistency_loss = 0
    correct = 0
    total = 0
    all_targets = []
    all_predictions = []
    
    # 获取数据集的类别数
    num_classes = getattr(args, 'num_classes', 10)
    
    # 确保有标签和无标签数据加载器的迭代次数一致
    labeled_iter = iter(labeled_loader)
    unlabeled_iter = iter(unlabeled_loader)
    max_iterations = max(len(labeled_loader), len(unlabeled_loader))
    
    for batch_idx in range(max_iterations):
        # 获取有标签数据批次
        try:
            inputs_labeled, targets_labeled = next(labeled_iter)
        except StopIteration:
            labeled_iter = iter(labeled_loader)
            inputs_labeled, targets_labeled = next(labeled_iter)
        
        # 获取无标签数据批次
        try:
            inputs_unlabeled, _ = next(unlabeled_iter)
        except StopIteration:
            unlabeled_iter = iter(unlabeled_loader)
            inputs_unlabeled, _ = next(unlabeled_iter)
        
        # 移动到GPU（如果可用）
        if use_cuda:
            inputs_labeled, targets_labeled = inputs_labeled.cuda(), targets_labeled.cuda()
            inputs_unlabeled = inputs_unlabeled.cuda()
        
        # 转换为Variable
        inputs_labeled, targets_labeled = Variable(inputs_labeled), Variable(targets_labeled)
        inputs_unlabeled = Variable(inputs_unlabeled)
        
        # 重置梯度
        optimizer.zero_grad()
        
        # 1. 计算有标签数据的监督损失
        outputs_labeled = net(inputs_labeled)
        loss_supervised = criterion(outputs_labeled, targets_labeled)
        
        # 2. 计算无标签数据的一致性损失
        # 对无标签数据应用弱增强和强增强
        inputs_unlabeled_weak = inputs_unlabeled.clone()
        
        # 应用强增强（这里简化处理）
        inputs_unlabeled_strong = inputs_unlabeled.clone()
        
        # 对于已经在GPU上的数据，我们需要先移到CPU，应用变换，再移回GPU
        if use_cuda:
            inputs_unlabeled_strong = inputs_unlabeled_strong.cpu()
        
        # 应用强增强（这里使用简单的随机翻转作为示例）
        batch_size = inputs_unlabeled_strong.size(0)
        for i in range(batch_size):
            # 随机水平翻转
            if torch.rand(1).item() > 0.5:
                inputs_unlabeled_strong[i] = inputs_unlabeled_strong[i].flip(2)
        
        if use_cuda:
            inputs_unlabeled_strong = inputs_unlabeled_strong.cuda()
        
        # 禁用梯度计算，使用模型当前状态生成伪标签
        with torch.no_grad():
            outputs_unlabeled_weak = net(inputs_unlabeled_weak)
            # 使用温度缩放的softmax来软化伪标签
            soft_predictions = F.softmax(outputs_unlabeled_weak / args.temperature, dim=1)
            # 获取置信度最高的类别作为伪标签
            max_probs, pseudo_labels = torch.max(soft_predictions, dim=1)
            # 只保留置信度高于阈值的伪标签
            mask = max_probs.ge(args.confidence_threshold).float()
        
        # 使用强增强后的图像进行预测
        outputs_unlabeled_strong = net(inputs_unlabeled_strong)
        
        # 计算一致性损失（MSE损失用于一致性正则化）
        consistency_criterion = nn.MSELoss(reduction='none')
        loss_consistency = consistency_criterion(
            F.softmax(outputs_unlabeled_strong / args.temperature, dim=1),
            soft_predictions.detach()
        ).mean(dim=1) * mask
        loss_consistency = loss_consistency.mean()
        
        # 3. 组合监督损失和一致性损失
        loss = loss_supervised + args.consistency_weight * loss_consistency
        
        # 反向传播和优化
        loss.backward()
        optimizer.step()
        
        # 更新统计信息
        batch_size_labeled = inputs_labeled.size(0)
        train_loss += loss.item() * batch_size_labeled
        supervised_loss += loss_supervised.item() * batch_size_labeled
        consistency_loss += loss_consistency.item() * batch_size_labeled
        
        # 计算准确率
        _, predicted = torch.max(outputs_labeled.data, 1)
        correct += predicted.eq(targets_labeled.data).cpu().sum().item()
        total += batch_size_labeled
        
        # 收集所有目标和预测结果用于计算召回率和F1分数
        all_targets.extend(targets_labeled.cpu().numpy())
        all_predictions.extend(predicted.cpu().numpy())
    
    # 计算召回率和F1分数
    recall = recall_score(all_targets, all_predictions, average='macro')
    f1 = f1_score(all_targets, all_predictions, average='macro')
    
    # 返回平均损失和性能指标
    return train_loss/total, supervised_loss/total, consistency_loss/total, \
           100 - 100.*correct/total, recall, f1

def name_save_folder(args):
    save_folder = args.dataset + '_'
    save_folder += args.model + '_' + str(args.optimizer) + '_lr=' + str(args.lr)
    if args.lr_decay != 0.1:
        save_folder += '_lr_decay=' + str(args.lr_decay)
    save_folder += '_bs=' + str(args.batch_size)
    save_folder += '_wd=' + str(args.weight_decay)
    save_folder += '_mom=' + str(args.momentum)
    save_folder += '_save_epoch=' + str(args.save_epoch)
    if args.loss_name != 'crossentropy':
        save_folder += '_loss=' + str(args.loss_name)
    if args.noaug:
        save_folder += '_noaug'
    if args.raw_data:
        save_folder += '_rawdata'
    if args.label_corrupt_prob > 0:
        save_folder += '_randlabel=' + str(args.label_corrupt_prob)
    if args.ngpu > 1:
        save_folder += '_ngpu=' + str(args.ngpu)
    if args.idx:
        save_folder += '_idx=' + str(args.idx)

    return save_folder

def find_lr(net, trainloader, optimizer, criterion, args, use_cuda=True):
    """
    Learning rate finder implementation to automatically find optimal learning rate.
    The method increases learning rate exponentially from min_lr to max_lr and records loss values.
    """

    
    # 保存当前后端，以便后续恢复
    original_backend = matplotlib.get_backend()
    print(f"进入find_lr函数，当前后端：{original_backend}")
    

    
    print("=== 开始学习率查找 ===")
    print(f"参数设置: 最小LR={args.min_lr}, 最大LR={args.max_lr}, 查找轮次={args.lr_find_epochs}")
    
    # 设置初始学习率
    current_lr = args.min_lr
    
    # 保存学习率和损失值
    lrs = []
    losses = []
    best_loss = float('inf')
    
    # 计算每个批次的学习率乘数
    num_iterations = len(trainloader) * args.lr_find_epochs
    lr_multiplier = (args.max_lr / args.min_lr) ** (1 / num_iterations)
    
    # 设置模型为训练模式
    net.train()
    
    iteration = 0
    stop_training = False
    
    try:
        for epoch in range(args.lr_find_epochs):
            if stop_training:
                break
                
            for batch_idx, (inputs, targets) in enumerate(trainloader):
                # 更新学习率
                for param_group in optimizer.param_groups:
                    param_group['lr'] = current_lr
                
                # 记录当前学习率
                lrs.append(current_lr)
                
                # 将数据移至设备
                if use_cuda:
                    try:
                        inputs, targets = inputs.cuda(), targets.cuda()
                    except:
                        print("CUDA不可用，使用CPU")
                inputs, targets = Variable(inputs), Variable(targets)
                
                # 梯度清零
                optimizer.zero_grad()
                
                # 前向传播
                outputs = net(inputs)
                
                # 根据损失函数类型计算损失
                if isinstance(criterion, nn.CrossEntropyLoss):
                    loss = criterion(outputs, targets)
                elif isinstance(criterion, nn.MSELoss):
                    one_hot_targets = torch.FloatTensor(inputs.size(0), 10).zero_()
                    one_hot_targets = one_hot_targets.scatter_(1, targets.view(inputs.size(0), 1), 1.0)
                    one_hot_targets = one_hot_targets.float()
                    if use_cuda:
                        one_hot_targets = one_hot_targets.cuda()
                    one_hot_targets = Variable(one_hot_targets)
                    outputs = F.softmax(outputs)
                    loss = criterion(outputs, one_hot_targets)
                else:
                    loss = criterion(outputs, targets)
                
                # 保存损失值
                current_loss = loss.item()
                losses.append(current_loss)
                
                # 更新最佳损失
                if current_loss < best_loss:
                    best_loss = current_loss
                
                # 反向传播
                loss.backward()
                optimizer.step()
                
                # 更新学习率
                current_lr *= lr_multiplier
                iteration += 1
                
                # 打印进度
                if batch_idx % 10 == 0:
                    print(f'批次: {batch_idx}, LR: {current_lr:.2e}, 损失: {current_loss:.4f}')
                
                # 检查是否需要停止
                if np.isnan(current_loss):
                    print(f"损失为NaN，在学习率 {current_lr:.2e} 处停止")
                    stop_training = True
                    break
                
                if current_loss > 10 * best_loss:
                    print(f"损失爆炸，在学习率 {current_lr:.2e} 处停止")
                    stop_training = True
                    break
    except KeyboardInterrupt:
        print("学习率查找被用户中断")
    except Exception as e:
        print(f"学习率查找过程中出错: {str(e)}")
    
    print(f"学习率查找完成，共记录 {len(lrs)} 个学习率点")
    
    # 移除初始不稳定的损失值
    if len(losses) > 10:
        print("移除前10个不稳定的损失值")
        lrs = lrs[10:]
        losses = losses[10:]
    
    # 找到损失下降最快的学习率
    best_lr = args.min_lr
    best_lr_idx = 0
    
    if len(losses) > 1:
        # 计算损失梯度
        gradients = np.gradient(losses)
        # 找到梯度最负的点（损失下降最快的点）
        if np.any(gradients < 0):
            best_lr_idx = np.argmin(gradients)
            best_lr = lrs[best_lr_idx]
            print(f"找到最佳学习率: {best_lr:.2e}")
        else:
            best_lr = lrs[0]  # 如果所有梯度都为正，选择最小学习率
            print(f"未找到最佳学习率，使用最小学习率: {best_lr:.2e}")
    else:
        print(f"数据不足，使用最小学习率: {best_lr:.2e}")
    
    # 确保日志目录存在 - 使用绝对路径
    log_dir = os.path.join(os.getcwd(), 'tensorboard_logs', f'{args.dataset}_{args.model}_lr_finder')
    print(f"日志目录: {log_dir}")
    os.makedirs(log_dir, exist_ok=True)
    
    # 强制创建空文件作为测试
    test_file_path = os.path.join(log_dir, 'test_file.txt')
    with open(test_file_path, 'w') as f:
        f.write('测试文件创建成功')
    print(f"测试文件创建: {test_file_path}")
    
    # 保存数据到CSV文件
    csv_path = os.path.join(log_dir, 'lr_finder_data.csv')
    try:
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Learning Rate', 'Loss'])
            for lr, loss_val in zip(lrs, losses):
                writer.writerow([lr, loss_val])
        print(f"CSV数据文件已保存至: {csv_path}")
    except Exception as e:
        print(f"保存CSV文件失败: {str(e)}")
    
    # 保存学习率与损失的关系图
    if len(lrs) > 0 and len(losses) > 0:
        try:
            # 临时设置非交互式后端用于保存图片
            matplotlib.use('Agg')
            print("临时设置matplotlib后端为Agg用于保存图片")
            
            # 创建新的图形
            plt.figure(figsize=(10, 6))
            plt.semilogx(lrs, losses, 'b-', alpha=0.7, label='Loss vs Learning Rate')
            
            # 标记最佳学习率
            if len(losses) > 0:
                plot_idx = min(best_lr_idx, len(losses) - 1)
                plt.scatter(best_lr, losses[plot_idx], 
                           color='red', marker='o', s=100, label=f'最佳学习率: {best_lr:.2e}')
            
            # 设置图表属性
            plt.xlabel('学习率 (对数刻度)', fontsize=12)
            plt.ylabel('损失值', fontsize=12)
            plt.title(f'学习率查找 - {args.dataset} {args.model}', fontsize=14)
            plt.grid(True, which='both', linestyle='--', alpha=0.7)
            plt.legend(fontsize=10)
            plt.tight_layout()
            
            # 保存图表
            plot_path = os.path.join(log_dir, 'lr_finder.png')
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"学习率查找图表已保存至: {plot_path}")
            
        except Exception as e:
            print(f"保存图表失败: {str(e)}")
            import traceback
            traceback.print_exc()
        finally:
            # 恢复原始后端
            matplotlib.use(original_backend)
            print(f"已恢复matplotlib后端为: {original_backend}")
    else:
        print("没有足够的数据生成图表")
    
    print("=== 学习率查找完成 ===")
    return best_lr

def plot_optimizer_comparison(args):
    """
    绘制不同优化器的对比图
    
    Args:
        args: 包含数据集、模型等信息的参数字典
    """
    import matplotlib.pyplot as plt
    import os
    import re
    import numpy as np
    
    # 设置中文字体支持
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 获取优化器列表
    optimizers = [opt.strip() for opt in args.optimizer_list.split(',')]
    
    # 定义要绘制的指标
    metrics = {
        'test_accuracy': {'label': '测试准确率 (%)', 'y_lim': [0, 100], 'color': ['blue', 'green', 'red', 'purple', 'orange', 'cyan']},
        'test_loss': {'label': '测试损失', 'y_lim': [0, 3], 'color': ['blue', 'green', 'red', 'purple', 'orange', 'cyan']},
        'train_loss': {'label': '训练损失', 'y_lim': [0, 3], 'color': ['blue', 'green', 'red', 'purple', 'orange', 'cyan']}
    }
    
    # 为每个指标创建一个字典，用于存储不同优化器的数据
    metric_data = {}
    for metric in metrics:
        metric_data[metric] = {}
    
    # 读取每个优化器的日志文件
    for opt_name in optimizers:
        # 构建日志文件路径
        args_copy = argparse.Namespace(**vars(args))
        args_copy.optimizer = opt_name
        save_folder = name_save_folder(args_copy)
        log_path = 'trained_nets/' + save_folder + '/log.out'
        
        if not os.path.exists(log_path):
            print(f"警告：日志文件 {log_path} 不存在，跳过该优化器 {opt_name}")
            continue
        
        # 读取日志文件
        try:
            with open(log_path, 'r') as f:
                lines = f.readlines()
        except Exception as e:
            print(f"读取日志文件 {log_path} 失败：{e}")
            continue
        
        # 初始化数据列表
        epochs = []
        test_accuracies = []
        test_losses = []
        train_losses = []
        
        # 解析日志文件
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # 跳过初始日志行
            if line.startswith('训练开始:'):
                continue
            
            # 匹配日志格式：e: 0 loss: 2.30259 train_err: 90.000 test_top1: 90.000 test_loss 2.30259 test_recall: 0.100 test_f1: 0.182
            match = re.match(r'e: (\d+) loss: ([\d.]+) train_err: ([\d.]+) test_top1: ([\d.]+) test_loss ([\d.]+) test_recall: ([\d.]+) test_f1: ([\d.]+)', line)
            if match:
                epoch = int(match.group(1))
                train_loss = float(match.group(2))
                train_err = float(match.group(3))
                test_err = float(match.group(4))
                test_loss = float(match.group(5))
                
                # 计算准确率
                test_accuracy = 100 - test_err
                
                # 保存数据
                epochs.append(epoch)
                test_accuracies.append(test_accuracy)
                test_losses.append(test_loss)
                train_losses.append(train_loss)
        
        # 保存到metric_data字典
        if epochs:
            metric_data['test_accuracy'][opt_name] = {'epochs': epochs, 'values': test_accuracies}
            metric_data['test_loss'][opt_name] = {'epochs': epochs, 'values': test_losses}
            metric_data['train_loss'][opt_name] = {'epochs': epochs, 'values': train_losses}
        else:
            print(f"警告：优化器 {opt_name} 没有有效的训练数据，跳过该优化器")
    
    # 如果没有任何优化器的数据，直接返回
    if not any(metric_data.values()):
        print("没有找到有效的训练数据，无法绘制对比图")
        return
    
    # 创建保存图表的目录
    plot_dir = 'plots'
    os.makedirs(plot_dir, exist_ok=True)
    
    # 绘制每个指标的对比图
    for metric_name, metric_info in metrics.items():
        plt.figure(figsize=(12, 6))
        
        # 为每个优化器绘制曲线
        plotted_optimizers = 0
        for i, (opt_name, data) in enumerate(metric_data[metric_name].items()):
            if len(data['epochs']) > 0:
                plt.plot(data['epochs'], data['values'], 
                        label=opt_name, 
                        color=metric_info['color'][i % len(metric_info['color'])],
                        linewidth=2, 
                        marker='o', 
                        markersize=4)
                plotted_optimizers += 1
        
        # 如果没有绘制任何曲线，跳过该指标
        if plotted_optimizers == 0:
            print(f"警告：指标 {metric_name} 没有有效的数据，跳过该指标")
            plt.close()
            continue
        
        # 设置图表属性
        plt.title(f'{args.dataset} {args.model} 不同优化器{metric_info["label"]}对比', fontsize=14)
        plt.xlabel('训练轮次 (Epoch)', fontsize=12)
        plt.ylabel(metric_info['label'], fontsize=12)
        plt.ylim(metric_info['y_lim'])
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(fontsize=10)
        plt.tight_layout()
        
        # 保存图表
        plot_path = os.path.join(plot_dir, f'{args.dataset}_{args.model}_{metric_name}_comparison.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"图表已保存至: {plot_path}")
        
        # 显示图表
        plt.show()
    
    print("所有优化器对比图绘制完成！")


def train_with_optimizer(args, optimizer_name):
    # 创建一个新的参数副本，修改优化器名称
    args_copy = argparse.Namespace(**vars(args))
    args_copy.optimizer = optimizer_name
    
    # 设置保存文件夹名称，包含优化器信息
    save_folder = name_save_folder(args_copy)
    
    # 创建保存目录
    os.makedirs('trained_nets/' + save_folder, exist_ok=True)
    
    # 创建Tensorboard日志目录，使用优化器名称作为子目录
    tb_log_dir = 'tensorboard_logs/comparison/' + args.dataset + '_' + args.model + '/' + optimizer_name
    os.makedirs(tb_log_dir, exist_ok=True)
    
    # 初始化SummaryWriter
    writer = SummaryWriter(tb_log_dir)
    
    # 打开日志文件
    log_path = 'trained_nets/' + save_folder + '/log.out'
    print(f"日志文件路径: {log_path}")
    f = open(log_path, 'a', buffering=1)  # 行缓冲模式，确保实时写入
    
    # 加载数据
    trainloader, testloader = dataloader.get_data_loaders(args_copy)
    
    if args_copy.label_corrupt_prob and not args_copy.resume_model:
        torch.save(trainloader, 'trained_nets/' + save_folder + '/trainloader.dat')
        torch.save(testloader, 'trained_nets/' + save_folder + '/testloader.dat')
    
    # 加载模型
    if args_copy.resume_model:
        print('\n==> Resuming from checkpoint..')
        checkpoint = torch.load(args_copy.resume_model)
        net = model_loader.load(args_copy.model)
        net.load_state_dict(checkpoint['state_dict'])
        start_epoch = checkpoint['epoch'] + 1
    else:
        net = model_loader.load(args_copy.model)
        init_params(net)
    
    if args_copy.ngpu > 1:
        net = torch.nn.DataParallel(net)
    
    # 设置损失函数
    criterion = nn.CrossEntropyLoss()
    if args_copy.loss_name == 'mse':
        criterion = nn.MSELoss()
    
    # 将模型和损失函数移至GPU
    if use_cuda:
        net.cuda()
        criterion = criterion.cuda()
    
    # 创建优化器
    if optimizer_name == 'sgd':
        optimizer = optim.SGD(net.parameters(), lr=args_copy.lr, momentum=args_copy.momentum, weight_decay=args_copy.weight_decay, nesterov=True)
    elif optimizer_name == 'sgd_momentum':
        optimizer = optim.SGD(net.parameters(), lr=args_copy.lr, momentum=args_copy.momentum, weight_decay=args_copy.weight_decay, nesterov=False)
    elif optimizer_name == 'adam':
        optimizer = optim.Adam(net.parameters(), lr=args_copy.lr, weight_decay=args_copy.weight_decay)
    elif optimizer_name == 'adamw':
        optimizer = optim.AdamW(net.parameters(), lr=args_copy.lr, weight_decay=args_copy.weight_decay)
    elif optimizer_name == 'nadam':
        # PyTorch 1.10+ 支持Nadam，如果版本不兼容需要使用第三方库
        optimizer = optim.RAdam(net.parameters(), lr=args_copy.lr, weight_decay=args_copy.weight_decay)
    elif optimizer_name == 'rmsprop':
        optimizer = optim.RMSprop(net.parameters(), lr=args_copy.lr, momentum=args_copy.momentum, weight_decay=args_copy.weight_decay)
    elif optimizer_name == 'adagrad':
        optimizer = optim.Adagrad(net.parameters(), lr=args_copy.lr, weight_decay=args_copy.weight_decay)
    else:
        raise ValueError(f'Unknown optimizer: {optimizer_name}')
    
    if args_copy.resume_opt:
        checkpoint_opt = torch.load(args_copy.resume_opt)
        optimizer.load_state_dict(checkpoint_opt['optimizer'])
    
    # 记录初始模型性能
    if not args_copy.resume_model:
        train_loss, train_err, train_recall, train_f1 = test(trainloader, net, criterion, use_cuda)
        test_loss, test_err, test_recall, test_f1 = test(testloader, net, criterion, use_cuda)
        status = 'e: %d loss: %.5f train_err: %.3f test_top1: %.3f test_loss %.5f test_recall: %.3f test_f1: %.3f \n' % (0, train_loss, train_err, test_err, test_loss, test_recall, test_f1)
        print(f'[{optimizer_name}] ' + status)
        f.write(status)
        
        # 记录初始状态到Tensorboard
        writer.add_scalar('Loss/train', train_loss, 0)
        writer.add_scalar('Loss/test', test_loss, 0)
        writer.add_scalar('Error/train', train_err, 0)
        writer.add_scalar('Error/test', test_err, 0)
        writer.add_scalar('Accuracy/test', 100 - test_err, 0)
        writer.add_scalar('Recall/test', test_recall, 0)
        writer.add_scalar('F1/test', test_f1, 0)
        writer.add_scalar('Learning Rate', args_copy.lr, 0)
        
        state = {
            'acc': 100 - test_err,
            'epoch': 0,
            'state_dict': net.module.state_dict() if args_copy.ngpu > 1 else net.state_dict()
        }
        opt_state = {
            'optimizer': optimizer.state_dict()
        }
        torch.save(state, 'trained_nets/' + save_folder + '/model_0.t7')
        torch.save(opt_state, 'trained_nets/' + save_folder + '/opt_state_0.t7')
    
    best_acc = 0.0
    lr = args_copy.lr
    start_epoch = 1 if not args_copy.resume_model else start_epoch
    
    for epoch in range(start_epoch, args_copy.epochs + 1):
        loss, train_err, train_recall, train_f1 = train(trainloader, net, criterion, optimizer, use_cuda)
        test_loss, test_err, test_recall, test_f1 = test(testloader, net, criterion, use_cuda)
        
        status = 'e: %d loss: %.5f train_err: %.3f test_top1: %.3f test_loss %.5f test_recall: %.3f test_f1: %.3f \n' % (epoch, loss, train_err, test_err, test_loss, test_recall, test_f1)
        print(f'[{optimizer_name}] ' + status)
        f.write(status)
        
        # 记录到Tensorboard
        writer.add_scalar('Loss/train', loss, epoch)
        writer.add_scalar('Loss/test', test_loss, epoch)
        writer.add_scalar('Error/train', train_err, epoch)
        writer.add_scalar('Error/test', test_err, epoch)
        writer.add_scalar('Accuracy/test', 100 - test_err, epoch)
        writer.add_scalar('Recall/train', train_recall, epoch)
        writer.add_scalar('Recall/test', test_recall, epoch)
        writer.add_scalar('F1/train', train_f1, epoch)
        writer.add_scalar('F1/test', test_f1, epoch)
        writer.add_scalar('Learning Rate', lr, epoch)
        writer.flush()  # 刷新Tensorboard写入
        
        # 记录半监督学习特有的损失指标
        if args.use_semisupervised:
            # 确保sup_loss和unsup_loss变量已定义
            if 'sup_loss' in locals() and 'unsup_loss' in locals():
                writer.add_scalar('Loss/supervised', sup_loss, epoch)
                writer.add_scalar('Loss/unsupervised', unsup_loss, epoch)
        
        # 保存检查点
        acc = 100 - test_err
        
        if epoch == 1 or epoch % args_copy.save_epoch == 0 or epoch == 150 or (acc > best_acc and epoch > (args_copy.epochs - 40)):
            if (acc > best_acc and epoch > (args_copy.epochs - 40)):
                best_acc = acc
            
            state = {
                'acc': acc,
                'epoch': epoch,
                'state_dict': net.module.state_dict() if args_copy.ngpu > 1 else net.state_dict(),
            }
            opt_state = {
                'optimizer': optimizer.state_dict()
            }
            
            print(f'[{optimizer_name}] saving model epoch: ', epoch)
            torch.save(state, 'trained_nets/' + save_folder + '/model_' + str(epoch) + '.t7')
            torch.save(opt_state, 'trained_nets/' + save_folder + '/opt_state_' + str(epoch) + '.t7')
        
        # 学习率衰减
        if int(epoch) == 150 or int(epoch) == 225 or int(epoch) == 275:
            lr *= args_copy.lr_decay
            for param_group in optimizer.param_groups:
                param_group['lr'] *= args_copy.lr_decay
    
    # 关闭资源
    f.flush()  # 确保所有内容都写入文件
    f.close()
    writer.flush()
    writer.close()
    print("日志文件已正确关闭")

if __name__ == '__main__':
    # Training options
    parser = argparse.ArgumentParser(description='PyTorch STL10 Training')
    parser.add_argument('--dataset', default='stl10', type=str, choices=['stl10'], help='Dataset: stl10')
    parser.add_argument('--data_dir', default='~/dataset', type=str)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
    parser.add_argument('--lr_decay', default=0.1, type=float, help='learning rate decay rate')
    parser.add_argument('--optimizer', default='sgd', help='optimizer: sgd | sgd_momentum | adam | adamw | rmsprop | adagrad')
    parser.add_argument('--find_lr', action='store_true', help='Enable learning rate finder to automatically find optimal learning rate')
    parser.add_argument('--min_lr', type=float, default=1e-7, help='Minimum learning rate for finder')
    parser.add_argument('--max_lr', type=float, default=10.0, help='Maximum learning rate for finder')
    parser.add_argument('--lr_find_epochs', type=int, default=1, help='Number of epochs for learning rate finder')
    parser.add_argument('--compare_optimizers', action='store_true', help='Compare multiple optimizers in Tensorboard')
    parser.add_argument('--optimizer_list', default='sgd,adam', help='List of optimizers to compare, separated by commas')
    parser.add_argument('--weight_decay', default=0.0005, type=float)
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--epochs', default=300, type=int, metavar='N', help='number of total epochs to run')
    parser.add_argument('--save', default='trained_nets',help='path to save trained nets')
    parser.add_argument('--save_epoch', default=10, type=int, help='save every save_epochs')
    parser.add_argument('--ngpu', type=int, default=0, help='number of GPUs to use (0 for CPU)')
    parser.add_argument('--rand_seed', default=0, type=int, help='seed for random num generator')
    parser.add_argument('--resume_model', default='', help='resume model from checkpoint')
    parser.add_argument('--resume_opt', default='', help='resume optimizer from checkpoint')

    # model parameters
    parser.add_argument('--model', '-m', default='vgg11', choices=['vgg11', 'vgg16', 'vgg19', 'resnet18', 'resnet34'], help='Model: vgg11 | vgg16 | vgg19 | resnet18 | resnet34')
    parser.add_argument('--loss_name', '-l', default='crossentropy', help='loss functions: crossentropy | mse')

    # data parameters
    parser.add_argument('--raw_data', action='store_true', default=False, help='do not normalize data')
    parser.add_argument('--noaug', default=False, action='store_true', help='no data augmentation')
    parser.add_argument('--label_corrupt_prob', type=float, default=0.0)
    parser.add_argument('--trainloader', default='', help='path to the dataloader with random labels')
    parser.add_argument('--testloader', default='', help='path to the testloader with random labels')
    
    # 半监督学习参数
    parser.add_argument('--use_semisupervised', action='store_true', default=False, 
                        help='启用半监督学习')
    parser.add_argument('--unlabeled_ratio', type=float, default=0.8, 
                        help='对于CIFAR数据集，无标签数据的比例')
    parser.add_argument('--labeled_batch_size', type=int, default=64, 
                        help='有标签数据的批次大小')
    parser.add_argument('--unlabeled_batch_size', type=int, default=64, 
                        help='无标签数据的批次大小')
    parser.add_argument('--consistency_weight', type=float, default=1.0, 
                        help='一致性损失的权重')
    parser.add_argument('--temperature', type=float, default=0.4, 
                        help='伪标签温度参数')
    parser.add_argument('--confidence_threshold', type=float, default=0.95, 
                        help='伪标签置信度阈值')

    parser.add_argument('--idx', default=0, type=int, help='the index for the repeated experiment')


    args = parser.parse_args()

    model_loader.get_args(args)

    print('\nLearning Rate: %f' % args.lr)
    print('\nDecay Rate: %f' % args.lr_decay)

    use_cuda = torch.cuda.is_available()
    if use_cuda:
        print('Current devices: ' + str(torch.cuda.current_device()))
        print('Device count: ' + str(torch.cuda.device_count()))
    else:
        print('CUDA not available, using CPU')
    
    # 创建保存目录（如果不存在）
    if not os.path.isdir(args.save):
        os.mkdir(args.save)

    # Set the seed for reproducing the results
    random.seed(args.rand_seed)
    np.random.seed(args.rand_seed)
    torch.manual_seed(args.rand_seed)
    if use_cuda:
        torch.cuda.manual_seed_all(args.rand_seed)
        cudnn.benchmark = True

    lr = args.lr  # current learning rate
    start_epoch = 1  # start from epoch 1 or last checkpoint epoch

    if not os.path.isdir(args.save):
        os.mkdir(args.save)

    save_folder = name_save_folder(args)
    
    # 确保保存目录存在
    save_dir = os.path.join('trained_nets', save_folder)
    os.makedirs(save_dir, exist_ok=True)
    print(f"保存目录: {save_dir}")
    
    # 创建Tensorboard日志目录
    tb_log_dir = 'tensorboard_logs/' + save_folder
    os.makedirs(tb_log_dir, exist_ok=True)
    print(f"Tensorboard日志目录: {tb_log_dir}")
    
    # 初始化SummaryWriter
    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter(tb_log_dir)
    
    # 使用绝对路径确保日志文件位置正确
    log_path = os.path.join(save_dir, 'log.out')
    print(f"日志文件绝对路径: {os.path.abspath(log_path)}")
    
    # 直接使用with语句打开文件，确保正确关闭
    log_file_handle = open(log_path, 'a', buffering=1)  # 行缓冲模式
    print(f"日志文件已打开: {log_path}")
    
    # 写入初始日志信息
    initial_log = f"训练开始: 模型={args.model}, 数据集={args.dataset}, 学习率={args.lr}\n"
    print(f"写入初始日志: {initial_log.strip()}")
    log_file_handle.write(initial_log)
    log_file_handle.flush()

    # 根据是否使用半监督学习选择数据加载方式
    if args.use_semisupervised:
        print('使用半监督学习模式')
        labeled_loader, unlabeled_loader, testloader, transform_weak, transform_strong = \
            dataloader.get_semisupervised_data_loaders(args)
        print(f'有标签数据样本数: {len(labeled_loader.dataset)}')
        print(f'无标签数据样本数: {len(unlabeled_loader.dataset)}')
        print(f'测试数据样本数: {len(testloader.dataset)}')
    else:
        trainloader, testloader = dataloader.get_data_loaders(args)

    if args.label_corrupt_prob and not args.resume_model:
        torch.save(trainloader, 'trained_nets/' + save_folder + '/trainloader.dat')
        torch.save(testloader, 'trained_nets/' + save_folder + '/testloader.dat')

    # Model
    if args.resume_model:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        checkpoint = torch.load(args.resume_model)
        net = model_loader.load(args.model)
        net.load_state_dict(checkpoint['state_dict'])
        start_epoch = checkpoint['epoch'] + 1
    else:
        net = model_loader.load(args.model)
        print(net)
        init_params(net)

    if args.ngpu > 1:
        net = torch.nn.DataParallel(net)

    criterion = nn.CrossEntropyLoss()
    if args.loss_name == 'mse':
        criterion = nn.MSELoss()

    if use_cuda:
        net.cuda()
        criterion = criterion.cuda()

    # 判断是否需要比较多个优化器
    if args.compare_optimizers:
        print('\nComparing multiple optimizers: ' + args.optimizer_list)
        optimizers = [opt.strip() for opt in args.optimizer_list.split(',')]
        
        for opt_name in optimizers:
            print(f'\nTraining with {opt_name} optimizer...')
            train_with_optimizer(args, opt_name)
        
        print('\nAll optimizers training completed.')
        print('To visualize results in Tensorboard, run:')
        print('tensorboard --logdir=tensorboard_logs/comparison')
        
        # 绘制不同优化器的对比图
        plot_optimizer_comparison(args)
        
        # 直接退出，因为已经在train_with_optimizer中处理了所有逻辑
        import sys
        sys.exit(0)
    # 学习率查找模式
    if args.find_lr:
        print("启动学习率查找模式...")
        best_lr = find_lr(net, trainloader, optimizer, criterion, args)
        print(f"最佳学习率为: {best_lr:.2e}")
        print("学习率查找完成，请使用找到的学习率重新运行训练")
        import sys
        sys.exit(0)

    # record the performance of initial model
    if not args.resume_model:
        train_loss, train_err, train_recall, train_f1 = test(trainloader, net, criterion, use_cuda)
        test_loss, test_err, test_recall, test_f1 = test(testloader, net, criterion, use_cuda)
        status = 'e: %d loss: %.5f train_err: %.3f test_top1: %.3f test_loss %.5f test_recall: %.3f test_f1: %.3f \n' % (0, train_loss, train_err, test_err, test_loss, test_recall, test_f1)
        print(status)
        log_file_handle.write(status)
        log_file_handle.flush()

        # 记录初始状态到Tensorboard
        writer.add_scalar('Loss/train', train_loss, 0)
        writer.add_scalar('Loss/test', test_loss, 0)
        writer.add_scalar('Error/train', train_err, 0)
        writer.add_scalar('Error/test', test_err, 0)
        writer.add_scalar('Accuracy/test', 100 - test_err, 0)
        writer.add_scalar('Recall/train', train_recall, 0)
        writer.add_scalar('Recall/test', test_recall, 0)
        writer.add_scalar('F1/train', train_f1, 0)
        writer.add_scalar('F1/test', test_f1, 0)
        writer.add_scalar('Learning Rate', lr, 0)
        writer.flush()

        state = {
            'acc': 100 - test_err,
            'epoch': 0,
            'state_dict': net.module.state_dict() if args.ngpu > 1 else net.state_dict()
        }
        opt_state = {
            'optimizer': optimizer.state_dict()
        }
        torch.save(state, 'trained_nets/' + save_folder + '/model_0.t7')
        torch.save(opt_state, 'trained_nets/' + save_folder + '/opt_state_0.t7')
    
    best_acc = 0.0

    for epoch in range(start_epoch, args.epochs + 1):
        # 根据是否使用半监督学习选择不同的训练函数
        if args.use_semisupervised:
            # 半监督学习训练
            loss, sup_loss, unsup_loss, train_err, train_recall, train_f1 = \
                train_semisupervised(labeled_loader, unlabeled_loader, net, criterion, optimizer, 
                                    transform_weak, transform_strong, args, use_cuda)
        else:
            # 传统监督学习训练
            loss, train_err, train_recall, train_f1 = train(trainloader, net, criterion, optimizer, use_cuda)
            
        test_loss, test_err, test_recall, test_f1 = test(testloader, net, criterion, use_cuda)

        # 根据是否使用半监督学习输出不同的状态信息
        if args.use_semisupervised:
            status = 'e: %d loss: %.5f sup_loss: %.5f unsup_loss: %.5f train_err: %.3f test_top1: %.3f test_loss %.5f test_recall: %.3f test_f1: %.3f \n' % \
                (epoch, loss, sup_loss, unsup_loss, train_err, test_err, test_loss, test_recall, test_f1)
        else:
            status = 'e: %d loss: %.5f train_err: %.3f test_top1: %.3f test_loss %.5f test_recall: %.3f test_f1: %.3f \n' % (
            epoch, loss, train_err, test_err, test_loss, test_recall, test_f1)
        print(f"epoch {epoch} 状态: {status.strip()}")
        
        # 确保日志文件仍在打开状态
        try:
            log_file_handle.write(status)
            log_file_handle.flush()
            print(f"日志已写入文件: {os.path.abspath(log_path)}")
        except Exception as e:
            print(f"日志写入错误: {e}")
            # 尝试重新打开文件
            try:
                log_file_handle.close()
                log_file_handle = open(log_path, 'a', buffering=0)
                log_file_handle.write(status)
                log_file_handle.flush()
                print("重新打开文件并成功写入")
            except Exception as e2:
                print(f"重新打开文件也失败: {e2}")
        
        # 记录到Tensorboard
        writer.add_scalar('Loss/train', loss, epoch)
        writer.add_scalar('Loss/test', test_loss, epoch)
        writer.add_scalar('Error/train', train_err, epoch)
        writer.add_scalar('Error/test', test_err, epoch)
        writer.add_scalar('Accuracy/test', 100 - test_err, epoch)
        writer.add_scalar('Recall/train', train_recall, epoch)
        writer.add_scalar('Recall/test', test_recall, epoch)
        writer.add_scalar('F1/train', train_f1, epoch)
        writer.add_scalar('F1/test', test_f1, epoch)
        writer.add_scalar('Learning Rate', lr, epoch)

        # Save checkpoint.
        acc = 100 - test_err

        if epoch == 1 or epoch % args.save_epoch == 0 or epoch == 150 or (acc > best_acc and epoch > (args.epochs - 40)):
            if (acc > best_acc and epoch > (args.epochs - 40)):
                best_acc = acc

            state = {
                'acc': acc,
                'epoch': epoch,
                'state_dict': net.module.state_dict() if args.ngpu > 1 else net.state_dict(),
            }
            opt_state = {
                'optimizer': optimizer.state_dict()
            }

            print('savig model epoch: ', epoch)
            torch.save(state, 'trained_nets/' + save_folder + '/model_' + str(epoch) + '.t7')
            torch.save(opt_state, 'trained_nets/' + save_folder + '/opt_state_' + str(epoch) + '.t7')

        if int(epoch) == 150 or int(epoch) == 225 or int(epoch) == 275:
            lr *= args.lr_decay
            for param_group in optimizer.param_groups:
                param_group['lr'] *= args.lr_decay

    # 关闭资源
    print("正在关闭资源...")
    try:
        writer.flush()
        writer.close()
        print("Tensorboard writer已关闭")
    except Exception as e:
        print(f"关闭Tensorboard writer错误: {e}")
    
    try:
        log_file_handle.flush()
        log_file_handle.close()
        print(f"日志文件已关闭: {os.path.abspath(log_path)}")
    except Exception as e:
        print(f"关闭日志文件错误: {e}")
    
    # 生成训练结果图
    print("\n正在生成训练结果图...")
    
    # 直接使用当前的save_folder和日志文件路径来生成图表
    # 首先设置matplotlib后端，这必须在导入pyplot之前完成
    import matplotlib
    # 尝试不同的后端以确保兼容性，优先使用交互式后端
    interactive_backends = ['TkAgg', 'Qt5Agg', 'Qt4Agg', 'GTK3Agg', 'WXAgg']
    current_backend = None
    
    for backend in interactive_backends:
        try:
            matplotlib.use(backend)
            current_backend = backend
            print(f"已设置matplotlib后端为{backend}")
            break
        except Exception as e:
            print(f"无法使用{backend}后端：{e}")
    
    # 如果所有交互式后端都无法使用，再尝试非交互式后端
    if current_backend is None:
        try:
            matplotlib.use('Agg')
            current_backend = 'Agg'
            print("已设置matplotlib后端为Agg（非交互式）")
        except Exception as e:
            print(f"警告：无法设置matplotlib后端：{e}")
    
    # 输出当前使用的后端
    print(f"当前使用的matplotlib后端：{matplotlib.get_backend()}")
    
    import matplotlib.pyplot as plt
    import os
    import numpy as np
    import time
    
    # 设置中文字体支持
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['figure.figsize'] = (12, 6)
    
    # 确保save_dir是绝对路径
    save_dir = os.path.abspath(save_dir)
    print(f"绝对保存路径: {save_dir}")
    
    # 读取当前训练的日志文件
    log_path = os.path.join(save_dir, 'log.out')
    print(f"绘图代码中使用的日志文件路径: {log_path}")
    print(f"日志文件是否存在: {os.path.exists(log_path)}")
    
    if not os.path.exists(log_path):
        print(f"警告：日志文件 {log_path} 不存在，无法生成结果图")
    else:
        # 初始化数据列表
        epochs = []
        test_accuracies = []
        test_losses = []
        train_losses = []
        
        # 读取日志文件
        try:
            with open(log_path, 'r') as log_file:
                lines = log_file.readlines()
            print(f"成功读取日志文件，共 {len(lines)} 行")
        except Exception as e:
            print(f"读取日志文件 {log_path} 失败：{e}")
        else:
            # 解析日志文件
            import re
            # 修复正则表达式，test_loss后面没有冒号
            pattern = r'e: (\d+) loss: ([\d.]+) train_err: ([\d.]+) test_top1: ([\d.]+) test_loss ([\d.]+) test_recall: ([\d.]+) test_f1: ([\d.]+)'
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                
                # 跳过初始日志行
                if line.startswith('训练开始:'):
                    continue
                
                # 匹配日志格式
                match = re.match(pattern, line)
                if match:
                    epoch = int(match.group(1))
                    train_loss = float(match.group(2))
                    train_err = float(match.group(3))
                    test_err = float(match.group(4))
                    test_loss = float(match.group(5))
                    
                    # 计算准确率
                    test_accuracy = 100 - test_err
                    
                    # 保存数据
                    epochs.append(epoch)
                    test_accuracies.append(test_accuracy)
                    test_losses.append(test_loss)
                    train_losses.append(train_loss)
                    print(f"成功解析第 {epoch} 轮数据：test_accuracy={test_accuracy:.2f}, test_loss={test_loss:.4f}, train_loss={train_loss:.4f}")
                else:
                    print(f"无法匹配日志行：{line}")
        
        # 如果没有数据，跳过绘图
        if not epochs:
            print("警告：日志文件中没有有效的训练数据，无法生成结果图")
        else:
            # 创建保存图表的目录
            plot_dir = 'plots'
            os.makedirs(plot_dir, exist_ok=True)
            
            # 定义要绘制的指标
            metrics = {
                'test_accuracy': {'label': '测试准确率 (%)', 'y_lim': [0, 100], 'values': test_accuracies},
                'test_loss': {'label': '测试损失', 'y_lim': [0, 3], 'values': test_losses},
                'train_loss': {'label': '训练损失', 'y_lim': [0, 3], 'values': train_losses}
            }
            
            # 获取当前使用的优化器名称
            current_optimizer = args.optimizer
            
            # 绘制每个指标的图表
            for metric_name, metric_info in metrics.items():
                # 创建一个新的图表对象
                fig = plt.figure(figsize=(12, 6))
                
                # 绘制曲线
                plt.plot(epochs, metric_info['values'], 
                        label=current_optimizer, 
                        color='blue',
                        linewidth=2, 
                        marker='o', 
                        markersize=4)
                
                # 设置图表属性
                plt.title(f'{args.dataset} {args.model} {current_optimizer}优化器{metric_info["label"]}变化曲线', fontsize=14)
                plt.xlabel('训练轮次 (Epoch)', fontsize=12)
                plt.ylabel(metric_info['label'], fontsize=12)
                plt.ylim(metric_info['y_lim'])
                plt.grid(True, linestyle='--', alpha=0.7)
                plt.legend(fontsize=10)
                plt.tight_layout()
                
                # 保存图表
                plot_path = os.path.join(plot_dir, f'{args.dataset}_{args.model}_{current_optimizer}_{metric_name}_curve.png')
                plt.savefig(plot_path, dpi=300, bbox_inches='tight')
                print(f"图表已保存至: {plot_path}")
                
                # 显示图表
                # 如果使用的是非交互式后端，跳过显示
                if current_backend != 'Agg':
                    try:
                        print(f"正在显示图表：{metric_name}")
                        plt.show(block=True)  # 使用阻塞模式显示图片，等待用户关闭
                    except Exception as e:
                        print(f"显示图表失败：{e}")
                        print("图表已保存，但无法显示，可能是因为使用了非交互式后端")
                    finally:
                        # 关闭当前图表，释放内存
                        plt.close(fig)
                else:
                    print("当前使用非交互式后端，跳过图表显示")
                    # 关闭当前图表，释放内存
                    plt.close(fig)
            
            print("所有训练结果图绘制完成！")
