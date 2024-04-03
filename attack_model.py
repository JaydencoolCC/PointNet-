"""
Author: Benny
Date: Nov 2019
"""

import os
import sys
import torch
import numpy as np
import copy
import datetime
import logging
import provider
import importlib
import shutil
import argparse

from pathlib import Path
from tqdm import tqdm
from data_utils.ModelNetDataLoader import ModelNetDataLoader
from utils.dataset_tools import get_train_val_split, get_subsampled_dataset, print_attack_results, get_member_non_member_split

from attack import ThresholdAttack, SalemAttack, EntropyAttack, MetricAttack, KnnAttack
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))

def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('training')
    parser.add_argument('--use_cpu', action='store_true', default=False, help='use cpu mode')
    parser.add_argument('--gpu', type=str, default='1', help='specify gpu device')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size in training')
    parser.add_argument('--model', default='pointnet2_cls_ssg', help='model name [default: pointnet_cls]')
    parser.add_argument('--num_category', default=40, type=int, choices=[10, 40],  help='training on ModelNet10/40')
    parser.add_argument('--epoch', default=200, type=int, help='number of epoch in training')
    parser.add_argument('--learning_rate', default=0.001, type=float, help='learning rate in training')
    parser.add_argument('--num_point', type=int, default=1024, help='Point Number')
    parser.add_argument('--optimizer', type=str, default='Adam', help='optimizer for training')
    parser.add_argument('--log_dir', type=str, default='pointnet2_cls_ssg', help='experiment root')
    parser.add_argument('--decay_rate', type=float, default=1e-4, help='decay rate')
    parser.add_argument('--use_normals', action='store_true', default=False, help='use normals')
    parser.add_argument('--process_data', action='store_true', default=False, help='save data offline')
    parser.add_argument('--use_uniform_sample', action='store_true', default=False, help='use uniform sampiling')
    parser.add_argument('--train', action='store_true', default=False, help='train model')
    parser.add_argument('--seed', type=int, default=42, help='random seed')

    return parser.parse_args()


def inplace_relu(m):
    classname = m.__class__.__name__
    if classname.find('ReLU') != -1:
        m.inplace=True


def test(model, loader, num_class=40):
    mean_correct = []
    class_acc = np.zeros((num_class, 3))
    classifier = model.eval()

    for j, (points, target) in tqdm(enumerate(loader), total=len(loader)):

        if not args.use_cpu:
            points, target = points.cuda(), target.cuda()

        points = points.transpose(2, 1)
        pred, _ = classifier(points)
        pred_choice = pred.data.max(1)[1]

        for cat in np.unique(target.cpu()):
            classacc = pred_choice[target == cat].eq(target[target == cat].long().data).cpu().sum()
            class_acc[cat, 0] += classacc.item() / float(points[target == cat].size()[0])
            class_acc[cat, 1] += 1

        correct = pred_choice.eq(target.long().data).cpu().sum()
        mean_correct.append(correct.item() / float(points.size()[0]))

    class_acc[:, 2] = class_acc[:, 0] / class_acc[:, 1]
    class_acc = np.mean(class_acc[:, 2])
    instance_acc = np.mean(mean_correct)

    return instance_acc, class_acc


def main(args):
    def log_string(str):
        logger.info(str)
        print(str)

    '''HYPER PARAMETER'''
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    '''CREATE DIR'''
    timestr = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))
    exp_dir = Path('./log/')
    exp_dir.mkdir(exist_ok=True)
    exp_dir = exp_dir.joinpath('classification')
    exp_dir.mkdir(exist_ok=True)
    if args.log_dir is None:
        exp_dir = exp_dir.joinpath(timestr)
    else:
        exp_dir = exp_dir.joinpath(args.log_dir)
    exp_dir.mkdir(exist_ok=True)
    checkpoints_dir = exp_dir.joinpath('checkpoints/')
    checkpoints_dir.mkdir(exist_ok=True)
    log_dir = exp_dir.joinpath('logs/')
    log_dir.mkdir(exist_ok=True)

    '''LOG'''
    args = parse_args()
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/%s.txt' % (log_dir, args.model))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    log_string('PARAMETER ...')
    log_string(args)

    '''DATA LOADING'''
    log_string('Load dataset ...')
    data_path = '/mnt/sharedata/ssd/users/zhanghx/dataset/modelnet40_normal_resampled'

    train_dataset = ModelNetDataLoader(root=data_path, args=args, split='train', process_data=args.process_data)
    test_dataset = ModelNetDataLoader(root=data_path, args=args, split='test', process_data=args.process_data)
    
    train_size = 4400
    test_size = 1000
    train_dataset = get_subsampled_dataset(train_dataset, dataset_size=train_size*2, proportion=None)
    train_target, train_shadow = get_train_val_split(train_dataset, train_size, seed=args.seed, stratify=False, targets=None)
    
    test_dataset = get_subsampled_dataset(test_dataset, dataset_size=test_size*2, proportion=0.5)
    test_target, test_shadow= get_train_val_split(test_dataset, test_size, seed=args.seed, stratify=False, targets=None)
    
    trainDataLoader_target = torch.utils.data.DataLoader(train_target, batch_size=args.batch_size, shuffle=True, num_workers=10, drop_last=True)
    trainDataLoader_shadow = torch.utils.data.DataLoader(train_shadow, batch_size=args.batch_size, shuffle=True, num_workers=10, drop_last=True)
    
    testDataLoade_target = torch.utils.data.DataLoader(test_target, batch_size=args.batch_size, shuffle=False, num_workers=10)
    testDataLoade_shadow = torch.utils.data.DataLoader(test_shadow, batch_size=args.batch_size, shuffle=False, num_workers=10)
    
    '''MODEL LOADING'''
    num_class = args.num_category
    model = importlib.import_module(args.model)
    shutil.copy('./models/%s.py' % args.model, str(exp_dir))
    shutil.copy('models/pointnet2_utils.py', str(exp_dir))
    shutil.copy('./train_classification.py', str(exp_dir))
    
    classifier = model.get_model(num_class, normal_channel=args.use_normals)
    criterion = model.get_loss()
    classifier.apply(inplace_relu)

    target_model = copy.deepcopy(classifier)
    shadow_model = copy.deepcopy(classifier)

    
    if not args.use_cpu:
        classifier = classifier.cuda()
        criterion = criterion.cuda()
        target_model = target_model.cuda()
        shadow_model = shadow_model.cuda()

    try: 
        if not args.train:
            checkpoint_target = torch.load(os.path.join(str(checkpoints_dir), "target" + "_epoch_%d" %(args.epoch) + '_no_augmentation' + '_model.pth'))
            checkpoint_shadow = torch.load(os.path.join(str(checkpoints_dir), "shadow" + "_epoch_%d" %(args.epoch) + '_no_augmentation' + '_model.pth'))

            target_model.load_state_dict(checkpoint_target['model_state_dict'])
            shadow_model.load_state_dict(checkpoint_shadow['model_state_dict'])

            start_epoch = checkpoint_target['epoch']

            log_string('Use pretrain model')
        else:
            log_string('No existing model, starting training from scratch...')
            start_epoch = 0
    except:
        log_string('No existing model, starting training from scratch...')
        start_epoch = 0

    if args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(
            classifier.parameters(),
            lr=args.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=args.decay_rate
        )
    else:
        optimizer = torch.optim.SGD(classifier.parameters(), lr=0.01, momentum=0.9)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.7)
    
    if(True):
        #test for MIA
        #generate dataset
        member_target, non_member_target = get_member_non_member_split(train_target, test_target, 1000)
        member_shadow, non_member_shadow = get_member_non_member_split(train_shadow, test_shadow, 1000)
    
        # with torch.no_grad():
        #     instance_acc, class_acc = test(target_model.eval(), trainDataLoader_target, num_class=num_class)
        #     print('Train Instance Accuracy: %f, Class Accuracy: %f' % (instance_acc, class_acc))
        #     instance_acc, class_acc = test(target_model.eval(), testDataLoade_target, num_class=num_class)
        #     print('Test Instance Accuracy: %f, Class Accuracy: %f' % (instance_acc, class_acc))
        
            
        attacks = [
            #ThresholdAttack(apply_softmax= False),
            #SalemAttack(apply_softmax= False, k=3),
            #EntropyAttack(apply_softmax= False),
            MetricAttack(apply_softmax= False)
            #KnnAttack(apply_softmax= False, batch_size = args.batch_size)
        ]
        name = ["MetricAttack"]
        attack_list = []
        for i in range(len(attacks)):
            attack = attacks[i]
            attack.learn_attack_parameters(shadow_model, member_shadow, non_member_shadow)
            result = attack.evaluate(target_model, member_target, non_member_target)
            attack_list.append(result)
            print_attack_results(name[i], result)    
        
        # Attack = KnnAttack(apply_softmax= False)
        # #Attack.attack(target_model, member_target, non_member_target)
        # Attack.learn_attack_parameters(shadow_model, member_shadow, non_member_shadow)
        
def train(classifier, start_epoch, logger, log_string, trainDataLoader, criterion, testDataLoader, checkpoints_dir, num_class=40, train_type="target"):
    
    global_epoch = 0
    global_step = 0
    best_instance_acc = 0.0
    best_class_acc = 0.0
    '''TRANING'''
    
    logger.info('Start training %s model' % (train_type))
    if args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(
            classifier.parameters(),
            lr=args.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=args.decay_rate
        )
    else:
        optimizer = torch.optim.SGD(classifier.parameters(), lr=0.01, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.7)
    
        
    for epoch in range(start_epoch, args.epoch):
        log_string('Epoch %d (%d/%s):' % (global_epoch + 1, epoch + 1, args.epoch))
        mean_correct = []
        classifier = classifier.train()

        scheduler.step()
        for batch_id, (points, target) in tqdm(enumerate(trainDataLoader, 0), total=len(trainDataLoader), smoothing=0.9):
            optimizer.zero_grad()

            points = points.data.numpy()
            points = provider.random_point_dropout(points)
            points[:, :, 0:3] = provider.random_scale_point_cloud(points[:, :, 0:3])
            points[:, :, 0:3] = provider.shift_point_cloud(points[:, :, 0:3])
            points = torch.Tensor(points)
            points = points.transpose(2, 1)

            if not args.use_cpu:
                points, target = points.cuda(), target.cuda()

            pred, trans_feat = classifier(points)
            loss = criterion(pred, target.long(), trans_feat)
            pred_choice = pred.data.max(1)[1]

            correct = pred_choice.eq(target.long().data).cpu().sum()
            mean_correct.append(correct.item() / float(points.size()[0]))
            loss.backward()
            optimizer.step()
            global_step += 1

        train_instance_acc = np.mean(mean_correct)
        log_string('Train Instance Accuracy: %f' % train_instance_acc)

        with torch.no_grad():

            instance_acc, class_acc = test(classifier.eval(), testDataLoader, num_class=num_class)

            if (instance_acc >= best_instance_acc):
                best_instance_acc = instance_acc
                best_epoch = epoch + 1

            if (class_acc >= best_class_acc):
                best_class_acc = class_acc
            log_string('Test Instance Accuracy: %f, Class Accuracy: %f' % (instance_acc, class_acc))
            #log_string('Best Instance Accuracy: %f, Class Accuracy: %f' % (best_instance_acc, best_class_acc))

            if (epoch == args.epoch - 1):
                logger.info('Save model...')
                addtion = "_no_augmentation"
                
                savepath = os.path.join(str(checkpoints_dir), train_type + "_epoch_%d" %(epoch+1) + addtion + '_model.pth')
                log_string('Saving at %s' % savepath)
                state = {
                    'epoch': epoch+1,
                    'instance_acc': instance_acc,
                    'class_acc': class_acc,
                    'model_state_dict': classifier.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }
                torch.save(state, savepath)
            global_epoch += 1
    logger.info('End of training...')
     

        
if __name__ == '__main__':
    args = parse_args()
    main(args)
