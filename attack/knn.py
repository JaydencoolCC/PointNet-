import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.distributions import Categorical
import numpy as np
from utils.roc import get_roc

from attack import PredictionScoreAttack
from .attack_utils import cross_entropy, write_to_csv

from models.pointnet2_utils import farthest_point_sample, index_points

class KnnAttack(PredictionScoreAttack):
    def __init__(self, apply_softmax: bool, batch_size: int = 128, log_training: bool = True, npoint = 1024, k=5):
        super().__init__('Entropy Attack')

        self.batch_size = batch_size
        self.theta = 0.0
        self.apply_softmax = apply_softmax
        self.log_training = log_training
        self.npoint = npoint
        self.k = k
    def learn_attack_parameters(
        self, shadow_model: nn.Module, member_dataset: torch.utils.data.Dataset, non_member_dataset: Dataset
    ):
        # Gather entropy of predictions by shadow model
        shadow_model.to(self.device)
        shadow_model.eval()
        scores = []
        membership_labels = []
        if self.log_training:
            print('Compute attack model dataset')
        with torch.no_grad():
            shadow_model.eval()
            for indx, dataset in enumerate([non_member_dataset, member_dataset]):
                loader = DataLoader(dataset, batch_size=self.batch_size, num_workers=4)
                for x, y in loader:
                    x, y = x.to(self.device), y.to(self.device) # x: [b,n,c]
                    points = self.generate_data(x, k=self.k, npoint=self.npoint) # b k npoint c
                    points = points.transpose(2,3)
                    for i in range(points.shape[0]):
                        x = points[i,:,:,:]
                        output,_ = shadow_model(x)
                        if self.apply_softmax:
                            prediction_scores = torch.softmax(output, dim=1) # k, classes
                        else:
                            prediction_scores = output
                    
                        scores.append(prediction_scores.cpu()) # k, classes len = n
                    membership_labels.append(torch.full_like(y, indx)) #member = 1, non-member = 0 
        membership_labels = torch.cat(membership_labels, dim=0).cpu() # (n, 1)
        #scores = torch.stack(scores) #n k classes
        
        scoers_range = []
        for val in scores:
            max_val = torch.max(val)
            min_val = torch.min(val)
            scoers_range.append(max_val - min_val) #n
        
        
        scoers_range = np.array(scoers_range)
        membership_labels = np.array(membership_labels)
        
        # Compute threshold
        theta_best = 0.0
        num_corrects_best = 0
        for theta in np.linspace(min(scoers_range), max(scoers_range), 10000):
            num_corrects = (scoers_range[membership_labels == 1] >=
                            theta).sum() + (scoers_range[membership_labels == 0] < theta).sum()
            if num_corrects > num_corrects_best:
                num_corrects_best = num_corrects
                theta_best = theta
        self.attack_treshold = theta_best
        if self.log_training:
            print(
                f'Theta set to {self.attack_treshold} achieving {num_corrects_best / (len(member_dataset) + len(non_member_dataset))}'
            )
        
        # membership_labels = membership_labels.tolist()
        # self.shadow_fpr, self.shadow_tpr, self.thresholds, self.auroc = get_roc(membership_labels, scoers_range)
        # threshold_idx = (self.shadow_tpr - self.shadow_fpr).argmax()
        # self.attack_treshold = self.thresholds[threshold_idx]
        

        scoers_range_member = scoers_range[membership_labels==1]
        scoers_range_nonmeber = scoers_range[membership_labels==0]
        
        
        print("Threshold: ", self.attack_treshold)
        nums_member = sum(scoers_range_member > self.attack_treshold)
        nums_non_member = sum(scoers_range_nonmeber < self.attack_treshold)
        print("shadow number of members: ", nums_member)
        print("shadow number of non members: ", nums_non_member)

    def predict_membership(self, target_model: nn.Module, dataset: Dataset):
        values = []
        target_model.eval()
        with torch.no_grad():
            for x, y in DataLoader(dataset, batch_size=self.batch_size, num_workers=4):
                x, y = x.to(self.device), y.to(self.device)
                points = self.generate_data(x, k=self.k, npoint=self.npoint) # b k npoint c
                points = points.transpose(2, 3)
                for i in range(points.shape[0]):
                    x = points[i,:,:,:]
                    output, _ = target_model(x)
                    if self.apply_softmax:
                        pred_scores = torch.softmax(output, dim=1)
                    else:
                        pred_scores = output
                
                    max_val = torch.max(pred_scores)
                    min_val = torch.min(pred_scores)
                    values.append(max_val - min_val)
                
        values = torch.tensor(values).cpu()
        nums_member = sum(values > self.attack_treshold).item()
        nums_non_member = sum(values < self.attack_treshold).item()
        # print("predict number of members: ", nums_member)
        # print("predict number of non members: ", nums_non_member)
        
        return values > self.attack_treshold

    def get_attack_model_prediction_scores(self, target_model: nn.Module, dataset: Dataset) -> torch.Tensor:
        values = []
        target_model.eval()
        with torch.no_grad():
            for x, y in DataLoader(dataset, batch_size=self.batch_size):
                x, y = x.to(self.device), y.to(self.device)
                points = self.generate_data(x, k=self.k, npoint=self.npoint)
                points = points.transpose(2, 3)
                for i in range(points.shape[0]):
                    x = points[i,:,:,:]
                    output, _ = target_model(x)
                    if self.apply_softmax:
                        pred_scores = torch.softmax(output, dim=1)
                    else:
                        pred_scores = output
                    max_val = torch.max(pred_scores)
                    min_val = torch.min(pred_scores)
                    values.append(max_val - min_val)
                
        values = torch.tensor(values)
        return values.cpu()
    
    def farthest_point_sample(xyz, npoint):
        """
        Input:
            xyz: pointcloud data, [B, N, 3]
            npoint: number of samples
        Return:
            centroids: sampled pointcloud index, [B, npoint]
        """
        device = xyz.device
        B, N, C = xyz.shape
        centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
        distance = torch.ones(B, N).to(device) * 1e10
        farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
        batch_indices = torch.arange(B, dtype=torch.long).to(device)
        for i in range(npoint):
            centroids[:, i] = farthest
            centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
            dist = torch.sum((xyz - centroid) ** 2, -1)
            mask = dist < distance
            distance[mask] = dist[mask]
            farthest = torch.max(distance, -1)[1]
        return centroids
    
    def generate_data(self, xyz, npoint=1024, k=5):
        """generate data

        Args:
            xyz: pointcloud data, [B, N, 3]
            npoint: number of samples
            k (int, optional): k. Defaults to 5.

        Returns:
            points: _description_
        """
        data = []
        for i in range(k):
            fps_idx = farthest_point_sample(xyz, npoint) # [B, npoint, C]
            new_xyz = index_points(xyz, fps_idx)
            data.append(new_xyz) 
        data = torch.stack(data, dim=1) #[B, k, npoint, C]
        return data
        
    
    def attack(
        self, shadow_model: nn.Module, member_dataset: torch.utils.data.Dataset, non_member_dataset: Dataset):
        # Gather entropy of predictions by shadow model
        shadow_model.to(self.device)
        shadow_model.eval()
        scores = []
        membership_labels = []
        labels = []
        if self.log_training:
            print('Compute attack model dataset')
        with torch.no_grad():
            shadow_model.eval()
            for idx, dataset in enumerate([non_member_dataset, member_dataset]):
                loader = DataLoader(dataset, batch_size=self.batch_size, num_workers=4)
                for x, y in loader:
                    x, y = x.to(self.device), y.to(self.device) # x: [b,n,c]
                    points = self.generate_data(x, k=self.k, npoint=self.npoint) # b k n c
                    points = points.transpose(2,3)
                    for i in range(points.shape[0]):
                        x = points[i,:,:,:]
                        output,_ = shadow_model(x)
                        if self.apply_softmax:
                            prediction_scores = torch.softmax(output, dim=1) # k, classes
                        else:
                            prediction_scores = output
                    
                        scores.append(prediction_scores) # (k, classes), len = n
                    labels.append(y)
                    membership_labels.append(torch.full_like(y, idx)) #member = 1, non-member = 0
                    
        membership_labels = torch.cat(membership_labels, dim=0)
        labels = torch.cat(labels, dim=0).cpu().numpy()
        labels_total = []
        membership_labels_total = []
        loss = []
        print("scores length", len(scores))
        print("shape: ", scores[0].shape)
        for i in range(len(scores)):
            device = scores[i].device
            each_scores = scores[i] #k, classes
            each_labels = torch.full((each_scores.shape[0], ), labels[i]).to(device)
            each_loss = cross_entropy(each_scores, each_labels)
            mem_lable = torch.full((each_scores.shape[0], ), membership_labels[i])
            loss.append(each_loss)
            labels_total.append(each_labels)
            membership_labels_total.append(mem_lable)
            
        loss_values = torch.cat(loss, dim=0).cpu().numpy() #n*k
        scores = torch.cat(scores, dim=0).cpu().numpy() #n*k, classes
        labels_total = torch.cat(labels_total, dim=0).cpu().numpy() #n*k
        membership_labels_total = torch.cat(membership_labels_total, dim=0).cpu().numpy()
        
        print("loss_values ", loss_values.shape)
        print("scores shape", scores.shape)
        print("labels_total shape", labels_total.shape)
        print("membership_labels_total shape", membership_labels_total.shape)
        
        # write
        write_to_csv(scores, membership_labels_total, labels_total, loss_values)