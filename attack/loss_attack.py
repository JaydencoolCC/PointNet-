import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.distributions import Categorical
import numpy as np
from utils.roc import get_roc

from attack import PredictionScoreAttack
from .attack_utils import cross_entropy

class MetricAttack(PredictionScoreAttack):
    def __init__(self, apply_softmax: bool, batch_size: int = 128, log_training: bool = True):
        super().__init__('Entropy Attack')

        self.batch_size = batch_size
        self.theta = 0.0
        self.apply_softmax = apply_softmax
        self.log_training = log_training

    def learn_attack_parameters(
        self, shadow_model: nn.Module, member_dataset: torch.utils.data.Dataset, non_member_dataset: Dataset
    ):
        # Gather entropy of predictions by shadow model
        shadow_model.to(self.device)
        shadow_model.eval()
        values = []
        membership_labels = []
        if self.log_training:
            print('Compute attack model dataset')
        with torch.no_grad():
            shadow_model.eval()
            for i, dataset in enumerate([non_member_dataset, member_dataset]):
                loader = DataLoader(dataset, batch_size=self.batch_size, num_workers=4)
                for x, y in loader:
                    x, y = x.to(self.device), y.to(self.device)
                    x = x.transpose(2, 1)
                    output,_ = shadow_model(x)
                    if self.apply_softmax:
                        prediction_scores = torch.softmax(output, dim=1)
                    else:
                        prediction_scores = output
                    ce_loss = cross_entropy(prediction_scores, y)
                    values.append(ce_loss)
                    membership_labels.append(torch.full_like(y, i)) #member = 1, non-member = 0

        loss_values = torch.cat(values, dim=0).cpu().numpy()
        membership_labels = torch.cat(membership_labels, dim=0).cpu().numpy()

        # Compute threshold
        theta_best = 0.0
        num_corrects_best = 0
        for theta in np.linspace(min(loss_values), max(loss_values), 10000):
            num_corrects = (loss_values[membership_labels == 0] >=
                            theta).sum() + (loss_values[membership_labels == 1] < theta).sum()
            if num_corrects > num_corrects_best:
                num_corrects_best = num_corrects
                theta_best = theta
        self.theta = theta_best
        if self.log_training:
            print(
                f'Theta set to {self.theta} achieving {num_corrects_best / (len(member_dataset) + len(non_member_dataset))}'
            )
        
        #compute the threshold
        # self.shadow_fpr, self.shadow_tpr, self.thresholds, self.auroc = get_roc(membership_labels, loss_values)
        # threshold_idx = (self.shadow_tpr - self.shadow_fpr).argmax()
        # self.theta = self.thresholds[threshold_idx]
        # print(
        #          f'Theta set to {self.theta}'
        #      )
        
    def predict_membership(self, target_model: nn.Module, dataset: Dataset):
        values = []
        target_model.eval()
        with torch.no_grad():
            for x, y in DataLoader(dataset, batch_size=self.batch_size, num_workers=4):
                x, y = x.to(self.device), y.to(self.device)
                x = x.transpose(2, 1)
                output, _ = target_model(x)
                if self.apply_softmax:
                    pred_scores = torch.softmax(output, dim=1)
                else:
                    pred_scores = output
                ce_loss = cross_entropy(pred_scores, y)
                values.append(ce_loss)
        values = torch.cat(values, dim=0)
        return (values < self.theta).cpu().numpy()

    def get_attack_model_prediction_scores(self, target_model: nn.Module, dataset: Dataset) -> torch.Tensor:
        values = []
        target_model.eval()
        with torch.no_grad():
            for x, y in DataLoader(dataset, batch_size=self.batch_size):
                x, y = x.to(self.device), y.to(self.device)
                x = x.transpose(2, 1)
                output, _ = target_model(x)
                if self.apply_softmax:
                    pred_scores = torch.softmax(output, dim=1)
                else:
                    pred_scores = output
                ce_loss = cross_entropy(pred_scores, y)
                values.append(ce_loss)
        values = torch.cat(values, dim=0)
        return values.cpu()
    
    def get_metric_method(self, name: str ):
        if name == "Entropy":
            return self.get_attack_model_prediction_scores
        elif name == "loss":
            return self.predict_membership
        elif name == "score":
            return self.get_attack_model_prediction_scores
        return "Entropy"
        #TODO complete the method
        
    def get_loss_threshold(self,):
        return self.theta
    
    