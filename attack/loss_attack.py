import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.distributions import Categorical
import numpy as np
from torchmetrics.functional import roc, auroc, precision_recall_curve
from torchmetrics.utilities.compute import auc
from .attack_utils import AttackResult
from attack import PredictionScoreAttack
from .attack_utils import cross_entropy

class MetricAttack(PredictionScoreAttack):
    def __init__(self, apply_softmax: bool, batch_size: int = 128, log_training: bool = True, metric_method = "Entropy"):
        super().__init__('Entropy Attack')
        self.metric = self.get_metric_method(metric_method)
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
        
    def predict_membership(self, target_model: nn.Module, dataset: Dataset):
        values = []
        scores = []
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
                scores.append(pred_scores.cpu()) #unit class_num
        values = torch.cat(values, dim=0) #torch
        scores = torch.cat(scores) if len(scores) > 0 else torch.empty(0)
        return (values < self.theta).cpu().numpy(), scores, values.cpu()
    
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
    
    def evaluate(self, target_model: nn.Module, member_dataset: Dataset, non_member_dataset: Dataset) -> AttackResult:
        """
        Evaluates the attack by predicting the membership for the member dataset as well as the non-member dataset.
        Returns a `AttackResult`-object.
        :param target_model: The given target model
        :param member_dataset: The member dataset that was used to train the target model.
        :param non_member_dataset: The non-member dataset that was **not** used to train the target model.
        :param kwargs: Additional optional parameters for the `predict_membership`-method.
        """
        member_predictions, member_pred_scores, member_values= self.predict_membership(target_model, member_dataset) 
        non_member_predictions, non_member_pred_scores, non_member_values= self.predict_membership(target_model, non_member_dataset)
        tp = member_predictions.sum()
        tn = len(non_member_dataset) - non_member_predictions.sum()
        fp = non_member_predictions.sum()
        fn = len(member_dataset) - member_predictions.sum()
        tpr = tp / len(member_dataset)
        tnr = tn / len(non_member_dataset)
        fpr = 1 - tnr
        fnr = 1 - tpr
        pre = tp / (tp + fp) if tp != 0 else 0.0
        rec = tp / (tp + fn)
        acc = (tp + tn) / (tp + tn + fp + fn)
        
        concat_preds = torch.cat((non_member_values, member_values))
        concat_targets = torch.tensor([0 for _ in non_member_values] + [1 for _ in member_values])

        # get the auroc, aupr and FPR@95%TPR
        auroc_value: float = auroc(
            preds=concat_preds,
            target=concat_targets,
            task="binary"
        ).item()
        pr_precision, pr_recall, pr_thresholds = precision_recall_curve(
            preds=concat_preds,
            target=concat_targets,
            task="binary"
        )
        aupr_value: float = auc(x=pr_recall, y=pr_precision, reorder=True).item()
        tm_fpr, tm_tpr, tm_thresholds = roc(preds=concat_preds, target=concat_targets,task="binary")
        tpr_greater95_indices = np.where(tm_tpr >= 0.95)[0]
        fpr_at_tpr95 = tm_fpr[tpr_greater95_indices[0]].item()

        # get the mmps values
        tp_pred_scores = self.get_pred_score_classified_as_members(
            member_predictions, member_pred_scores
        )
        tp_mmps = tp_pred_scores.max(dim=1)[0].mean() if len(tp_pred_scores) > 0 else 0

        fp_pred_scores = self.get_pred_score_classified_as_members(
            non_member_predictions, non_member_pred_scores
        )
        fp_mmps = fp_pred_scores.max(dim=1)[0].mean() if len(fp_pred_scores) > 0 else 0

        fn_pred_scores = self.get_pred_score_classified_as_non_members(
            member_predictions, member_pred_scores
        )
        fn_mmps = fn_pred_scores.max(dim=1)[0].mean() if len(fn_pred_scores) > 0 else 0

        tn_pred_scores = self.get_pred_score_classified_as_non_members(
            non_member_predictions,  non_member_pred_scores
        )
        tn_mmps = tn_pred_scores.max(dim=1)[0].mean() if len(tn_pred_scores) > 0 else 0

        result = AttackResult(
            attack_acc=acc,
            precision=pre,
            recall=rec,
            auroc=auroc_value,
            aupr=aupr_value,
            fpr_at_tpr95=fpr_at_tpr95,
            tpr=tpr,
            tnr=tnr,
            fpr=fpr,
            fnr=fnr,
            tp_mmps=tp_mmps,
            fp_mmps=fp_mmps,
            fn_mmps=fn_mmps,
            tn_mmps=tn_mmps
        )
        return result

    def get_pred_score_classified_as_members(self, membership_predictions, member_pred_scores):
        predicted_member_indices = torch.nonzero(torch.tensor(membership_predictions).squeeze()).squeeze() #get index that label is member
        if predicted_member_indices.ndim == 0:
            predicted_member_indices = predicted_member_indices.unsqueeze(0)    
        prediction_scores = []
        for idx in predicted_member_indices:
            prediction_scores.append(member_pred_scores[idx])
        
        prediction_scores = torch.stack(prediction_scores) if len(prediction_scores) > 0 else torch.empty(0)
        return prediction_scores
    
    def get_pred_score_classified_as_non_members(self, non_membership_predictions, non_member_pred_scores):
        # logical_not = not
        predicted_member_indices = torch.nonzero(torch.logical_not(torch.tensor(non_membership_predictions)).squeeze() 
                                                 ).squeeze()
        if predicted_member_indices.ndim == 0:
            predicted_member_indices = predicted_member_indices.unsqueeze(0)
        prediction_scores = []
        for idx in predicted_member_indices:
            prediction_scores.append(non_member_pred_scores[idx])

        prediction_scores = torch.stack(prediction_scores) if len(prediction_scores) > 0 else torch.empty(0)
        return prediction_scores
    
        """@return: pre_scores, labels: torch.Tensor, torch.Tensor
        """
        
    def get_metric_method(self, name: str):
        if name == "Entropy":
            #return Categorical(probs=prediction_scores).entropy()
            pass
        elif name == "loss":
            #return cross_entropy(prediction_scores, labels)
            pass
        elif name == "max_score":
            return self.get_attack_model_prediction_scores
        else:
            raise ValueError("Not know function")
        
    def get_loss_threshold(self,):
        return self.theta
    
    