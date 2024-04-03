import numpy as np
import torch
import torch.nn.functional as F

class AttackResult:
    attack_acc: float
    precision: float
    recall: float
    tpr: float
    tnr: float
    fpr: float
    fnr: float
    tp_mmps: float
    fp_mmps: float
    fn_mmps: float
    tn_mmps: float

    def __init__(
        self,
        attack_acc: float,
        precision: float,
        recall: float,
        auroc: float,
        aupr: float,
        fpr_at_tpr95: float,
        tpr: float,
        tnr: float,
        fpr: float,
        fnr: float,
        tp_mmps: float,
        fp_mmps: float,
        fn_mmps: float,
        tn_mmps: float
    ):
        self.attack_acc = attack_acc
        self.precision = precision
        self.recall = recall
        self.auroc = auroc
        self.aupr = aupr
        self.fpr_at_tpr95 = fpr_at_tpr95
        self.tpr = tpr
        self.tnr = tnr
        self.fpr = fpr
        self.fnr = fnr
        self.tp_mmps = tp_mmps
        self.fp_mmps = fp_mmps
        self.fn_mmps = fn_mmps
        self.tn_mmps = tn_mmps


def cross_entropy(prob, label):
    epsilon = 1e-12
    prob = torch.clamp(prob, epsilon, 1.0 - epsilon) #  lower bound the probability to avoid log(0)
    one_hot_label = torch.zeros_like(prob)
    label = label.long()  # ensure the label is int64
    one_hot_label.scatter_(1, label.unsqueeze(1), 1)
    return -torch.sum(one_hot_label * torch.log(prob), dim=1)

def write_to_csv(confidence, membership_labels, labels, value):
    import csv
    with open('confidence.csv', mode='w') as file:
        writer = csv.writer(file)
        writer.writerow(['confidence', 'membership_labels', 'labels', 'value'])
        for i in range(len(confidence)):
            row = [val for val in confidence[i]]
            row.append(membership_labels[i])
            row.append(labels[i])
            row.append(value[i])
            writer.writerow(row)
    