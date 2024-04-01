import numpy as np
import torch


import torch.nn.functional as F

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
    