import torch
import os
import net
import torch.nn as nn


def load_net(dataset, net_type, nb_classes):

    if net_type == 'bn_inception':
        model = net.bn_inception(pretrained=True)
        model.last_linear = nn.Linear(1024, nb_classes)
        model.load_state_dict(torch.load('net/finetuned_' + dataset + '_' + net_type + '.pth'))

    elif net_type == 'densenet121':
        model = net.densenet121(pretrained=True)
        model.classifier = nn.Linear(1024, nb_classes)
        model.load_state_dict(torch.load('net/finetuned_' + dataset + '_' + net_type + '.pth'))

    elif net_type == 'densenet161':
        model = net.densenet161(pretrained=True)
        model.classifier = nn.Linear(2208, nb_classes)
        model.load_state_dict(torch.load('net/finetuned_' + dataset + '_' + net_type + '.pth'))

    elif net_type == 'densenet169':
        model = net.densenet169(pretrained=True)
        model.classifier = nn.Linear(1664, nb_classes)
        model.load_state_dict(torch.load('net/finetuned_' + dataset + '_' + net_type + '.pth'))

    elif net_type == 'densenet201':
        model = net.densenet201(pretrained=True)
        model.classifier = nn.Linear(1920, nb_classes)
        model.load_state_dict(torch.load('net/finetuned_' + dataset + '_' + net_type + '.pth'))
    return model

