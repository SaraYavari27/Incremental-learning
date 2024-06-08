
from __future__ import absolute_import

import torch
from torch import nn
from torch.autograd import Variable


class TripletLoss(nn.Module):
    def __init__(self, margin=0, num_instances=None):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)

    def forward(self, inputs, targets, mean_neg, mean_label):

        targets = torch.cat((targets, mean_label), dim=0)
        inputs = torch.cat((inputs, mean_neg), dim=0)
        n = inputs.size(0)
        m = mean_neg.size(0)
        # Compute pairwise distance, replace by the official when merged
        dist = torch.pow(inputs, 2).sum(dim=1, keepdim=True).expand(n, n)
        dist = dist + dist.t()
        dist.addmm_(inputs, inputs.t(), beta=1, alpha=-2)
        dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
        # For each anchor, find the hardest positive and negative
        mask = targets.expand(n, n).eq(targets.expand(n, n).t())
        dist_ap, dist_an = [], []
        # dist_np = dist.cpu().detach().numpy()
        for i in range(n):
            dist_ap.append(dist[i][mask[i]].max())
            # p1 = [mask[i] == 0]
            # p2 = dist[i][p1]
            # p3 = dist[i][mask[i] == 0]
            # p4 = dist[i][mask[i] == 0].min()
            dist_an.append(dist[i][mask[i] == 0].min())
        dist_ap = torch.stack(dist_ap)
        dist_an = torch.stack(dist_an)
        # Compute ranking hinge loss
        y = dist_an.data.new()
        y.resize_as_(dist_an.data)
        y.fill_(1)
        y = Variable(y)
        loss = self.ranking_loss(dist_an, dist_ap, y)
        prec = (dist_an.data > dist_ap.data).sum() * 1. / y.size(0)
        dist_p = torch.mean(dist_ap).data
        dist_n = torch.mean(dist_an).data
        return loss, prec, dist_p, dist_n
