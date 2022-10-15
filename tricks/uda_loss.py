import torch
from torch import nn
from torch.nn import functional as F
import math


class UDALoss(nn.Module):
    '''UDALoss，使用时候需要继承一下，因为forward需要使用到global_step和total_steps
    https://arxiv.org/abs/1904.12848
    '''

    def __init__(self, tsa_schedule=None, total_steps=None, start_p=0, end_p=1, return_all_loss=True):
        super().__init__()
        self.loss_sup = nn.CrossEntropyLoss()
        self.loss_unsup = nn.KLDivLoss(reduction='batchmean')
        self.tsa_schedule = tsa_schedule
        self.start = start_p
        self.end = end_p
        if self.tsa_schedule:
            assert self.tsa_schedule in {'linear_schedule', 'exp_schedule',
                                         'log_schedule'}, 'tsa_schedule config illegal'
        self.return_all_loss = return_all_loss

    def forward(self, y_pred, y_true_sup, global_step, total_steps):
        sup_size = y_true_sup.size(0)
        unsup_size = (y_pred.size(0) - sup_size) // 2

        # 有监督部分, 用交叉熵损失
        y_pred_sup = y_pred[:sup_size]
        if self.tsa_schedule is None:
            loss_sup = self.loss_sup(y_pred_sup, y_true_sup)
        else:  # 使用tsa来去掉预测概率较高的有监督样本
            threshold = self.get_tsa_threshold(self.tsa_schedule, global_step, total_steps, self.start, self.end)
            true_prob = torch.gather(F.softmax(y_pred_sup, dim=-1), dim=1, index=y_true_sup[:, None])
            sel_rows = true_prob.lt(threshold).sum(dim=-1).gt(0)  # 仅保留小于阈值的样本
            loss_sup = self.loss_sup(y_pred_sup[sel_rows], y_true_sup[sel_rows]) if sel_rows.sum() > 0 else 0

        # 无监督部分，这里用KL散度，也可以用交叉熵
        y_true_unsup = y_pred[sup_size:sup_size + unsup_size]
        y_true_unsup = F.softmax(y_true_unsup.detach(), dim=-1)
        y_pred_unsup = F.log_softmax(y_pred[sup_size + unsup_size:], dim=-1)
        loss_unsup = self.loss_unsup(y_pred_unsup, y_true_unsup)
        if self.return_all_loss:
            return loss_sup + loss_unsup, loss_sup, loss_unsup
        else:
            return loss_sup + loss_unsup

    @staticmethod
    def get_tsa_threshold(schedule, global_step, num_train_steps, start, end):
        training_progress = global_step / num_train_steps
        if schedule == "linear_schedule":
            threshold = training_progress
        elif schedule == "exp_schedule":
            scale = 5
            threshold = math.exp((training_progress - 1) * scale)
        elif schedule == "log_schedule":
            scale = 5
            threshold = 1 - math.exp((-training_progress) * scale)
        return threshold * (end - start) + start
