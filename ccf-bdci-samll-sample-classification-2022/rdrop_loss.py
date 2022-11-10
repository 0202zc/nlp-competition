import torch
from torch import nn
from torch.nn import functional as F
from focal_loss import FocalLoss


class RDropLoss(nn.Module):
    '''R-Drop的Loss实现，官方项目：https://github.com/dropreg/R-Drop
    '''

    def __init__(self, class_num=2, alpha=4, rank='adjacent'):
        super().__init__()
        self.alpha = alpha
        # 支持两种方式，一种是奇偶相邻排列，一种是上下排列
        assert rank in {'adjacent', 'updown'}, "rank kwarg only support 'adjacent' and 'updown' "
        self.rank = rank
        # self.loss_sup = nn.CrossEntropyLoss()
        self.loss_sup = FocalLoss(class_num=class_num)
        self.loss_rdrop = nn.KLDivLoss(reduction='none')

    def forward(self, *args):
        '''支持两种方式: 一种是y_pred, y_true, 另一种是y_pred1, y_pred2, y_true
        '''
        assert len(args) in {2, 3}, 'RDropLoss only support 2 or 3 input args'
        # y_pred是1个Tensor
        if len(args) == 2:
            y_pred, y_true = args
            loss_sup = self.loss_sup(y_pred, y_true)  # 两个都算

            if self.rank == 'adjacent':
                y_pred1 = y_pred[1::2]
                y_pred2 = y_pred[::2]
                if len(y_pred2) > len(y_pred1):
                    y_pred2 = y_pred2[:-1]
            elif self.rank == 'updown':
                half_btz = y_true.shape[0] // 2
                y_pred1 = y_pred[:half_btz]
                y_pred2 = y_pred[half_btz:half_btz * 2]

        # y_pred是两个tensor
        else:
            y_pred1, y_pred2, y_true = args
            loss_sup = self.loss_sup(y_pred1, y_true)

        loss_rdrop1 = self.loss_rdrop(F.log_softmax(y_pred1, dim=-1), F.softmax(y_pred2, dim=-1))
        loss_rdrop2 = self.loss_rdrop(F.log_softmax(y_pred2, dim=-1), F.softmax(y_pred1, dim=-1))
        return loss_sup + torch.mean(loss_rdrop1 + loss_rdrop2) / 4 * self.alpha
