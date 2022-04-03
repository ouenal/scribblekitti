import torch
import torch.nn as nn

class PartialConsistencyLoss(nn.Module):
    def __init__(self, H, ignore_index=0):
        super().__init__()
        self.ignore_index = ignore_index
        self.supervised_loss = H(ignore_index=ignore_index, reduction='sum')
        self.consistency_loss = nn.KLDivLoss(reduction='sum')

    def forward(self, student_output, teacher_output, student_label):
        loss_s = self.compute_supervised_loss(student_output, student_label)
        mask = student_label == self.ignore_index
        loss_u = self.compute_consistency_loss(student_output, teacher_output, mask=mask)
        return (loss_s + loss_u)/student_output.shape[-1]

    def compute_supervised_loss(self, student_output, student_label):
        return self.supervised_loss(student_output, student_label)

    def compute_consistency_loss(self, student_output, teacher_output, mask=None):
        student_output_reshaped = student_output.permute(0,2,1).log_softmax(-1) 
        teacher_output_reshaped = teacher_output.permute(0,2,1).softmax(-1)
        return self.consistency_loss(student_output_reshaped[mask],
                                     teacher_output_reshaped[mask])