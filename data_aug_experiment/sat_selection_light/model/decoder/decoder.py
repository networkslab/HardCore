import torch
import torch.nn as nn
import torch.nn.functional as F

# class BinClassifierDecoder(nn.module):
#     def __init__(self, sat_dim, sol_dim, solver_enc_mode, num_classes=1):
#         super().__init__()
#         self.solver_enc_mode = solver_enc_mode
#         self.sat_dim = sat_dim
#         self.sol_dim = sol_dim
#         self.num_classes = num_classes 
#         if self.solver_enc_mode == 'none':
#             # If solver is not encoded, in_dim equals to dimension of SAT instance encoder.
#             self.classifier = nn.Linear(self.sat_dim, self.num_classes)
#         elif sat_dim != sol_dim:
#             self.lin = nn.Linear(self.sat_dim, self.sol_dim)

#     def forward(self, sat_h, solver_h):
#         if self.solver_enc_mode == 'none':
#             logits = self.classifier(sat_h)
#             # logits = F.normalize(torch.sigmoid(h), p=1)
#         else:
#             if self.sat_dim != self.sol_dim:
#                 sat_h = self.lin(F.relu(sat_h))
#             assert sat_h.shape[1] == solver_h.shape[1]
#             logits = torch.mm(sat_h, solver_h.T)
#         return logits

class ClassifierDecoder(nn.Module):
    def __init__(self, sat_dim, sol_dim, solver_enc_mode, num_classes=7):
        super().__init__()
        self.solver_enc_mode = solver_enc_mode
        self.sat_dim = sat_dim
        self.sol_dim = sol_dim
        self.num_classes = num_classes 
        if self.solver_enc_mode == 'none':
            # If solver is not encoded, in_dim equals to dimension of SAT instance encoder.
            self.classifier = nn.Linear(self.sat_dim, self.num_classes)
        elif sat_dim != sol_dim:
            self.lin = nn.Linear(self.sat_dim, self.sol_dim)

    def forward(self, sat_h, solver_h):
        if self.solver_enc_mode == 'none':
            logits = self.classifier(sat_h)
            # logits = F.normalize(torch.sigmoid(h), p=1)
        else:
            if self.sat_dim != self.sol_dim:
                sat_h = self.lin(F.relu(sat_h))
            assert sat_h.shape[1] == solver_h.shape[1]
            logits = torch.mm(sat_h, solver_h.T)
        return logits

class ClassifierDecoderV1(nn.Module):
    def __init__(self, sat_dim, sol_dim, hidden_dim, num_classes, solver_enc_mode):
        super().__init__()
        self.num_classes = num_classes
        self.solver_enc_mode = solver_enc_mode
        if self.solver_enc_mode == 'none':
            # If solver is not encoded, in_dim equals to dimension of SAT instance encoder.
            self.classifier = nn.Linear(sat_dim, num_classes)
        else:
            self.lin1 = nn.Linear(sat_dim + sol_dim, hidden_dim)
            self.lin2 = nn.Linear(hidden_dim, hidden_dim//2)
            self.classifier = nn.Linear(hidden_dim//2, 1)

    def forward(self, sat_h, solver_h):
        # assert self.num_classes == solver_h.shape[0]
        if self.solver_enc_mode == 'none':
            h = self.classifier(sat_h)
            probs = F.normalize(torch.sigmoid(h), p=1)
        else:
            bs = sat_h.shape[0]
            sat_h_rep = sat_h.unsqueeze(1).repeat(1, self.num_classes, 1)   # [bs, num_classes, hidden_dim]
            solver_h_rep = solver_h.unsqueeze(0).repeat(bs, 1, 1)           # [bs, num_classes, hidden_dim]
            h = torch.cat([sat_h_rep, solver_h_rep], dim=2)      # [bs, num_classes, hidden_dim*2]
            h = F.relu(self.lin1(h))
            h = F.relu(self.lin2(h))
            h = self.classifier(h).squeeze(-1)
            probs = F.normalize(torch.sigmoid(h), p=1)
        return probs
