import torch
import torch.nn as nn


class arch_model(nn.Module):
    def __init__(self, num_ops = 9, num_choices = 2):
        super(arch_model, self).__init__()
        self.arch_weights = nn.Parameter(1e-3*torch.ones(num_ops, num_choices))



        self.optimizer = torch.optim.Adam([self.arch_weights], lr=3e-4, betas=(0.5, 0.999), weight_decay=1e-3)


    def step(self, rewards, log_probs):
        self.optimizer.zero_grad()
        rewards = torch.stack(rewards, dim=0).unsqueeze(1)
        log_probs = torch.stack(log_probs, dim=0).unsqueeze(1)
        if len(rewards) > 1:
            loss = -(rewards - rewards.mean()) * log_probs
        else:
            loss = -rewards * log_probs
        loss = loss.mean()
        loss.backward()
        self.optimizer.step()

