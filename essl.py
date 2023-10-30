import os
import sys
import time
import json
import signal
import argparse
import datetime
import subprocess
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
import torchvision

from utils import Transform, LARS, adjust_learning_rate, handle_sigusr1, handle_sigterm, off_diagonal


def parse_args():
    parser = argparse.ArgumentParser(description='Contrastive Learning Pretrain with Barlow Twins Loss')
    # base settings
    parser.add_argument('--data', type=str, default='/pub/data/hujie/zdata/data',
                        help='path to dataset')
    parser.add_argument("--set", type=str, choices=['stl10', 'cifar10', 'cifar100', 'tiny'], default='stl10',
                        help='dataset')
    parser.add_argument('--workers', type=int, default=8,
                        help='number of data loader workers')
    parser.add_argument('--epochs', type=int, default=500,
                        help='number of total epochs to run')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='mini-batch size')
    parser.add_argument('--print_freq', type=int, default=100,
                        help='print frequency')
    parser.add_argument('--save_freq', type=int, default=10,
                        help='print frequency')
    parser.add_argument('--agent_freq', type=int, default=128,
                        help='print frequency')
    parser.add_argument('--resume', type=str, default=None,
                        help='path to checkpoint directory')
    parser.add_argument('--model_path', type=str, default="models_pt",
                        help='path to save model')
    parser.add_argument('--tb_path', type=str, default="runs_pt",
                        help='path to tensorboard')
    parser.add_argument('--name', type=str, default=None,
                        help='algo name')
    # training settings
    parser.add_argument('--learning_rate_weights', type=float, default=0.2,
                        help='base learning rate for weights')
    parser.add_argument('--learning_rate_biases', type=float, default=0.0048,
                        help='base learning rate for biases and batch norm parameters')
    parser.add_argument('--weight-decay', type=float, default=0.0004,
                        help='weight decay')
    parser.add_argument('--lambd', type=float, default=0.0051,
                        help='weight on off-diagonal terms')
    parser.add_argument('--projector', type=str, default='8192-8192-8192',
                        help='projector MLP')
    parser.add_argument('--temperature', type=float, default=0.07,
                        help='')
    parser.add_argument('--n-views', type=int, default=2,
                        help='')
    parser.add_argument('--agent', type=bool, default=True,
                        help='')
    parser.add_argument('--balance', type=float, default=0.01,
                        help='')
    args = parser.parse_args()

    pretrain_time = str(datetime.datetime.now().replace(microsecond=0).strftime("%Y%m%d-%H%M"))
    args.time = pretrain_time

    if args.name is None:
        args.name = "ensemble"
    if args.set == "stl10":
        save_path_base = "saved/" + args.name + "_" + pretrain_time + "_STL-10"
        args.data = os.path.join(args.data, "STL-10/unlabeled")
    elif args.set == "cifar10":
        save_path_base = "saved/" + args.name + "_" + pretrain_time + "_CIFAR-10"
        args.data = os.path.join(args.data, "CIFAR-10/unlabeled")
    elif args.set == "cifar100":
        save_path_base = "saved/" + args.name + "_" + pretrain_time + "_CIFAR-100"
        args.data = os.path.join(args.data, "CIFAR-100/unlabeled")
    elif args.set == "tiny":
        save_path_base = "saved/" + args.name + "_" + pretrain_time + "_Tiny"
        args.data = os.path.join(args.data, "tiny-imagenet-200/unlabeled")
    elif args.set == "imagenet":
        save_path_base = "saved/" + args.name + "_" + pretrain_time + "_ImageNet"
        args.data = os.path.join(args.data, "imagenet/unlabeled")
    else:
        raise FileNotFoundError

    args.model_path = os.path.join(save_path_base, args.model_path)
    args.tb_path = os.path.join(save_path_base, args.tb_path)

    args.save_list = list(range(100, args.epochs, 100))
    args.save_list.extend(list(range(args.epochs - 40, args.epochs + 1, 10)))

    if not os.path.isdir(save_path_base):
        os.makedirs(save_path_base)
        args.save_path_base = save_path_base
    if not os.path.isdir(args.model_path):
        os.makedirs(args.model_path)
    if not os.path.isdir(args.tb_path):
        os.makedirs(args.tb_path)
    if not os.path.isdir(args.data):
        raise ValueError('data path not exist: {}'.format(args.data))

    return args


def main():
    args = parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)
    torch.backends.cudnn.benchmark = True

    dataset = torchvision.datasets.ImageFolder(args.data, Transform())
    print("n samples:", len(dataset))
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True, drop_last=True)
    model = ESSL(args, device).cuda()
    param_weights = []
    param_biases = []
    for param in model.parameters():
        if param.ndim == 1:
            param_biases.append(param)
        else:
            param_weights.append(param)
    parameters = [{'params': param_weights}, {'params': param_biases}]
    optimizer = LARS(parameters, lr=0, weight_decay=args.weight_decay,
                     weight_decay_filter=True,
                     lars_adaptation_filter=True)

    agent = Agent().cuda()
    states = torch.zeros((200, 1, 1024)).to(device)
    actions = torch.zeros((200, 1, 2)).to(device)
    logprobs = torch.zeros((200, 1)).to(device)
    rewards = torch.zeros((200, 1)).to(device)
    dones = torch.ones((200, 1)).to(device)
    values = torch.zeros((200, 1)).to(device)
    index = 0

    anchor = torch.tensor([[0.85, 0.85]])

    scaler = torch.cuda.amp.GradScaler()
    start_time = time.time()
    for epoch in range(0, args.epochs):
        for step, ((y1, y2), _) in enumerate(loader):
            y1 = y1.cuda(non_blocking=True)
            y2 = y2.cuda(non_blocking=True)
            adjust_learning_rate(args, optimizer, loader, step)
            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                infonce, barlowtwins, next_state = model.forward(y1, y2)
                states[index] = next_state
                action, logprob, value = agent.get_action(next_state)
                alpha = action[0][0].item()
                beta = action[0][1].item()
                actions[index] = action
                logprobs[index] = logprob
                values[index] = value
                loss = (0.5 + alpha) * infonce + (0.5 + beta) * barlowtwins
                reward = F.cosine_similarity(anchor, action.cpu())
                rewards[index] = reward
                index += 1
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            if step % args.print_freq == 0:
                stats = dict(epoch=epoch, step=step,
                             loss=loss.item(),
                             time=int(time.time() - start_time))
                print(json.dumps(stats))

            if index % args.agent_freq == 0:
                data = states, actions, logprobs, rewards, dones, values
                agent.learn(data)
                states = torch.zeros((args.agent_freq, 1, 1024)).to(device)
                actions = torch.zeros((args.agent_freq, 1, 2)).to(device)
                logprobs = torch.zeros((args.agent_freq, 1)).to(device)
                rewards = torch.zeros((args.agent_freq, 1)).to(device)
                dones = torch.ones((args.agent_freq, 1)).to(device)
                values = torch.zeros((args.agent_freq, 1)).to(device)
                index = 0


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    def __init__(self):
        super().__init__()
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(1024, 512)),
            nn.ReLU(inplace=True),
            layer_init(nn.Linear(512, 2), std=0.01),
        )
        self.actor_logstd = nn.Parameter(torch.zeros(1, 2))
        self.critic = nn.Sequential(
            layer_init(nn.Linear(1024, 512)),
            nn.ReLU(inplace=True),
            layer_init(nn.Linear(512, 1), std=1)
        )
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.0001, eps=1e-5)
        self.device = torch.device("cuda:0")

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        action_mean = self.actor_mean(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(x)

    @torch.no_grad()
    def get_action(self, x):
        action, logprob, _, value = self.get_action_and_value(x)
        action = F.softmax(action, dim=1)
        value = value.flatten()
        return action.detach(), logprob, value

    def learn(self, data):
        states, actions, logprobs, rewards, dones, values = data
        with torch.no_grad():
            advantages = torch.zeros_like(rewards).to(self.device)
            lastgaelam = 0
            for t in reversed(range(128)):
                if t == 128 - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + 0.99 * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + 0.99 * 0.95 * nextnonterminal * lastgaelam
            returns = advantages + values

            b_obs = obs.reshape((-1, 1024))
            b_logprobs = logprobs.reshape(-1)
            b_actions = actions.reshape((-1, 2))
            b_advantages = advantages.reshape(-1)
            b_returns = returns.reshape(-1)
            b_values = values.reshape(-1)

            # Optimizing the policy and value network
            b_inds = np.arange(128)
            clipfracs = []
            for epoch in range(10):
                np.random.shuffle(b_inds)
                for start in range(0, 128, 32):
                    end = start + 32
                    mb_inds = b_inds[start:end]

                    _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions[mb_inds])
                    logratio = newlogprob - b_logprobs[mb_inds]
                    ratio = logratio.exp()

                    with torch.no_grad():
                        # calculate approx_kl http://joschu.net/blog/kl-approx.html
                        old_approx_kl = (-logratio).mean()
                        approx_kl = ((ratio - 1) - logratio).mean()
                        clipfracs += [((ratio - 1.0).abs() > 0.2).float().mean().item()]

                    mb_advantages = b_advantages[mb_inds]
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                    # Policy loss
                    pg_loss1 = -mb_advantages * ratio
                    pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - 0.2, 1 + 0.2)
                    pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                    # Value loss
                    newvalue = newvalue.view(-1)
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -0.2,
                        0.2,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()

                    entropy_loss = entropy.mean()
                    loss = pg_loss - 0.01 * entropy_loss + v_loss * 0.5

                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(agent.parameters(), 0.5)
                    optimizer.step()


class ESSL(nn.Module):
    def __init__(self, args, device):
        super().__init__()
        self.args = args
        self.backbone = torchvision.models.resnet18(zero_init_residual=True)
        self.backbone.fc = nn.Identity()

        # projector
        sizes = [512] + list(map(int, args.projector.split('-')))
        layers = []
        for i in range(len(sizes) - 2):
            layers.append(nn.Linear(sizes[i], sizes[i + 1], bias=False))
            layers.append(nn.BatchNorm1d(sizes[i + 1]))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Linear(sizes[-2], sizes[-1], bias=False))
        self.projector = nn.Sequential(*layers)

        # normalization layer for the representations z1 and z2
        self.bn = nn.BatchNorm1d(sizes[-1], affine=False)
        self.device = device

    def infonce(self, z1, z2):
        batch_size = z1.shape[0]

        features = torch.cat((z1, z2), dim=0)
        labels = torch.cat([torch.arange(batch_size) for i in range(self.args.n_views)], dim=0)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        labels = labels.cuda(self.device)

        features = F.normalize(features, dim=1)
        similarity_matrix = torch.matmul(features, features.T)
        mask = torch.eye(labels.shape[0], dtype=torch.bool).cuda()
        labels = labels[~mask].view(labels.shape[0], -1)
        similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)

        # similarity_matrix.div_(similarity_matrix)
        # torch.distributed.all_reduce(similarity_matrix)

        positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)
        negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

        logits = torch.cat([positives, negatives], dim=1)
        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()
        logits = logits / self.args.temperature
        loss = torch.nn.CrossEntropyLoss()(logits, labels)
        return loss

    def barlowtwins(self, z1, z2):
        N = z1.size(0)
        D = z2.size(1)

        # cross-correlation matrix
        c = torch.mm(z1.T, z2) / N  # DxD
        # loss
        c_diff = (c - torch.eye(D, device=self.device)).pow(2)  # DxD
        # multiply off-diagonal elems of c_diff by lambda
        c_diff[~torch.eye(D, dtype=bool)] *= 5e-3
        loss = c_diff.sum()
        return loss

    def forward(self, y1, y2):
        h1 = self.backbone(y1)
        h2 = self.backbone(y2)
        z1 = self.bn(self.projector(h1))
        z2 = self.bn(self.projector(h2))
        infonce = self.infonce(z1, z2)
        barlowtwins = self.barlowtwins(z1, z2)
        return infonce, barlowtwins, torch.mean(torch.cat((h1.detach(), h2.detach()), dim=1), dim=0, keepdim=True)


if __name__ == '__main__':
    main()
