import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from torch import optim
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
import os
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import seaborn as sns

class Trainer():
    def __init__(self, model, train_dataset, test_dataset, args):
        self.model = model

        self.writer = SummaryWriter(log_dir=args.ckpt_folder + '/learner_logs')
        self.ckpt_every = args.ckpt_every
        self.ckpts_folder = Path(args.ckpt_folder + '/learner_ckpts')
        self.attention_heatmap_folder = Path(args.ckpt_folder + '/attention_heatmaps')

        self.ds = train_dataset
        self.dl = DataLoader(self.ds, batch_size = args.batch_size, shuffle=True, pin_memory=True, num_workers=args.num_workers)
        self.test_ds = test_dataset
        if test_dataset is not None:
            self.test_dl = DataLoader(self.test_ds, batch_size = args.batch_size, shuffle=True, pin_memory=True, num_workers=args.num_workers)

        self.batch_size = args.batch_size
        self.opt = Adam(model.parameters(), lr = args.lr)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.opt, T_max=100)

        self.memory_multiplier = args.memory_multiplier
        self.memory_loss_heads = args.memory_loss_heads
        self.viz_heads = args.viz_heads
        self.viz_num_layers = max(layer for layer, _ in self.viz_heads)
        self.viz_num_heads = max(head for _, head in self.viz_heads)
    
        self.eval_every = args.eval_every
        self.visualize_every = args.viz_every
        self.epoch = 1

    def save(self, milestone):
        "save model: string is for special milestones, int is for regular milestones"
        data = {
            'epoch': self.epoch,
            'model': self.model.state_dict(),
        }
        os.makedirs(str(self.ckpts_folder), exist_ok=True)
        torch.save(data, str(self.ckpts_folder / f'model-{milestone}.pt'))

        # delete previous milestone
        if isinstance(milestone, int):
            last_mile = str(self.ckpts_folder / f'model-{milestone - 1}.pt')
            if os.path.exists(last_mile):
                os.remove(last_mile)

    def load(self, ckpt):
        data = torch.load(ckpt)
        self.epoch = data['epoch']
        self.model.load_state_dict(data['model'])

    def load_new_dataset(self, dataset):
        self.ds = dataset
        self.dl = DataLoader(self.ds, batch_size = self.batch_size, shuffle=True, pin_memory=True)

    def visualize_attention(self, attention_weights, ground_truth_attention_weights):
        batch = 0 # visualize the first batch
        num_selected_pairs = len(self.viz_heads)
        fig, axes = plt.subplots(self.viz_num_layers+1, self.viz_num_heads, figsize=(6 * (self.viz_num_layers + 1), 6 * self.viz_num_heads))

        # Display the ground truth attention in the first row
        for head in range(self.viz_num_heads):
            ground_truth_attention_map = ground_truth_attention_weights[batch].numpy(force=True)
            ax = axes[0, head]
            sns.heatmap(ground_truth_attention_map, ax=ax, cmap='YlGnBu', cbar=True, square=True)
            ax.set_title(f'Ground Truth Attention')
            ax.set_xlabel('Key')
            ax.set_ylabel('Query')

        # Display selected model attention in the subsequent rows
        for layer, head in self.viz_heads:
            attention_map = attention_weights[layer - 1][batch][head - 1].numpy(force=True)
            ax = axes[self.viz_num_layers - layer + 1, head - 1]

            sns.heatmap(attention_map, ax=ax, cmap='YlGnBu', cbar=False, square=True)
            ax.set_title(f'Model Attention - Layer {layer}, Head {head}')
            ax.set_xlabel('Key')
            ax.set_ylabel('Query')
        
        # Clear unused subplots
        for layer in range(1, self.viz_num_layers + 1):
            for head in range(1, self.viz_num_heads + 1):
                if (layer, head) not in self.viz_heads:
                    axes[self.viz_num_layers - layer + 1, head - 1].axis('off')  # Turn off the axis 

        os.makedirs(str(self.attention_heatmap_folder), exist_ok=True)
        plt.tight_layout()
        plt.savefig(str(self.attention_heatmap_folder / f'epoch-{self.epoch}.png'), bbox_inches='tight', dpi=300)
        plt.close()

    def get_test_accuracy(self):
        self.model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for batch_idx, (obs, actions, pad_mask, correct_attention_weights, attention_pad_mask) in enumerate(self.test_dl):
                obs, actions, pad_mask = obs.cuda(), actions.cuda(), pad_mask.cuda()
                output, _, attention_weights = self.model(obs, actions)
                pred = output.argmax(dim=-1)
                pred, actions = torch.masked_select(pred, pad_mask), torch.masked_select(actions, pad_mask)
                correct += pred.eq(actions).sum().item()
                total += torch.numel(actions)
        return correct / total

    def train(self, num_epoch):
        self.model.train()
        for _ in range(num_epoch):
            total_loss, total_output_loss, total_attention_loss = 0, 0, 0
            correct, total = 0, 0
            attention_weights, correct_attention_weights = [], [] # for visualization
            for batch_idx, (obs, actions, pad_mask, correct_attention_weights, attention_pad_mask) in enumerate(self.dl):
                self.opt.zero_grad()
                obs, actions, pad_mask, correct_attention_weights, attention_pad_mask = obs.cuda(), actions.cuda(), pad_mask.cuda(), correct_attention_weights.cuda(), attention_pad_mask.cuda()
                output, _, attention_weights = self.model(obs, actions)

                # compute loss
                output = output.permute(0, 2, 1)
                output_loss = nn.CrossEntropyLoss(reduction='none')(output, actions)
                output_loss = torch.masked_select(output_loss, pad_mask).mean()
                attention_loss = 0
                for layer, head in self.memory_loss_heads:
                    head_attention_loss = nn.BCELoss(reduction='none')(attention_weights[layer - 1][:, head - 1, :, :], correct_attention_weights)
                    head_attention_loss = torch.masked_select(head_attention_loss, attention_pad_mask).mean()
                    attention_loss += head_attention_loss
                attention_loss *= self.memory_multiplier / len(self.memory_loss_heads)
                loss = output_loss + attention_loss
                total_loss += loss.item()
                total_output_loss += output_loss.item()
                total_attention_loss += attention_loss.item()
                loss.backward()
                self.opt.step()

                # compute accuracy
                output = output.permute(0, 2, 1)
                pred = output.argmax(dim=-1)
                pred, actions = torch.masked_select(pred, pad_mask), torch.masked_select(actions, pad_mask)
                correct += pred.eq(actions).sum().item()
                total += torch.numel(actions)

            total_loss /= len(self.dl.dataset)
            total_output_loss /= len(self.dl.dataset)
            total_attention_loss /= len(self.dl.dataset)
            self.writer.add_scalar('Loss/train', total_loss, self.epoch)
            self.writer.add_scalar('Loss/output', total_output_loss, self.epoch)
            self.writer.add_scalar('Loss/attention', total_attention_loss, self.epoch)
            self.writer.add_scalar('Accuracy/train', correct / total, self.epoch)

            if self.epoch % self.ckpt_every == 0:
                milestone = self.epoch // self.ckpt_every
                self.save(milestone)

            if self.epoch % self.visualize_every == 0:
                self.visualize_attention(attention_weights, correct_attention_weights)

            if self.test_ds is not None and self.epoch % self.eval_every == 0 or self.epoch == 1:
                acc = self.get_test_accuracy()
                self.model.train()
                self.writer.add_scalar('Accuracy/test', acc, self.epoch)

            # self.writer.add_scalar('Learning Rate', self.scheduler.get_lr()[0], self.epoch)
            # self.scheduler.step()

            self.epoch += 1