import torch
from torch.utils.data import Dataset
import pickle
import numpy as np

class MemoryDataset(Dataset):
    """Memory Gym Dataset"""

    def __init__(self, file, benchmark, heads, even_split, trajectory_len=None):
        super().__init__()
        self.benchmark = benchmark
        self.heads = heads
        self.even_split = even_split
        self.max_len = -1
        if benchmark == 'MortarMayhem-Grid-v0':
            self.max_len = 118
        elif benchmark == 'MysteryPath-Grid-v0':
            self.max_len = 128
        elif benchmark == 'SearingSpotlights-v0':
            self.max_len = 75
        if trajectory_len is not None:
            self.max_len = trajectory_len
        assert self.max_len != -1

        # unpickle file
        with open(file, 'rb') as f:
            self.demos = pickle.load(f)
        
        for i in range(len(self.demos)):
            trajectory, memory_associations = self.demos[i]
            observations, actions = [], []
            for j in range(len(trajectory)):
                obs, action = trajectory[j]
                obs = np.transpose(obs, (2, 0, 1)) # (H, W, C) -> (C, H, W)
                observations.append(torch.from_numpy(obs) / 255.0)
                actions.append(torch.tensor(action))
            self.demos[i] = (observations, actions, memory_associations)

    def __len__(self):
        return len(self.demos) 
    
    def _get_head(self, x, y):
        """Returns the head that timestep belongs to. All head sizes are equal except for the last head which may be smaller."""
        max_len = x + 1
        if y < 0 or y >= max_len:
            raise ValueError(f'Invalid timestep {y}. Must be between 0 and {max_len - 1}.')
    
        head_size = (max_len + self.heads - 1) // self.heads # round up 
        return y // head_size

    def __getitem__(self, idx):
        # create attention map labels
        observations, actions, memory_associations = self.demos[idx]

        # create attention pad mask (needs to be twice as long because of observations and actions)
        # "traj" is the concatenation of observations and actions
        max_traj_len = 2 * self.max_len
        traj_len = 2 * len(observations) # observations and actions
        padding_needed = max_traj_len - traj_len
        attention_pad_mask = torch.zeros((max_traj_len, max_traj_len))
        attention_pad_mask[:traj_len, :traj_len] = 1

        # create pad mask for action predictions
        padding_needed = self.max_len - len(observations)
        pad_mask = [1] * len(observations) + [0] * padding_needed

        # pad observations and actions
        obs_pad, action_pad = torch.zeros_like(observations[0]), torch.zeros_like(actions[0])
        observations += [obs_pad] * padding_needed
        actions += [action_pad] * padding_needed
        
        # create correction attention weights for memory loss
        if self.even_split:
            correct_attention_weights = [torch.zeros((max_traj_len, max_traj_len)) for _ in range(self.heads)]
            for x, y in memory_associations:
                head = self._get_head(x, y)
                correct_attention_weights[head][x][y] = 1
        else:
            correct_attention_weights = [torch.zeros((max_traj_len, max_traj_len))]
            for x, y in memory_associations:
                correct_attention_weights[0][x][y] = 1

        observations = torch.stack(observations)
        actions = torch.stack(actions)
        pad_mask = torch.tensor(pad_mask).bool()
        attention_pad_mask = attention_pad_mask.bool()
        return observations, actions, pad_mask, attention_pad_mask, *correct_attention_weights
