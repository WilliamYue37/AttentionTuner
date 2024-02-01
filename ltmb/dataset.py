import torch
from torch.utils.data import Dataset
import pickle
import numpy as np
import torch.nn.functional as F

# Each tile is encoded as a 3 dimensional tuple: (OBJECT_IDX, COLOR_IDX, STATE)
OBJECT_IDX = 0
COLOR_IDX = 1
STATE = 2

class LTMBDataset(Dataset):
    """LTMB Dataset"""

    def __init__(self, file, max_len):
        super().__init__()
        self.max_len = max_len

        # unpickle file
        with open(file, 'rb') as f:
            self.demos = pickle.load(f)
        
        for i in range(len(self.demos)):
            trajectory, memory_associations = self.demos[i]
            observations, actions = [], []
            for j in range(len(trajectory)):
                obs, action = trajectory[j]
                obs = self.one_hot(torch.from_numpy(obs['image']).long()) # (7, 7, 20)
                obs = obs.permute(2, 0, 1) # (20, 7, 7)
                assert obs.shape == (20, 7, 7)
                observations.append(obs)
                actions.append(torch.tensor(action))
            self.demos[i] = (observations, actions, memory_associations)

    def one_hot(self, obs):
        """converts (7, 7, 3) tensor obs to (7, 7, 20) one hot representation"""
        objects, colors, states = obs[:, :, OBJECT_IDX], obs[:, :, COLOR_IDX], obs[:, :, STATE]
        objects = F.one_hot(objects, num_classes=11).float()
        colors = F.one_hot(colors, num_classes=6).float()
        states = F.one_hot(states, num_classes=3).float()
        return torch.cat([objects, colors, states], dim=-1)

    def __len__(self):
        return len(self.demos) 

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
        correct_attention_weights = torch.zeros((max_traj_len, max_traj_len))
        for x, y in memory_associations:
            correct_attention_weights[x][y] = 1

        observations = torch.stack(observations)
        actions = torch.stack(actions)
        pad_mask = torch.tensor(pad_mask).bool()
        attention_pad_mask = attention_pad_mask.bool()
        return observations, actions, pad_mask, correct_attention_weights, attention_pad_mask
