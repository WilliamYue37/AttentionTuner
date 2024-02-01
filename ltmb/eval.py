import gymnasium as gym
import numpy as np
from transformer import Transformer
import ltmb
import argparse, configparser
import torch
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import os
import random
import torch.nn.functional as F

# Each tile is encoded as a 3 dimensional tuple: (OBJECT_IDX, COLOR_IDX, STATE)
OBJECT_IDX = 0
COLOR_IDX = 1
STATE = 2

def one_hot(obs):
        """converts (7, 7, 3) tensor obs to (7, 7, 20) one hot representation"""
        objects, colors, states = obs[:, :, OBJECT_IDX], obs[:, :, COLOR_IDX], obs[:, :, STATE]
        objects = F.one_hot(objects, num_classes=11).float()
        colors = F.one_hot(colors, num_classes=6).float()
        states = F.one_hot(states, num_classes=3).float()
        return torch.cat([objects, colors, states], dim=-1)

def evaluate_policy(env, policy: Transformer):
    """Evaluate a policy on a single episode and returns 1 if episode was successful, 0 otherwise"""
    obs, info = env.reset(seed=random.randint(0, 10**9))
    done = False
    obs_history, action_history = [], []
    rewards = 0
    while not done:
        obs = one_hot(torch.from_numpy(obs['image']).long()) # (7, 7, 20)
        obs = obs.permute(2, 0, 1).cuda() # (20, 7, 7)
        assert obs.shape == (20, 7, 7)
        obs_history.append(obs)
        obs_input = torch.stack(obs_history).unsqueeze(0) # (1, seq len, C, H, W)
        action_history.append(0) # dummy action so that obs_input and action_input have the same length
        action_input = torch.LongTensor(action_history).unsqueeze(0).cuda() # (1, seq len)
        output, _, attention_weights = policy(obs_input, action_input)
        output = output.squeeze(0).argmax(dim=-1)
        action = output[-1].item() # last action
        action_history.pop() # remove dummy action
        action_history.append(action)

        obs, reward, terminated, truncation, info = env.step(action)
        done = terminated or truncation
        rewards += reward
        if done and 'success' in info and info['success']:
            return 1, rewards, len(obs_history) + 1

    return 0, rewards, len(obs_history) + 1

def record_video(env_name: str, policy: Transformer, filename: str, length: int):
    env = gym.make(env_name, render_mode='rgb_array', length=length)
    env = gym.wrappers.RecordVideo(env, filename)
    obs, info = env.reset(seed=random.randint(0, 10**9))
    done = False
    obs_history, action_history = [], []

    while not done:
        obs = one_hot(torch.from_numpy(obs['image']).long()) # (7, 7, 20)
        obs = obs.permute(2, 0, 1).cuda() # (20, 7, 7)
        assert obs.shape == (20, 7, 7)
        obs_history.append(obs)
        obs_input = torch.stack(obs_history).unsqueeze(0) # (1, seq len, C, H, W)
        action_history.append(0) # dummy action so that obs_input and action_input have the same length
        action_input = torch.LongTensor(action_history).unsqueeze(0).cuda() # (1, seq len)
        output, _, attention_weights = policy(obs_input, action_input)
        output = output.squeeze(0).argmax(dim=-1)
        action = output[-1].item() # last action
        action_history.pop() # remove dummy action
        action_history.append(action)

        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

    env.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, help='config file for training run')
    parser.add_argument('--model', type=str, default=None, help='path to the learner model file')
    parser.add_argument('--runs', type=int, default=10, help='number of runs to average over')
    parser.add_argument('--ckpt_folder', type=str, default=None, help='folder used to save checkpoints and logs, evaluation results will be saved here in a folder called \'eval\' if provided')
    parser.add_argument('--benchmark', type=str, choices=['LTMB-Hallway-v0', 'LTMB-Ordering-v0', 'LTMB-Counting-v0'], help='benchmark to evaluate on')
    parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
    parser.add_argument('--num_heads', type=int, default=2, help='number of heads for multi-head attention')
    parser.add_argument('--num_layers', type=int, default=2, help='number of layers for transformer')
    parser.add_argument('--length', type=int, default=10, help='Length of the task.')
    parser.add_argument('--test_freq', type=float, default=0.3, help='Frequency of test rooms for Counting task.')
    parser.add_argument('--seed', type=int, default=0, help='Random seed.')
    args = parser.parse_args()

    if args.config is not None:
        config = configparser.ConfigParser()
        config.read(args.config)
        defaults = dict(config.items("Defaults"))
        parser.set_defaults(**defaults)
        args = parser.parse_args()

    # check for required arguments
    if args.benchmark is None or args.model is None:
        raise ValueError('Missing one of the required arguments: --benchmark, --model')
    
    # set random seed
    random.seed(args.seed)

    # compute max length of episode and action dimension
    options = {'length': args.length}
    max_len = None
    if args.benchmark == 'LTMB-Hallway-v0':
        max_len = 4 * args.length + 25
    elif args.benchmark == 'LTMB-Ordering-v0':
        max_len = 18 + args.length
    elif args.benchmark == 'LTMB-Counting-v0':
        max_len = 7 * args.length
        options['test_freq'] = args.test_freq
    assert max_len is not None
    action_dim = 7
    
    env = gym.make(args.benchmark, **options)
    print(f"Evaluating on {args.benchmark}...")
    policy = Transformer(action_dim, args.d_model, args.num_heads, args.num_layers, max_len).cuda()
    policy.load_state_dict(torch.load(args.model)['model'])
    policy.eval()

    successes, avg_rewards, avg_length = 0, 0, 0
    for i in range(args.runs):
        success, reward, episode_len = evaluate_policy(env, policy)
        successes += success
        avg_rewards += reward
        avg_length += episode_len

    success_rate = successes / args.runs * 100
    avg_rewards /= args.runs
    avg_length /= args.runs

    # save evaluation results in ckpt folder if provided
    evaluation_folder = ''
    if args.ckpt_folder is not None:
        # create eval folder
        evaluation_folder = 'runs/' + args.ckpt_folder + '/evaluation_results'
        if not os.path.exists(evaluation_folder):
            os.mkdir(evaluation_folder)
        
        # save statistics in a file called stats.txt
        with open(evaluation_folder + '/stats.txt', 'w') as f:
            f.write(f"Statistics over {args.runs} runs:\n")
            f.write(f"Average Success Rate: {success_rate}%\n")
            f.write(f"Average Rewards: {avg_rewards}\n")
            f.write(f"Average Length: {avg_length}\n")

    print(f"Statistics over {args.runs} runs:")
    print(f"Average Success Rate: {success_rate}%")
    print(f"Average Rewards: {avg_rewards}")
    print(f"Average Length: {avg_length}")

    # visualize run
    record_video(args.benchmark, policy, evaluation_folder + '/video', args.length)

