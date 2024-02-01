import gymnasium as gym
import numpy as np
from transformer import Transformer
import memory_gym
import argparse, configparser
import torch
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import os

def visualize_and_save_video(video_frames, output_path='output_frames.gif'):
    fig, ax = plt.subplots()
    img = ax.imshow(video_frames[0])

    def update(frame):
        img.set_array(video_frames[frame])
        return img,

    ani = FuncAnimation(fig, update, frames=len(video_frames), blit=True)

    ani.save(output_path, writer='imagemagick', fps=10)

def evaluate_policy(env, policy, benchmark, options={}):
    """Evaluate a policy on a single episode and returns 1 if episode was successful, 0 otherwise"""
    obs, info = env.reset(options=options)
    done = False
    obs_history, action_history, video = [], [], []
    rewards = 0
    timestep = 0
    while not done:
        timestep += 1
        video.append(obs)
        obs = torch.from_numpy(np.transpose(obs, (2, 0, 1))).cuda() / 255.0 # (H, W, C) -> (C, H, W)
        obs_history.append(obs)
        obs_input = torch.stack(obs_history).unsqueeze(0) # (1, seq len, C, H, W)
        action_history.append(0) # dummy action so that obs_input and action_input have the same length
        action_input = torch.LongTensor(action_history).unsqueeze(0).cuda() # (1, seq len)
        output, _, attention_weights = policy(obs_input, action_input)
        output = output.squeeze(0).argmax(dim=-1)
        action = output[-1].item() # last action
        action_history.pop() # remove dummy action
        action_history.append(action)

        # do action transform from discrete to multi-discrete for searing spotlights
        if benchmark == 'SearingSpotlights-v0':
            action = [action // 3, action % 3]

        obs, reward, done, truncation, info = env.step(action)
        rewards += reward
        if done and info['success']:
            video.append(obs)
            return 1, rewards, timestep, video

    video.append(obs)
    return 0, rewards, timestep, video

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, help='config file for training run')
    parser.add_argument('--model', type=str, default=None, help='path to the learner model file')
    parser.add_argument('--runs', type=int, default=10, help='number of runs to average over')
    parser.add_argument('--ckpt_folder', type=str, default=None, help='folder used to save checkpoints and logs, evaluation results will be saved here in a folder called \'eval\' if provided')
    parser.add_argument('--benchmark', type=str, choices=['MortarMayhem-Grid-v0', 'MysteryPath-Grid-v0', 'SearingSpotlights-v0'], help='benchmark to evaluate on')
    parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
    parser.add_argument('--num_heads', type=int, default=2, help='number of heads for multi-head attention')
    parser.add_argument('--num_layers', type=int, default=2, help='number of layers for transformer')
    parser.add_argument('--command_count', type=int, default=10, help='number of commands to generate per episode (Mortar Mayhem only)')
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

    env = gym.make(args.benchmark)
    print(f"Evaluating on {args.benchmark}...")
    action_dim = 9 if args.benchmark == 'SearingSpotlights-v0' else 4
    policy = Transformer(action_dim, args.d_model, args.num_heads, args.num_layers).cuda()
    policy.load_state_dict(torch.load(args.model)['model'])
    policy.eval()

    options = {}
    if args.benchmark == 'MortarMayhem-Grid-v0':
        options = {'command_count': [args.command_count]}
    successes, avg_rewards, avg_length = 0, 0, 0
    successful_run, failed_run = None, None
    for i in range(args.runs):
        success, reward, length, video = evaluate_policy(env, policy, args.benchmark, options=options)
        successes += success
        avg_rewards += reward
        avg_length += length

        if success and successful_run is None:
            successful_run = video
        elif not success and failed_run is None:
            failed_run = video

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

    # visualize successful and failed runs if they exist
    if successful_run is not None:
        visualize_and_save_video(successful_run, output_path = evaluation_folder + f'/successful_run.gif')
    if failed_run is not None:
        visualize_and_save_video(failed_run, output_path = evaluation_folder + f'/failed_run.gif')


