from __future__ import print_function
import os
import os.path as osp
import argparse
import sys
import h5py
import time
import datetime
import numpy as np
from tabulate import tabulate
from tensorboardX import SummaryWriter

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.optim import lr_scheduler
# from torch.distributions import Bernoulli
from torch.distributions import Categorical

from utils import Logger, read_json, write_json, save_checkpoint
from models import ActorCritic, init_action_space
from rewards import compute_param_reward
import vsum_tools
from glob import  glob

parser = argparse.ArgumentParser("Pytorch code for unsupervised video summarization with REINFORCE")
# Dataset options
parser.add_argument('-d', '--dataset', type=str, default='/data/RL_project/Project_code/learning_code_gru/dataset/scannet_feat_clip_scene0000_00_0_to_clip_scene0247_01_13.h5', help="path to h5 dataset (required)")
parser.add_argument('-s', '--split', type=str, default='dataset/splits_small.json', help="path to split file (required)")
parser.add_argument('--split-id', type=int, default=0, help="split index (default: 0)")
parser.add_argument('-m', '--metric', type=str, default='summe', choices=['tvsum', 'summe'],
                    help="evaluation metric ['tvsum', 'summe']")
# Model options
parser.add_argument('--input-dim', type=int, default=64, help="input dimension (default: 1024)")
parser.add_argument('--hidden-dim', type=int, default=512, help="hidden unit dimension of DSN (default: 256)")
parser.add_argument('--num-layers', type=int, default=1, help="number of RNN layers (default: 1)")
parser.add_argument('--rnn-cell', type=str, default='lstm', help="RNN cell type (default: lstm)")

# Optimization options
parser.add_argument('--lr', type=float, default=1e-5, help="learning rate (default: 1e-05)")
parser.add_argument('--weight-decay', type=float, default=1e-05, help="weight decay rate (default: 1e-05)")
parser.add_argument('--max-epoch', type=int, default=50000, help="maximum epoch for training (default: 60)")
parser.add_argument('--stepsize', type=int, default=0, help="how many steps to decay learning rate (default: 30)")
parser.add_argument('--gamma', type=float, default=0.1, help="learning rate decay (default: 0.1)")
parser.add_argument('--num-episode', type=int, default=5, help="number of episodes (default: 5)")
parser.add_argument('--beta', type=float, default=0.01, help="weight for summary length penalty term (default: 0.01)")

# Misc
parser.add_argument('--seed', type=int, default=1, help="random seed (default: 1)")
parser.add_argument('--gpu', type=str, default='0', help="which gpu devices to use")
parser.add_argument('--use-cpu', action='store_true', help="use cpu device")
parser.add_argument('--evaluate', action='store_true', help="whether to do evaluation only")
parser.add_argument('--save-dir', type=str, default='log/image_rand_4_param_action16_imgto64_grad_clip', help="path to save output (default: 'log/')")
parser.add_argument('--resume', type=str, default='', help="path to resume file")
parser.add_argument('--verbose', action='store_true', help="whether to show detailed test results")
parser.add_argument('--save-results', action='store_true', help="whether to save output results")

args = parser.parse_args()

torch.manual_seed(args.seed)
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
use_gpu = torch.cuda.is_available()
if args.use_cpu: use_gpu = False

def compute_returns(next_value, rewards, gamma=1):
    R = next_value
    returns = []
    for step in reversed(range(len(rewards))):
        R = rewards[step] + gamma * R #* masks[step]
        returns.insert(0, R)
    return returns

def plane_metric(gt_param, init_param, action, w=1, lamda=5):
    action = torch.tensor(action).cuda()
    a = action[0]; b = action[1];
    r = action[2]; d = action[3];
    # a= b= r= 2.*np.pi/180; d=0.05 # lead to dist ~ 0.0083, if only d=0.05 lead to dist ~0.007, r=2degree ~ 0.0067, a or b=2degree => 0.0073
    H = torch.Tensor([[torch.cos(r)*torch.cos(b) - torch.sin(r)*torch.sin(a)*torch.sin(b), -torch.cos(a)*torch.sin(r), torch.cos(r)*torch.sin(b) + torch.cos(b)*torch.sin(r)*torch.sin(a)],
                      [torch.cos(b)*torch.sin(r) + torch.cos(r)*torch.sin(a)*torch.sin(b),  torch.cos(r)*torch.cos(a), torch.sin(r)*torch.sin(b) - torch.cos(r)*torch.cos(b)*torch.sin(a)],
                      [-torch.cos(a)*torch.sin(b),                                          torch.sin(a),               torch.cos(a)*torch.cos(b)]]).cuda()
    new_normal = torch.matmul(H,init_param[:-1,:])
    new_param = torch.cat([new_normal,init_param[-1:,:]+ d ])
    distance = (1 - torch.sum(gt_param[:-1,:] * new_normal)) + w/(1+torch.exp(lamda-torch.abs(gt_param[-1:,:]-new_param[-1:,:]))).squeeze()
    return distance, new_param


def main():
    if not args.evaluate:
        sys.stdout = Logger(osp.join(args.save_dir, 'log_train.txt'))
    else:
        sys.stdout = Logger(osp.join(args.save_dir, 'log_test.txt'))
    print("==========\nArgs:{}\n==========".format(args))

    if use_gpu:
        print("Currently using GPU {}".format(args.gpu))
        cudnn.benchmark = True
        torch.cuda.manual_seed_all(args.seed)
    else:
        print("Currently using CPU")

    print("Initialize dataset {}".format(args.dataset))
    dataset = h5py.File(args.dataset, 'r')
    num_videos = len(dataset.keys())
    splits = read_json(args.split)
    assert args.split_id < len(splits), "split_id (got {}) exceeds {}".format(args.split_id, len(splits))
    split = splits[args.split_id]
    train_keys = split['train_keys']
    test_keys = split['test_keys']
    print("# total videos {}. # train videos {}. # test videos {}".format(num_videos, len(train_keys), len(test_keys)))

    print("Initialize model")
    action_space, n_action = init_action_space()

    model = ActorCritic( num_inputs = args.input_dim, num_outputs = n_action, hidden_size = args.hidden_dim)
    print("Model size: {:.5f}M".format(sum(p.numel() for p in model.parameters())/1000000.0))

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # if args.stepsize > 0:
    #     scheduler = lr_scheduler.StepLR(optimizer, step_size=args.stepsize, gamma=args.gamma)

    if args.resume:
        print("Loading checkpoint from '{}'".format(args.resume))
        checkpoint = torch.load(args.resume)
        start_epoch = checkpoint['epoch']
        optimizer.load_state_dict(checkpoint['optimizer'])
        model.load_state_dict(checkpoint['state_dict'])
    else:
        start_epoch = 0

    if use_gpu:
        model = nn.DataParallel(model).cuda()

    train_writer = SummaryWriter(os.path.join(args.save_dir, 'train'))

    if args.evaluate:
        print("Evaluate only")
        evaluate(model, dataset, test_keys, use_gpu)
        return

    print("==> Start training")
    start_time = time.time()
    model.train()
    baselines = {key: 0. for key in train_keys} # baseline rewards for videos
    reward_writers = {key: [] for key in train_keys} # record reward changes for each video


    for epoch in range(start_epoch, args.max_epoch):
        idxs = np.arange(len(train_keys))
        np.random.shuffle(idxs) # shuffle indices
        cost_all = 0

        simu_time = 0
        loading_time = 0
        back_time = 0
        for ni, idx in enumerate(idxs):

            begin_time = time.time()
            key = train_keys[idx]
            seq_feat = dataset[key]['feature'][...] # sequence of features, (seq_len, dim)
            seq = torch.from_numpy(seq_feat).unsqueeze(0) # input shape (1, seq_len, dim) because they set batch first
            # label_feat_ = dataset[key]['label_feature'][...]
            # label_feat = torch.from_numpy(label_feat_).unsqueeze(0)
            gt_param = torch.from_numpy( dataset[key]['gt_param'][...]).reshape([4,1]).type(torch.float32)

            # use random value for init param
            init_norm = torch.randn([3,1])
            init_norm /= torch.norm(init_norm)
            init_param = torch.cat([init_norm, torch.randn([1,1])],dim=0)
            # init_param = torch.cat([torch.tensor([[0.],[0.],[1.]]), torch.randn([1, 1])], dim=0)

            if use_gpu:
                seq = seq.cuda()
                label_feat = label_feat.cuda()
                gt_param = gt_param.cuda()
                init_param = init_param.cuda()

            torch.cuda.synchronize()
            loading_time += time.time() - begin_time

            simu_time_ = time.time()
            m, _ = model(init_param) # output shape (1, seq_len, 1)

            # cost = args.beta * (probs.mean() - 0.5)**2 # minimize summary length penalty term [Eq.11]
            # m = Bernoulli(probs)
            # m = Categorical(probs)
            cost = 0
            epis_rewards = []

            for _ in range(args.num_episode):
                with torch.no_grad():
                    actions = m.sample()
                log_probs = m.log_prob(actions)
                reward = compute_param_reward(gt_param,init_param, actions, action_space, use_gpu=use_gpu)
                expected_reward = (log_probs * (reward - baselines[key])).mean()
                epis_rewards.append(reward.item())
                cost -= expected_reward # minimize negative expected reward


            cost_all += cost

            train_writer.add_scalar('epis_loss', cost/args.num_episode, epoch*len(train_keys)+ni)
            train_writer.add_scalar('epis_reward',np.mean(epis_rewards), epoch*len(train_keys)+ni )

            torch.cuda.synchronize()
            simu_time += time.time() - simu_time_

            back_time_ = time.time()
            optimizer.zero_grad()
            cost.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            baselines[key] = 0.9 * baselines[key] + 0.1 * np.mean(epis_rewards) # update baseline reward via moving average
            reward_writers[key].append(np.mean(epis_rewards))
            back_time += time.time() - back_time_

            if ni % 10 == 0:
                print("epoch: {}: processing: {}/{}, avg_load_time: {}, simu_time: {}, back_time: {} ".format(
                    epoch, ni, len(idxs), loading_time, simu_time, back_time))
                simu_time = 0
                loading_time = 0
                back_time = 0


        epoch_reward = np.mean([reward_writers[key][epoch] for key in train_keys])

        train_writer.add_scalar('epoch_reward', epoch_reward, epoch)
        train_writer.add_scalar('epoch_loss', cost_all/len(train_keys)/args.num_episode, epoch)

        print("epoch {}/{}\t reward {}\t".format(epoch+1, args.max_epoch, epoch_reward))

        if epoch %10 == 0:

            #max save 3
            ckpt_list = glob(args.save_dir + '/*.pth.tar')
            ckpt_list.sort(key=lambda f: int(filter(str.isdigit, f)))
            if len(ckpt_list) > 3:
                os.remove(ckpt_list[0])

            model_state_dict = model.module.state_dict() if use_gpu else model.state_dict()
            model_save_path = osp.join(args.save_dir, 'model_epoch_' + str(epoch) + '.pth.tar')
            # save_checkpoint({model_state_dict}, model_save_path)
            save_checkpoint({
                'epoch': epoch + 1,
                # 'arch': args.arch,
                'state_dict': model.module.state_dict(),
                'optimizer': optimizer.state_dict(),
                # 'div_flow': args.div_flow
            }, model_save_path)


            print("Model saved to {}".format(model_save_path))


    write_json(reward_writers, osp.join(args.save_dir, 'rewards.json'))
    evaluate(model, dataset, test_keys, use_gpu)

    elapsed = round(time.time() - start_time)
    elapsed = str(datetime.timedelta(seconds=elapsed))
    print("Finished. Total elapsed time (h:m:s): {}".format(elapsed))



    dataset.close()

def evaluate(model, dataset, test_keys, use_gpu):
    print("==> Test")
    with torch.no_grad():
        model.eval()
        fms = []
        eval_metric = 'avg' if args.metric == 'tvsum' else 'max'

        if args.verbose: table = [["No.", "Video", "F-score"]]

        if args.save_results:
            h5_res = h5py.File(osp.join(args.save_dir, 'result.h5'), 'w')

        for key_idx, key in enumerate(test_keys):
            seq = dataset[key]['features'][...]
            seq = torch.from_numpy(seq).unsqueeze(0)
            if use_gpu: seq = seq.cuda()
            probs = model(seq)
            probs = probs.data.cpu().squeeze().numpy()

            cps = dataset[key]['change_points'][...]
            num_frames = dataset[key]['n_frames'][()]
            nfps = dataset[key]['n_frame_per_seg'][...].tolist()
            positions = dataset[key]['picks'][...]
            user_summary = dataset[key]['user_summary'][...]

            machine_summary = vsum_tools.generate_summary(probs, cps, num_frames, nfps, positions)
            fm, _, _ = vsum_tools.evaluate_summary(machine_summary, user_summary, eval_metric)
            fms.append(fm)

            if args.verbose:
                table.append([key_idx+1, key, "{:.1%}".format(fm)])

            if args.save_results:
                h5_res.create_dataset(key + '/score', data=probs)
                h5_res.create_dataset(key + '/machine_summary', data=machine_summary)
                h5_res.create_dataset(key + '/gtscore', data=dataset[key]['gtscore'][...])
                h5_res.create_dataset(key + '/fm', data=fm)

    if args.verbose:
        print(tabulate(table))

    if args.save_results: h5_res.close()

    mean_fm = np.mean(fms)
    print("Average F-score {:.1%}".format(mean_fm))

    return mean_fm

if __name__ == '__main__':
    main()
