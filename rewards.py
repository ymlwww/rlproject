import torch
import sys
import numpy as np
def plane_metric(gt_param, init_param, action, w=1, lamda=5):
    a = action[0]; b = action[1];
    r = action[2]; d = action[3];
    # a= b= r= 2.*np.pi/180; d=0.05 # lead to dist ~ 0.0083, if only d=0.05 lead to dist ~0.007, r=2degree ~ 0.0067, a or b=2degree => 0.0073
    H = torch.Tensor([[torch.cos(r)*torch.cos(b) - torch.sin(r)*torch.sin(a)*torch.sin(b), -torch.cos(a)*torch.sin(r), torch.cos(r)*torch.sin(b) + torch.cos(b)*torch.sin(r)*torch.sin(a)],
                      [torch.cos(b)*torch.sin(r) + torch.cos(r)*torch.sin(a)*torch.sin(b),  torch.cos(r)*torch.cos(a), torch.sin(r)*torch.sin(b) - torch.cos(r)*torch.cos(b)*torch.sin(a)],
                      [-torch.cos(a)*torch.sin(b),                                          torch.sin(a),               torch.cos(a)*torch.cos(b)]]).cuda()
    new_normal = torch.matmul(H,init_param[:-1,:])
    new_param = torch.cat([new_normal,init_param[-1:,:]+ d ])
    dist = (1 - torch.sum(gt_param[:-1,:] * new_normal)) + w/(1+torch.exp(lamda-torch.abs(gt_param[-1:,:]-new_param[-1:,:]))).squeeze()
    return dist, new_param



# compute_param_reward(gt_param, actions, use_gpu=use_gpu)
def compute_param_reward(gt_param, init_param, actions, action_space, thresh = 0.02,  use_gpu=False):
    """
    Compute diversity reward and representativeness reward

    Args:
        seq: sequence of features, shape (1, seq_len, dim)
        actions: binary action sequence, shape (1, seq_len, 1)
        ignore_far_sim (bool): whether to ignore temporally distant similarity (default: True)
        temp_dist_thre (int): threshold for ignoring temporally distant similarity (default: 20)
        use_gpu (bool): whether to use GPU
    """
    step_reward = - torch.tensor(thresh / 4).cuda()
    _actions = actions.squeeze()#detach()
    # action_idxs = _actions.squeeze().nonzero().squeeze()
    # num_action = len(action_idxs) #if action_idxs.ndimension() > 0 else 1
    num_action = _actions.shape[-1]
    pre_dist , _ = plane_metric(gt_param, init_param, torch.Tensor([0,0,0,0]).cuda())
    init_dist = pre_dist.clone()

    reward = torch.tensor(0.)
    b_arrive = False

    if num_action == 0:
        # give zero reward is no frames are selected
        reward = torch.tensor(0.)
        if use_gpu: reward = reward.cuda()
        return reward

    # _seq = _seq.squeeze()
    # n = _seq.size(0)
    #
    # # compute diversity reward
    if num_action == 1:
        # reward =
        dist, fine_param = plane_metric(gt_param, init_param, action_space[_actions[0]])
        reward = step_reward + pre_dist - dist
        if dist <= thresh:
            reward += 10
    else:
        for i in range(num_action):
            trans_param = torch.tensor(action_space[_actions[i]]).cuda()
            dist, init_param = plane_metric(gt_param, init_param, trans_param)
            reward += step_reward + pre_dist - dist
            pre_dist = dist
            if dist <= thresh:
                if not b_arrive:
                    reward += 10
                    b_arrive = True
                else:
                    reward += - 10 * step_reward #note step reward is negative

    #     normed_seq = _seq / _seq.norm(p=2, dim=1, keepdim=True)
    #     dissim_mat = 1. - torch.matmul(normed_seq, normed_seq.t())  # dissimilarity matrix [Eq.4]
    #     dissim_submat = dissim_mat[pick_idxs, :][:, pick_idxs]
    #     if ignore_far_sim:
    #         # ignore temporally distant similarity
    #         pick_mat = pick_idxs.expand(num_picks, num_picks)
    #         temp_dist_mat = torch.abs(pick_mat - pick_mat.t())
    #         dissim_submat[temp_dist_mat > temp_dist_thre] = 1.
    #     reward_div = dissim_submat.sum() / (num_picks * (num_picks - 1.))  # diversity reward [Eq.3]
    #
    # # compute representativeness reward
    # dist_mat = torch.pow(_seq, 2).sum(dim=1, keepdim=True).expand(n, n)
    # dist_mat = dist_mat + dist_mat.t()
    # dist_mat.addmm_(1, -2, _seq, _seq.t())
    # dist_mat = dist_mat[:, pick_idxs]
    # dist_mat = dist_mat.min(1, keepdim=True)[0]
    # # reward_rep = torch.exp(torch.FloatTensor([-dist_mat.mean()]))[0] # representativeness reward [Eq.5]
    # reward_rep = torch.exp(-dist_mat.mean())
    #
    # # combine the two rewards
    if dist > thresh:
        reward -= 10 *  dist / init_dist

    return reward


# def compute_reward(seq, actions, ignore_far_sim=True, temp_dist_thre=20, use_gpu=False):
#     """
#     Compute diversity reward and representativeness reward
#
#     Args:
#         seq: sequence of features, shape (1, seq_len, dim)
#         actions: binary action sequence, shape (1, seq_len, 1)
#         ignore_far_sim (bool): whether to ignore temporally distant similarity (default: True)
#         temp_dist_thre (int): threshold for ignoring temporally distant similarity (default: 20)
#         use_gpu (bool): whether to use GPU
#     """
#     _seq = seq.detach()
#     _actions = actions.detach()
#     pick_idxs = _actions.squeeze().nonzero().squeeze()
#     num_picks = len(pick_idxs) if pick_idxs.ndimension() > 0 else 1
#
#     if num_picks == 0:
#         # give zero reward is no frames are selected
#         reward = torch.tensor(0.)
#         if use_gpu: reward = reward.cuda()
#         return reward
#
#     _seq = _seq.squeeze()
#     n = _seq.size(0)
#
#     # compute diversity reward
#     if num_picks == 1:
#         reward_div = torch.tensor(0.)
#         if use_gpu: reward_div = reward_div.cuda()
#     else:
#         normed_seq = _seq / _seq.norm(p=2, dim=1, keepdim=True)
#         dissim_mat = 1. - torch.matmul(normed_seq, normed_seq.t()) # dissimilarity matrix [Eq.4]
#         dissim_submat = dissim_mat[pick_idxs,:][:,pick_idxs]
#         if ignore_far_sim:
#             # ignore temporally distant similarity
#             pick_mat = pick_idxs.expand(num_picks, num_picks)
#             temp_dist_mat = torch.abs(pick_mat - pick_mat.t())
#             dissim_submat[temp_dist_mat > temp_dist_thre] = 1.
#         reward_div = dissim_submat.sum() / (num_picks * (num_picks - 1.)) # diversity reward [Eq.3]
#
#     # compute representativeness reward
#     dist_mat = torch.pow(_seq, 2).sum(dim=1, keepdim=True).expand(n, n)
#     dist_mat = dist_mat + dist_mat.t()
#     dist_mat.addmm_(1, -2, _seq, _seq.t())
#     dist_mat = dist_mat[:,pick_idxs]
#     dist_mat = dist_mat.min(1, keepdim=True)[0]
#     #reward_rep = torch.exp(torch.FloatTensor([-dist_mat.mean()]))[0] # representativeness reward [Eq.5]
#     reward_rep = torch.exp(-dist_mat.mean())
#
#     # combine the two rewards
#     reward = (reward_div + reward_rep) * 0.5
#
#     return reward
