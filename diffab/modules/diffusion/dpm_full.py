import torch
import torch.nn as nn
import torch.nn.functional as F
import functools
from tqdm.auto import tqdm

from diffab.modules.common.geometry import apply_rotation_to_vector, quaternion_1ijk_to_rotation_matrix
from diffab.modules.common.so3 import so3vec_to_rotation, rotation_to_so3vec, random_uniform_so3
from diffab.modules.encoders.ga import GAEncoder
from diffab.tools.ddg.predictor import predict_ddg_batch
from diffab.tools.eval.hydropathy import hydropathy_vector
from .transition import RotationTransition, PositionTransition, AminoacidCategoricalTransition


def rotation_matrix_cosine_loss(R_pred, R_true):
    """
    Args:
        R_pred: (*, 3, 3).
        R_true: (*, 3, 3).
    Returns:
        Per-matrix losses, (*, ).
    """
    size = list(R_pred.shape[:-2])
    ncol = R_pred.numel() // 3

    RT_pred = R_pred.transpose(-2, -1).reshape(ncol, 3) # (ncol, 3)
    RT_true = R_true.transpose(-2, -1).reshape(ncol, 3) # (ncol, 3)

    ones = torch.ones([ncol, ], dtype=torch.long, device=R_pred.device)
    loss = F.cosine_embedding_loss(RT_pred, RT_true, ones, reduction='none')  # (ncol*3, )
    loss = loss.reshape(size + [3]).sum(dim=-1)    # (*, )
    return loss


class EpsilonNet(nn.Module):

    def __init__(self, res_feat_dim, pair_feat_dim, num_layers, encoder_opt={}):
        super().__init__()
        self.current_sequence_embedding = nn.Embedding(25, res_feat_dim)  # 22 is padding
        self.res_feat_mixer = nn.Sequential(
            nn.Linear(res_feat_dim * 2, res_feat_dim), nn.ReLU(),
            nn.Linear(res_feat_dim, res_feat_dim),
        )
        self.encoder = GAEncoder(res_feat_dim, pair_feat_dim, num_layers, **encoder_opt)

        self.eps_crd_net = nn.Sequential(
            nn.Linear(res_feat_dim+3, res_feat_dim), nn.ReLU(),
            nn.Linear(res_feat_dim, res_feat_dim), nn.ReLU(),
            nn.Linear(res_feat_dim, 3)
        )

        self.eps_rot_net = nn.Sequential(
            nn.Linear(res_feat_dim+3, res_feat_dim), nn.ReLU(),
            nn.Linear(res_feat_dim, res_feat_dim), nn.ReLU(),
            nn.Linear(res_feat_dim, 3)
        )

        self.eps_seq_net = nn.Sequential(
            nn.Linear(res_feat_dim+3, res_feat_dim), nn.ReLU(),
            nn.Linear(res_feat_dim, res_feat_dim), nn.ReLU(),
            nn.Linear(res_feat_dim, 20), nn.Softmax(dim=-1) 
        )

    def forward(self, v_t, p_t, s_t, res_feat, pair_feat, beta, mask_generate, mask_res):
        """
        Args:
            v_t:    (N, L, 3).
            p_t:    (N, L, 3).
            s_t:    (N, L).
            res_feat:   (N, L, res_dim).
            pair_feat:  (N, L, L, pair_dim).
            beta:   (N,).
            mask_generate:    (N, L).
            mask_res:       (N, L).
        Returns:
            v_next: UPDATED (not epsilon) SO3-vector of orietnations, (N, L, 3).
            eps_pos: (N, L, 3).
        """
        N, L = mask_res.size()
        R = so3vec_to_rotation(v_t) # (N, L, 3, 3)

        # s_t = s_t.clamp(min=0, max=19)  # TODO: clamping is good but ugly.
        res_feat = self.res_feat_mixer(torch.cat([res_feat, self.current_sequence_embedding(s_t)], dim=-1)) # [Important] Incorporate sequence at the current step.
        res_feat = self.encoder(R, p_t, res_feat, pair_feat, mask_res)

        t_embed = torch.stack([beta, torch.sin(beta), torch.cos(beta)], dim=-1)[:, None, :].expand(N, L, 3)
        in_feat = torch.cat([res_feat, t_embed], dim=-1)

        # Position changes
        eps_crd = self.eps_crd_net(in_feat)    # (N, L, 3)
        eps_pos = apply_rotation_to_vector(R, eps_crd)  # (N, L, 3)
        eps_pos = torch.where(mask_generate[:, :, None].expand_as(eps_pos), eps_pos, torch.zeros_like(eps_pos))

        # New orientation
        eps_rot = self.eps_rot_net(in_feat)    # (N, L, 3)
        U = quaternion_1ijk_to_rotation_matrix(eps_rot) # (N, L, 3, 3)
        R_next = R @ U
        v_next = rotation_to_so3vec(R_next)     # (N, L, 3)
        v_next = torch.where(mask_generate[:, :, None].expand_as(v_next), v_next, v_t)

        # New sequence categorical distributions
        c_denoised = self.eps_seq_net(in_feat)  # Already softmax-ed, (N, L, 20)

        return v_next, R_next, eps_pos, c_denoised


class FullDPM(nn.Module):

    def __init__(
        self, 
        res_feat_dim, 
        pair_feat_dim, 
        num_steps, 
        eps_net_opt={}, 
        trans_rot_opt={}, 
        trans_pos_opt={}, 
        trans_seq_opt={},
        position_mean=[0.0, 0.0, 0.0],
        position_scale=[10.0],
        prior_b=0,
    ):
        super().__init__()
        self.eps_net = EpsilonNet(res_feat_dim, pair_feat_dim, **eps_net_opt)
        self.num_steps = num_steps
        self.trans_rot = RotationTransition(num_steps, **trans_rot_opt)
        self.trans_pos = PositionTransition(num_steps, **trans_pos_opt)
        self.trans_seq = AminoacidCategoricalTransition(num_steps, **trans_seq_opt, prior_b=prior_b)

        self.register_buffer('position_mean', torch.FloatTensor(position_mean).view(1, 1, -1))
        self.register_buffer('position_scale', torch.FloatTensor(position_scale).view(1, 1, -1))
        self.register_buffer('_dummy', torch.empty([0, ]))

    def _normalize_position(self, p):
        p_norm = (p - self.position_mean) / self.position_scale
        return p_norm

    def _unnormalize_position(self, p_norm):
        p = p_norm * self.position_scale + self.position_mean
        return p

    def forward(self, v_0, p_0, s_0, res_feat, pair_feat, mask_generate, mask_res, denoise_structure, denoise_sequence, t=None):
        N, L = res_feat.shape[:2]
        if t == None:
            t = torch.randint(0, self.num_steps, (N,), dtype=torch.long, device=self._dummy.device)
        p_0 = self._normalize_position(p_0)

        if denoise_structure:
            # Add noise to rotation
            R_0 = so3vec_to_rotation(v_0)
            v_noisy, _ = self.trans_rot.add_noise(v_0, mask_generate, t)
            # Add noise to positions
            p_noisy, eps_p = self.trans_pos.add_noise(p_0, mask_generate, t)
        else:
            R_0 = so3vec_to_rotation(v_0)
            v_noisy = v_0.clone()
            p_noisy = p_0.clone()
            eps_p = torch.zeros_like(p_noisy)

        if denoise_sequence:
            # Add noise to sequence
            _, s_noisy = self.trans_seq.add_noise(s_0, mask_generate, t)
        else:
            s_noisy = s_0.clone()

        beta = self.trans_pos.var_sched.betas[t]
        v_pred, R_pred, eps_p_pred, c_denoised = self.eps_net(
            v_noisy, p_noisy, s_noisy, res_feat, pair_feat, beta, mask_generate, mask_res
        )   # (N, L, 3), (N, L, 3, 3), (N, L, 3), (N, L, 20), (N, L)

        loss_dict = {}

        # Rotation loss
        loss_rot = rotation_matrix_cosine_loss(R_pred, R_0) # (N, L)
        loss_rot = (loss_rot * mask_generate).sum() / (mask_generate.sum().float() + 1e-8)
        loss_dict['rot'] = loss_rot

        # Position loss
        loss_pos = F.mse_loss(eps_p_pred, eps_p, reduction='none').sum(dim=-1)  # (N, L)
        loss_pos = (loss_pos * mask_generate).sum() / (mask_generate.sum().float() + 1e-8)
        loss_dict['pos'] = loss_pos

        # Sequence categorical loss
        post_true = self.trans_seq.posterior(s_noisy, s_0, t)
        log_post_pred = torch.log(self.trans_seq.posterior(s_noisy, c_denoised, t) + 1e-8)
        kldiv = F.kl_div(
            input=log_post_pred, 
            target=post_true, 
            reduction='none',
            log_target=False
        ).sum(dim=-1)    # (N, L)
        loss_seq = (kldiv * mask_generate).sum() / (mask_generate.sum().float() + 1e-8)
        loss_dict['seq'] = loss_seq

        return loss_dict

    @staticmethod
    def _get_argmin(x):
        return torch.min(x, dim=-1).indices

    @staticmethod
    def _get_argmax(x):
        return torch.max(x, dim=-1).indices

    @staticmethod
    def _sample_multinomial(x):
        return torch.multinomial(x, 1).flatten()

    @staticmethod
    def _sample_multinomial_sched(prob, beta):
        num_gen = prob.shape[-1]
        x = beta*prob + (1-beta)/num_gen
        x = x / (x.sum(dim=-1, keepdim=True) + 1e-8)
        return torch.multinomial(x, 1).flatten()

    def _choose_next_sample(self, store_next, metric_sampled, mode="min", beta=None):
        store_v, store_p, store_s = store_next

        # Stack tensors
        store_v = torch.stack(store_v)  # num samples x batch x seq len x feats
        store_p = torch.stack(store_p)
        store_s = torch.stack(store_s)
        # `metric_sampled` is a dictionary with metrics (add up metrics)
        metric_sampled = torch.stack(
            [torch.stack(item).T for item in metric_sampled.values()]
        ).sum(0)  # batch x num samples

        if mode == "min":
            # Get sample with minimum metric
            selected = self._get_argmin(metric_sampled)

        elif mode == "max":
            # Get sample with maximum metric
            selected = self._get_argmax(metric_sampled)

        elif mode == "exp":
            # Shift sampled metric to avoid negative values
            metric_sampled_min = torch.abs(metric_sampled.min(dim=-1, keepdim=True).values)
            metric_sampled_shift = metric_sampled + metric_sampled_min
            # Get probabilities (using exponential distribution)
            lambdas = (1 / metric_sampled_min)
            exp_distrib = lambdas * torch.exp(-lambdas*metric_sampled_shift)
            prob = exp_distrib / (exp_distrib.sum(dim=-1, keepdim=True) + 1e-8)
            # Sample
            selected = self._sample_multinomial(prob)

        elif mode == "softmax":
            # Get probabilities (using softmax) and sample
            prob = F.softmax(-metric_sampled, dim=-1)
            selected = self._sample_multinomial(prob)

        elif mode == "softmax_beta":
            # Get probabilities (using softmax) and sample
            prob = F.softmax(-metric_sampled, dim=-1)
            selected = self._sample_multinomial_sched(prob, beta)

        else:
            raise NotImplementedError(
                "Sample step modes are `min`, `max`, `exp`, `softmax` or `softmax_beta`!"
            )

        # Select
        num_idx = torch.arange(len(selected), device=selected.device)
        v_next = store_v[selected, num_idx]
        p_next = store_p[selected, num_idx]
        s_next = store_s[selected, num_idx]

        return v_next, p_next, s_next

    @torch.no_grad()
    def sample(
        self,
        batch_ref, v, p, s,
        res_feat, pair_feat,
        mask_generate, mask_res,
        sample_structure=True, sample_sequence=True, sample_predict_ddg=True,
        sample_step_by_ddg=False, sample_step_by_hydro=False,
        sample_step_mode="min", sample_step_num=20, sample_step_period=1,
        pbar=False,
    ):
        """
        Args:
            v:  Orientations of contextual residues, (N, L, 3).
            p:  Positions of contextual residues, (N, L, 3).
            s:  Sequence of contextual residues, (N, L).
        """
        N, L = v.shape[:2]
        p = self._normalize_position(p)

        # Set the orientation and position of residues to be predicted to random values
        if sample_structure:
            v_rand = random_uniform_so3([N, L], device=self._dummy.device)
            p_rand = torch.randn_like(p)
            v_init = torch.where(mask_generate[:, :, None].expand_as(v), v_rand, v)
            p_init = torch.where(mask_generate[:, :, None].expand_as(p), p_rand, p)
        else:
            v_init, p_init = v, p

        if sample_sequence:
            # s_rand = torch.randint_like(s, low=0, high=19) # uniform prior (it should be high=20)
            s_rand = self.trans_seq.prior(s) # multinomial prior
            s_init = torch.where(mask_generate, s_rand, s)
        else:
            s_init = s

        p_init = self._unnormalize_position(p_init)
        traj = {self.num_steps: (v_init, p_init, s_init)}

        # Get predicted ddG at init
        if sample_predict_ddg:
            pred_ddg = predict_ddg_batch(batch_ref, s_init, p_init)
            traj[self.num_steps] += (pred_ddg,)

        if pbar:
            pbar = functools.partial(tqdm, total=self.num_steps, desc='Sampling')
        else:
            pbar = lambda x: x

        for t in pbar(range(self.num_steps, 0, -1)):
            v_t, p_t, s_t = traj[t][:3]
            p_t = self._normalize_position(p_t)

            beta = self.trans_pos.var_sched.betas[t].expand([N, ])
            t_tensor = torch.full([N, ], fill_value=t, dtype=torch.long, device=self._dummy.device)

            step_sample_times = 1
            compute_ddg_step = False
            compute_hydro_step = False

            # Every `sample_step_period` timesteps:
            # Get `sample_step_num` samples and compute metric (ddG with respect to previous step)
            if ((t-1) % sample_step_period == 0):
                if sample_step_by_ddg or sample_step_by_hydro:
                    step_sample_times = sample_step_num
                    metric_sampled, store_v, store_p, store_s = dict(), [], [], []

                if sample_step_by_ddg:
                    compute_ddg_step = True
                    metric_sampled["ddG"] = []

                if sample_step_by_hydro:
                    compute_hydro_step = True
                    metric_sampled["hydro"] = []

            if pbar and (compute_ddg_step or compute_hydro_step):
                pbar_internal = functools.partial(tqdm, total=step_sample_times, desc='Sampling by ddG/hydro')
            else:
                pbar_internal = lambda x: x

            # Common to all samples (predictions)
            v_pred, R_pred, eps_p_pred, c_denoised = self.eps_net(
                v_t, p_t, s_t, res_feat, pair_feat, beta, mask_generate, mask_res
            )   # (N, L, 3), (N, L, 3, 3), (N, L, 3)

            for _ in pbar_internal(range(step_sample_times)):  # 1 or 20 (sample by metric)
                v_next = self.trans_rot.denoise(v_t, v_pred, mask_generate, t_tensor)
                p_next = self.trans_pos.denoise(p_t, eps_p_pred, mask_generate, t_tensor)
                _, s_next = self.trans_seq.denoise(s_t, c_denoised, mask_generate, t_tensor)

                if not sample_structure:
                    v_next, p_next = v_t, p_t
                if not sample_sequence:
                    s_next = s_t

                p_next = self._unnormalize_position(p_next)

                if compute_ddg_step:
                    # Compute predicted ddG with respect to previous states
                    _, p_prev, s_prev = traj[t][:3]
                    metric_sampled["ddG"].append(predict_ddg_batch(
                        batch_ref, s_next, p_next,
                        s_prev.to(self._dummy.device),
                        p_prev.to(self._dummy.device)
                    ))

                if compute_hydro_step:
                    # Compute average hydropathy score from generated sequence
                    hydro_tensor = torch.tensor(hydropathy_vector, device=s_next.device)
                    s_cdr = s_next[mask_generate].reshape(N, -1)  # select cdr
                    s_hydro = hydro_tensor[s_cdr]  # convert residues to hydro values
                    metric_sampled["hydro"].append(s_hydro.mean(-1))

                if compute_ddg_step or compute_hydro_step:  # store states
                    store_v.append(v_next)
                    store_p.append(p_next)
                    store_s.append(s_next)

            if compute_ddg_step or compute_hydro_step:
                # Choose next sample
                v_next, p_next, s_next = self._choose_next_sample(
                    (store_v, store_p, store_s), metric_sampled,
                    mode=sample_step_mode, beta=beta[:, None],
                )

            traj[t-1] = (v_next, p_next, s_next)

            # Get predicted ddG every 10 timesteps
            if ((t-1) % 10 == 0) and sample_predict_ddg:
                pred_ddg = predict_ddg_batch(batch_ref, s_next, p_next)
                traj[t-1] += (pred_ddg,)

            traj[t] = tuple(x.cpu() for x in traj[t])    # Move previous states to cpu memory.

        return traj

    @torch.no_grad()
    def optimize(
        self, 
        v, p, s, 
        opt_step: int,
        res_feat, pair_feat, 
        mask_generate, mask_res, 
        sample_structure=True, sample_sequence=True,
        pbar=False,
    ):
        """
        Description:
            First adds noise to the given structure, then denoises it.
        """
        N, L = v.shape[:2]
        p = self._normalize_position(p)
        t = torch.full([N, ], fill_value=opt_step, dtype=torch.long, device=self._dummy.device)

        # Set the orientation and position of residues to be predicted to random values
        if sample_structure:
            # Add noise to rotation
            v_noisy, _ = self.trans_rot.add_noise(v, mask_generate, t)
            # Add noise to positions
            p_noisy, _ = self.trans_pos.add_noise(p, mask_generate, t)
            v_init = torch.where(mask_generate[:, :, None].expand_as(v), v_noisy, v)
            p_init = torch.where(mask_generate[:, :, None].expand_as(p), p_noisy, p)
        else:
            v_init, p_init = v, p

        if sample_sequence:
            _, s_noisy = self.trans_seq.add_noise(s, mask_generate, t)
            s_init = torch.where(mask_generate, s_noisy, s)
        else:
            s_init = s

        traj = {opt_step: (v_init, self._unnormalize_position(p_init), s_init)}
        if pbar:
            pbar = functools.partial(tqdm, total=opt_step, desc='Optimizing')
        else:
            pbar = lambda x: x
        for t in pbar(range(opt_step, 0, -1)):
            v_t, p_t, s_t = traj[t]
            p_t = self._normalize_position(p_t)
            
            beta = self.trans_pos.var_sched.betas[t].expand([N, ])
            t_tensor = torch.full([N, ], fill_value=t, dtype=torch.long, device=self._dummy.device)

            v_next, R_next, eps_p, c_denoised = self.eps_net(
                v_t, p_t, s_t, res_feat, pair_feat, beta, mask_generate, mask_res
            )   # (N, L, 3), (N, L, 3, 3), (N, L, 3)

            v_next = self.trans_rot.denoise(v_t, v_next, mask_generate, t_tensor)
            p_next = self.trans_pos.denoise(p_t, eps_p, mask_generate, t_tensor)
            _, s_next = self.trans_seq.denoise(s_t, c_denoised, mask_generate, t_tensor)

            if not sample_structure:
                v_next, p_next = v_t, p_t
            if not sample_sequence:
                s_next = s_t

            traj[t-1] = (v_next, self._unnormalize_position(p_next), s_next)
            traj[t] = tuple(x.cpu() for x in traj[t])    # Move previous states to cpu memory.

        return traj
