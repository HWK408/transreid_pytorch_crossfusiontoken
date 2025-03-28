import torch
import numpy as np
import os
from utils.reranking import re_ranking
from utils.eval_sysu import eval_sysu
from utils.eval_regdb import eval_regdb
import scipy.io as sio
import logging
logger = logging.getLogger("feature extraction")

def euclidean_distance(qf, gf):
    m = qf.shape[0]
    n = gf.shape[0]
    dist_mat = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n) + \
               torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    dist_mat.addmm_(1, -2, qf, gf.t())
    return dist_mat.cpu().numpy()

def cosine_similarity(qf, gf):
    epsilon = 0.00001
    dist_mat = qf.mm(gf.t())
    qf_norm = torch.norm(qf, p=2, dim=1, keepdim=True)  # mx1
    gf_norm = torch.norm(gf, p=2, dim=1, keepdim=True)  # nx1
    qg_normdot = qf_norm.mm(gf_norm.t())

    dist_mat = dist_mat.mul(1 / qg_normdot).cpu().numpy()
    dist_mat = np.clip(dist_mat, -1 + epsilon, 1 - epsilon)
    dist_mat = np.arccos(dist_mat)
    return dist_mat


def eval_func(distmat, q_pids, g_pids, q_camids, g_camids, max_rank=50):
    """Evaluation with market1501 metric
        Key: for each query identity, its gallery images from the same camera view are discarded.
        """
    num_q, num_g = distmat.shape
    # distmat g
    #    q    1 3 2 4
    #         4 1 2 3
    if num_g < max_rank:
        max_rank = num_g
        print("Note: number of gallery samples is quite small, got {}".format(num_g))
    indices = np.argsort(distmat, axis=1)
    #  0 2 1 3
    #  1 2 3 0
    matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)
    # compute cmc curve for each query
    all_cmc = []
    all_AP = []
    num_valid_q = 0.  # number of valid query
    for q_idx in range(num_q):
        # get query pid and camid
        q_pid = q_pids[q_idx]
        q_camid = q_camids[q_idx]

        # remove gallery samples that have the same pid and camid with query
        order = indices[q_idx]  # select one row
        remove = (g_pids[order] == q_pid) & (g_camids[order] == q_camid)
        keep = np.invert(remove)

        # compute cmc curve
        # binary vector, positions with value 1 are correct matches
        orig_cmc = matches[q_idx][keep]
        if not np.any(orig_cmc):
            # this condition is true when query identity does not appear in gallery
            continue

        cmc = orig_cmc.cumsum()
        cmc[cmc > 1] = 1

        all_cmc.append(cmc[:max_rank])
        num_valid_q += 1.

        # compute average precision
        # reference: https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)#Average_precision
        num_rel = orig_cmc.sum()
        tmp_cmc = orig_cmc.cumsum()
        y = np.arange(1, tmp_cmc.shape[0] + 1) * 1.0
        tmp_cmc = tmp_cmc / y
        tmp_cmc = np.asarray(tmp_cmc) * orig_cmc
        AP = tmp_cmc.sum() / num_rel
        all_AP.append(AP)

    assert num_valid_q > 0, "Error: all query identities do not appear in gallery"

    all_cmc = np.asarray(all_cmc).astype(np.float32)
    all_cmc = all_cmc.sum(0) / num_valid_q
    mAP = np.mean(all_AP)

    return all_cmc, mAP


class R1_mAP_eval():
    def __init__(self, cfg, num_query, max_rank=50, feat_norm=True, reranking=False):
        super(R1_mAP_eval, self).__init__()
        self.num_query = num_query
        self.max_rank = max_rank
        self.feat_norm = feat_norm
        self.reranking = reranking
        self.dataset = cfg.DATASETS.NAMES

    def reset(self):
        self.gfeats = []
        self.sfeats = []
        self.feats = []
        self.pids = []
        self.camids = []
        self.imgpaths = []

    def update(self, output):  # called once for each batch
        gfeat, sfeat, feat, pid, camid, path = output
        self.gfeats.append(gfeat.cpu())
        self.sfeats.append(sfeat.cpu())
        self.feats.append(feat.cpu())
        self.pids.extend(np.asarray(pid))
        self.camids.extend(np.asarray(camid.cpu()))
        self.imgpaths.extend(np.asarray(path))

    def compute(self):  # called after each epoch
        
        ##global bn feat
        print('Global bn feature')
        gfeats = torch.cat(self.gfeats, dim=0)
        if self.feat_norm:
            print("The test feature is normalized")
            gfeats = torch.nn.functional.normalize(gfeats, dim=1, p=2)  # along channel
        # query
        qf = gfeats[:self.num_query]
        q_pids = np.asarray(self.pids[:self.num_query])
        q_camids = np.asarray(self.camids[:self.num_query])
        # gallery
        gf = gfeats[self.num_query:]
        g_pids = np.asarray(self.pids[self.num_query:])

        g_camids = np.asarray(self.camids[self.num_query:])

        q_img_paths = np.asarray(self.imgpaths[:self.num_query])
        g_img_paths = np.asarray(self.imgpaths[self.num_query:])
        # if self.reranking:
        #     print('=> Enter reranking')
        #     distmat = re_ranking(qf, gf, k1=20, k2=6, lambda_value=0.3)

        # else:
        #     print('=> Computing DistMat with euclidean_distance')
        #     distmat = euclidean_distance(qf, gf)
        # cmc0, mAP0 = eval_func(distmat, q_pids, g_pids, q_camids, g_camids)
        if self.dataset == 'sysu':
            perm = sio.loadmat(os.path.join('/data0/data_ccq/SYSU-MM01_pose/exp/', 'rand_perm_cam.mat'))['rand_perm_cam']
            mAP0, cmc0 = eval_sysu(qf, q_pids, q_camids, gf, g_pids, g_camids, g_img_paths, perm, mode='all', num_shots=1)
            eval_sysu(qf, q_pids, q_camids, gf, g_pids, g_camids, g_img_paths, perm, mode='all', num_shots=10)
            eval_sysu(qf, q_pids, q_camids, gf, g_pids, g_camids, g_img_paths, perm, mode='indoor', num_shots=1)
            eval_sysu(qf, q_pids, q_camids, gf, g_pids, g_camids, g_img_paths, perm, mode='indoor', num_shots=10)

        elif self.dataset == 'regdb':
            print('infrared to visible')
            mAP0, cmc0 = eval_regdb(qf, q_pids, q_camids, gf, g_pids, g_camids, g_img_paths)
            print('visible to infrared')
            eval_regdb(gf, g_pids, g_camids, qf, q_pids, q_camids, q_img_paths)

        ## specific bn sfeat
        print('Specific bn feature')
        sfeats = torch.cat(self.sfeats, dim=0)
        if self.feat_norm:
            print("The test feature is normalized")
            sfeats = torch.nn.functional.normalize(sfeats, dim=1, p=2)  # along channel
        # query
        qf = sfeats[:self.num_query]
        q_pids = np.asarray(self.pids[:self.num_query])
        q_camids = np.asarray(self.camids[:self.num_query])
        # gallery
        gf = sfeats[self.num_query:]
        g_pids = np.asarray(self.pids[self.num_query:])

        g_camids = np.asarray(self.camids[self.num_query:])

        if self.dataset == 'sysu':
            mAP, cmc = eval_sysu(qf, q_pids, q_camids, gf, g_pids, g_camids, g_img_paths, perm, mode='all', num_shots=1)
            eval_sysu(qf, q_pids, q_camids, gf, g_pids, g_camids, g_img_paths, perm, mode='all', num_shots=10)
            eval_sysu(qf, q_pids, q_camids, gf, g_pids, g_camids, g_img_paths, perm, mode='indoor', num_shots=1)
            eval_sysu(qf, q_pids, q_camids, gf, g_pids, g_camids, g_img_paths, perm, mode='indoor', num_shots=10)

        elif self.dataset == 'regdb':
            print('infrared to visible')
            mAP, cmc = eval_regdb(qf, q_pids, q_camids, gf, g_pids, g_camids, g_img_paths)
            print('visible to infrared')
            eval_regdb(gf, g_pids, g_camids, qf, q_pids, q_camids, q_img_paths)
        
        
        ## fusion bn feat
        print('Fusion bn feature')
        feats = torch.cat(self.feats, dim=0)
        if self.feat_norm:
            print("The test feature is normalized")
            feats = torch.nn.functional.normalize(feats, dim=1, p=2)  # along channel
        # query
        qf = feats[:self.num_query]
        q_pids = np.asarray(self.pids[:self.num_query])
        q_camids = np.asarray(self.camids[:self.num_query])
        # gallery
        gf = feats[self.num_query:]
        g_pids = np.asarray(self.pids[self.num_query:])

        g_camids = np.asarray(self.camids[self.num_query:])

        if self.dataset == 'sysu':
            mAP1, cmc1 = eval_sysu(qf, q_pids, q_camids, gf, g_pids, g_camids, g_img_paths, perm, mode='all', num_shots=1)
            eval_sysu(qf, q_pids, q_camids, gf, g_pids, g_camids, g_img_paths, perm, mode='all', num_shots=10)
            eval_sysu(qf, q_pids, q_camids, gf, g_pids, g_camids, g_img_paths, perm, mode='indoor', num_shots=1)
            eval_sysu(qf, q_pids, q_camids, gf, g_pids, g_camids, g_img_paths, perm, mode='indoor', num_shots=10)

        elif self.dataset == 'regdb':
            print('infrared to visible')
            mAP1, cmc1 = eval_regdb(qf, q_pids, q_camids, gf, g_pids, g_camids, g_img_paths)
            print('visible to infrared')
            eval_regdb(gf, g_pids, g_camids, qf, q_pids, q_camids, q_img_paths)
        

        # if self.reranking:
        #     print('=> Enter reranking')
        #     distmat = re_ranking(qf, gf, k1=20, k2=6, lambda_value=0.3)

        # else:
        #     print('=> Computing DistMat with euclidean_distance')
        #     distmat = euclidean_distance(qf, gf)
        # cmc, mAP = eval_func(distmat, q_pids, g_pids, q_camids, g_camids)

        return cmc0, mAP0, cmc, mAP, cmc1, mAP1#, distmat, self.pids, self.camids, qf, gf



