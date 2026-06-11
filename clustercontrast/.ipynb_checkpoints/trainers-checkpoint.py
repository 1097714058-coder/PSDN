from __future__ import print_function, absolute_import
import time
from .utils.meters import AverageMeter
import torch.nn as nn
import torch
from torch.nn import functional as F
from torch.nn import Module
import collections
from torch import einsum
from torch.autograd import Variable
from clustercontrast.models.cm import ClusterMemory
from clustercontrast.utils.faiss_rerank import compute_jaccard_distance,compute_ranked_list,compute_ranked_list_cm
import numpy as np
part=5
def normalize(x, axis=-1):
    """Normalizing to unit length along the specified dimension.
    Args:
      x: pytorch Variable
    Returns:
      x: pytorch Variable, same shape as input
    """
    x = 1. * x / (torch.norm(x, 2, axis, keepdim=True).expand_as(x) + 1e-12)
    return x

class KLDivLoss(nn.Module):
    def __init__(self):
        super(KLDivLoss, self).__init__()
    def forward(self, pred, label):
        # pred: 2D matrix (batch_size, num_classes)
        # label: 1D vector indicating class number
        T=3

        predict = F.log_softmax(pred/T,dim=1)
        target_data = F.softmax(label/T,dim=1)
        target_data =target_data+10**(-7)
        target = Variable(target_data.data.cuda(),requires_grad=False)
        loss=T*T*((target*(target.log()-predict)).sum(1).sum()/target.size()[0])
        return loss

def compute_cross_agreement_dd(features_g, features_p,features_g_s,features_p_s, k=20, search_option=3):
    print("Compute cross agreement score...")
    N, D = features_p.size()
    M, D = features_g.size()
    score = torch.FloatTensor()
    end = time.time()
    ranked_list_g = compute_ranked_list_cm(features_g,features_p, k=k, search_option=search_option, verbose=False)
    ranked_list_p_i = compute_ranked_list_cm(features_p_s,features_p_s, k=k, search_option=search_option, verbose=False)
    score_all =[]
    for i in range(M):
        intersect_i = torch.FloatTensor(
            [len(np.intersect1d(ranked_list_g[i], ranked_list_p_i[j])) for j in range(N)])
        union_i = torch.FloatTensor(
            [len(np.union1d(ranked_list_g[i], ranked_list_p_i[j])) for j in range(N)])
        score_i = intersect_i / union_i
        score_all.append(score_i)
    score = torch.cat(score_all, dim=0)
    # print(score_i.size())
    # print("Cross agreement score time cost: {}".format(time.time() - end))
    return score_i



class ClusterContrastTrainer_HSDN(object):
    def __init__(self, encoder, memory=None,matcher_rgb = None,matcher_ir = None):
        super(ClusterContrastTrainer_HSDN, self).__init__()
        self.encoder = encoder
        self.memory_ir = memory
        self.memory_rgb = memory
        self.wise_memory_ir =  memory
        self.wise_memory_rgb =  memory
        self.nameMap_ir =[]
        self.nameMap_rgb = []
        self.criterion_kl = KLDivLoss()
        self.cmlabel=0
        self.memory_ir_s = memory
        self.memory_rgb_s = memory
        self.wise_memory_ir_s =  memory
        self.wise_memory_rgb_s =  memory
        self.shared_memory =  memory
        self.shared_memory_s =  memory
        self.htsd=0

        self.hm=0
        self.ht=0


    def train(self, epoch, data_loader_ir,data_loader_rgb, optimizer, print_freq=10, train_iters=400):
        self.encoder.train()


        batch_time = AverageMeter()
        data_time = AverageMeter()

        losses = AverageMeter()

        loss_ir_log = AverageMeter()
        loss_rgb_log = AverageMeter()
        loss_camera_rgb_log = AverageMeter()
        loss_camera_ir_log = AverageMeter()
        ir_rgb_loss_log = AverageMeter()
        rgb_ir_loss_log = AverageMeter()
        rgb_rgb_loss_log = AverageMeter()
        ir_ir_loss_log = AverageMeter()
        loss_ins_ir_log = AverageMeter()
        loss_ins_rgb_log = AverageMeter()


        lamda_s_neibor=0.5
        lamda_d_neibor=1


        lamda_sd = 1
        lamda_c = 0.1#0.1

        end = time.time()
        for i in range(train_iters):

            # load data
            inputs_ir = data_loader_ir.next()
            inputs_rgb = data_loader_rgb.next()
            data_time.update(time.time() - end)


            inputs_ir,labels_ir, indexes_ir,cids_ir,name_ir = self._parse_data_ir(inputs_ir) #inputs_ir1
            inputs_rgb,inputs_rgb1, labels_rgb, indexes_rgb,cids_rgb,name_rgb = self._parse_data_rgb(inputs_rgb)
            
            targets_soft_rgb = []
            targets_soft_ir = []
            weights_rgb = []
            weights_ir = []
            

            for idx in indexes_rgb:
                targets_soft_rgb.append(self.correlations_rgb[idx.item()])
                weights_rgb.append(self.weights_rgb[idx.item()])
            for idx in indexes_ir:
                targets_soft_ir.append(self.correlations_ir[idx.item()])
                weights_ir.append(self.weights_ir[idx.item()])

            weights_rgb = torch.tensor(weights_rgb).cuda()
            weights_ir = torch.tensor(weights_ir).cuda()



            targets_soft_rgb = torch.stack(targets_soft_rgb, dim=0)
            targets_soft_ir = torch.stack(targets_soft_ir, dim=0)

            # forward
            inputs_rgb = torch.cat((inputs_rgb,inputs_rgb1),0)
            labels_rgb = torch.cat((labels_rgb,labels_rgb),-1)
            cids_rgb =  torch.cat((cids_rgb,cids_rgb),-1)
            targets_soft_rgb = torch.cat((targets_soft_rgb, targets_soft_rgb), 0)
            weights_rgb = torch.cat((weights_rgb, weights_rgb), -1)
 
            # inputs_ir = torch.cat((inputs_ir,inputs_ir1),0)
            # labels_ir = torch.cat((labels_ir,labels_ir),-1)
            # cids_ir =  torch.cat((cids_ir,cids_ir),-1)
    

            indexes_ir = torch.tensor([self.nameMap_ir[name] for name in name_ir]).cuda()
            indexes_rgb = torch.tensor([self.nameMap_rgb[name] for name in name_rgb])
            indexes_rgb = torch.cat((indexes_rgb,indexes_rgb),-1).cuda()

            _,f_out_rgb,f_out_ir,f_out_rgb_s,f_out_ir_s,labels_rgb,labels_ir,\
            cid_rgb,cid_ir,index_rgb,index_ir = self._forward(inputs_rgb,inputs_ir,label_1=labels_rgb,label_2=labels_ir,modal=0,\
                cid_rgb=cids_rgb,cid_ir=cids_ir,index_rgb=indexes_rgb,index_ir=indexes_ir)
            
            feats_ir = self.memory_ir(f_out_ir, labels_ir, return_out=True)
            feats_rgb = self.memory_rgb(f_out_rgb, labels_rgb, return_out=True)


            if feats_rgb.size(1) < targets_soft_rgb.size(1):
                feats_rgb = F.pad(feats_rgb, (0, targets_soft_rgb.size(1) - feats_rgb.size(1)))
                
            targets_soft_rgb = (self.balance * torch.zeros(feats_rgb.size()).scatter_(1, labels_rgb.unsqueeze(1).data.cpu(), 1) + (1.0 - self.balance) * targets_soft_rgb).cuda()
            targets_soft_ir = (self.balance * torch.zeros(feats_ir.size()).scatter_(1, labels_ir.unsqueeze(1).data.cpu(), 1) + (1.0 - self.balance) * targets_soft_ir).cuda()
            
            log_probs_ir = nn.LogSoftmax(dim=1)(feats_ir)
            log_probs_rgb = nn.LogSoftmax(dim=1)(feats_rgb)



            loss_ir_s = torch.tensor([0.]).cuda()
            loss_rgb_s = torch.tensor([0.]).cuda()
            loss_ir = torch.tensor([0.]).cuda()
            loss_rgb = torch.tensor([0.]).cuda()
            ir_rgb_loss = torch.tensor([0.]).cuda()
            rgb_ir_loss = torch.tensor([0.]).cuda()
            rgb_rgb_loss = torch.tensor([0.]).cuda()
            ir_ir_loss = torch.tensor([0.]).cuda()

            ir_rgb_loss_s = torch.tensor([0.]).cuda()
            rgb_ir_loss_s = torch.tensor([0.]).cuda()
            rgb_rgb_loss_s = torch.tensor([0.]).cuda()
            ir_ir_loss_s = torch.tensor([0.]).cuda()
            loss_shared = torch.tensor([0.]).cuda()
            loss_shared_s = torch.tensor([0.]).cuda()


            loss_ir_s = lamda_sd* self.memory_ir_s(f_out_ir_s, labels_ir)
            loss_rgb_s = lamda_sd* self.memory_rgb_s(f_out_rgb_s, labels_rgb)
            if epoch>=self.cmlabel:
                loss_ir = ((- targets_soft_ir * log_probs_ir).sum(1) * weights_ir).mean()
                loss_rgb = ((- targets_soft_rgb * log_probs_rgb).sum(1) * weights_rgb).mean()
            else:
                loss_ir = self.memory_ir(f_out_ir, labels_ir)
                loss_rgb = self.memory_rgb(f_out_rgb, labels_rgb)



            thresh=0.9
            hm_thresh = 0.9

################cpsrefine
# #############v2
            if epoch>=self.hm:#self.cmlabel:
                if epoch>=self.ht:#self.cmlabel:
##########################V4 wise neighbor
                    if epoch %2 ==0:
                        with torch.no_grad():
                            sim_prob_rgb_ir = self.wise_memory_rgb_s.features.detach()[index_rgb].mm(self.wise_memory_ir_s.features.detach().data.t())#F.softmax(F.normalize(f_out_rgb_s, dim=1).mm(self.wise_memory_ir_s.features.detach().data.t())/0.05,dim=1)#B N
                            sim_rgb_ir = self.wise_memory_rgb.features.detach()[index_rgb].mm(self.wise_memory_ir.features.detach().data.t())#F.softmax(F.normalize(f_out_rgb, dim=1).mm(self.wise_memory_ir.features.detach().data.t())/0.05,dim=1)

                            nearest_rgb_ir = sim_rgb_ir.max(dim=1, keepdim=True)[0]
                            nearest_prob_rgb_ir = sim_prob_rgb_ir.max(dim=1, keepdim=True)[0]
                            mask_neighbor_rgb_ir = torch.gt(sim_rgb_ir, nearest_rgb_ir * thresh).detach().data#nearest_intra * self.neighbor_eps)self.neighbor_eps
                            mask_neighbor_prob_rgb_ir = torch.gt(sim_prob_rgb_ir, nearest_prob_rgb_ir * thresh)#.cuda()#nearest_intra * self.neighbor_eps)self.neighbor_eps
                            num_neighbor_rgb_ir = mask_neighbor_rgb_ir.mul(mask_neighbor_prob_rgb_ir).sum(dim=1)+1


                        sim_rgb_ir = F.normalize(f_out_rgb, dim=1).mm(self.wise_memory_ir.features.detach().data.t())#F.softmax(F.normalize(f_out_rgb, dim=1).mm(self.wise_memory_ir.features.detach().data.t())/0.05,dim=1)
                        sim_prob_rgb_ir = F.normalize(f_out_rgb_s, dim=1).mm(self.wise_memory_ir_s.features.detach().data.t())
                        sim_rgb_ir_exp =sim_rgb_ir /0.05  # 64*13638
                        score_intra_rgb_ir =   F.softmax(sim_rgb_ir_exp,dim=1)##sim_exp_intra / sim_exp_intra.sum(dim=1, keepdim=True)#
                        # print('score_intra',score_intra)
                        score_intra_rgb_ir = score_intra_rgb_ir.clamp_min(1e-8)
                        # count_rgb_ir = (mask_neighbor_rgb_ir).sum(dim=1)
                        rgb_ir_loss = -score_intra_rgb_ir.log().mul(mask_neighbor_rgb_ir).mul(mask_neighbor_prob_rgb_ir).sum(dim=1)#.mul(sim_wise) mul(mask_neighbor_intra) .mul(mask_neighbor_intra)
                        rgb_ir_loss = 0.1*lamda_d_neibor*rgb_ir_loss.div(num_neighbor_rgb_ir).mean()#.mul(rgb_ir_ca).mul(rgb_ir_ca).mul(rgb_ir_ca).mul(rgb_ir_ca).mul(mask_neighbor_intra_soft) ##.mul(rgb_ir_ca)
                        # print('rgb_ir_loss',rgb_ir_loss)
                        sim_prob_rgb_ir_exp =sim_prob_rgb_ir /0.05  # 64*13638
                        score_intra_rgb_ir_s =   F.softmax(sim_prob_rgb_ir_exp,dim=1)
                        score_intra_rgb_ir_s = score_intra_rgb_ir_s.clamp_min(1e-8)

                        rgb_ir_loss_s = -score_intra_rgb_ir_s.log().mul(mask_neighbor_rgb_ir).mul(mask_neighbor_prob_rgb_ir).sum(dim=1)#.mul(sim_wise) mul(mask_neighbor_intra) .mul(mask_neighbor_intra)

                        rgb_ir_loss_s = 0.1*lamda_s_neibor*rgb_ir_loss_s.div(num_neighbor_rgb_ir).mean()#.mul(rgb_ir_ca).mul(rgb_ir_ca) ##.mul(rgb_ir_ca)


                    else:
                        with torch.no_grad():
                            sim_prob_rgb_ir = self.wise_memory_ir_s.features.detach()[index_ir].mm(self.wise_memory_rgb_s.features.detach().data.t())#F.softmax(F.normalize(f_out_rgb_s, dim=1).mm(self.wise_memory_ir_s.features.detach().data.t())/0.05,dim=1)#B N
                            sim_rgb_ir = self.wise_memory_ir.features.detach()[index_ir].mm(self.wise_memory_rgb.features.detach().data.t())#F.softmax(F.normalize(f_out_rgb, dim=1).mm(self.wise_memory_ir.features.detach().data.t())/0.05,dim=1)

                            nearest_rgb_ir = sim_rgb_ir.max(dim=1, keepdim=True)[0]
                            nearest_prob_rgb_ir = sim_prob_rgb_ir.max(dim=1, keepdim=True)[0]
                            mask_neighbor_rgb_ir = torch.gt(sim_rgb_ir, nearest_rgb_ir * thresh).detach().data#nearest_intra * self.neighbor_eps)self.neighbor_eps
                            mask_neighbor_prob_rgb_ir = torch.gt(sim_prob_rgb_ir, nearest_prob_rgb_ir * thresh)#.cuda()#nearest_intra * self.neighbor_eps)self.neighbor_eps
                            num_neighbor_rgb_ir = mask_neighbor_rgb_ir.mul(mask_neighbor_prob_rgb_ir).sum(dim=1)+1


                        sim_prob_rgb_ir = F.normalize(f_out_ir_s, dim=1).mm(self.wise_memory_rgb_s.features.detach().data.t())#F.softmax(F.normalize(f_out_rgb_s, dim=1).mm(self.wise_memory_ir_s.features.detach().data.t())/0.05,dim=1)#B N
                        sim_rgb_ir = F.normalize(f_out_ir, dim=1).mm(self.wise_memory_rgb.features.detach().data.t())#F.softmax(F.normalize(f_out_rgb, dim=1).mm(self.wise_memory_ir.features.detach().data.t())/0.05,dim=1)
                        sim_rgb_ir_exp =sim_rgb_ir /0.05  # 64*13638
                        score_intra_rgb_ir =   F.softmax(sim_rgb_ir_exp,dim=1)##sim_exp_intra / sim_exp_intra.sum(dim=1, keepdim=True)#
                        # print('score_intra',score_intra)
                        score_intra_rgb_ir = score_intra_rgb_ir.clamp_min(1e-8)
                        # count_rgb_ir = (mask_neighbor_rgb_ir).sum(dim=1)
                        ir_rgb_loss = -score_intra_rgb_ir.log().mul(mask_neighbor_rgb_ir).mul(mask_neighbor_prob_rgb_ir).sum(dim=1)#.mul(sim_wise) mul(mask_neighbor_intra) .mul(mask_neighbor_intra)
                        ir_rgb_loss = lamda_d_neibor*ir_rgb_loss.div(num_neighbor_rgb_ir).mean()#.mul(ir_rgb_ca).mul(ir_rgb__ca)mul(rgb_ir_ca).mul(mask_neighbor_intra_soft) ##.mul(rgb_ir_ca)

                        sim_prob_rgb_ir_exp =sim_prob_rgb_ir /0.05  # 64*13638
                        score_intra_rgb_ir_s =   F.softmax(sim_prob_rgb_ir_exp,dim=1)
                        score_intra_rgb_ir_s = score_intra_rgb_ir_s.clamp_min(1e-8)
                        # count_rgb_ir = (mask_neighbor_rgb_ir).sum(dim=1)
                        ir_rgb_loss_s = -score_intra_rgb_ir_s.log().mul(mask_neighbor_rgb_ir).mul(mask_neighbor_prob_rgb_ir).sum(dim=1)#.mul(sim_wise) mul(mask_neighbor_intra) .mul(mask_neighbor_intra)
                        # print('rgb_ir_loss_s',rgb_ir_loss_s.size())
                        ir_rgb_loss_s = lamda_s_neibor*ir_rgb_loss_s.div(num_neighbor_rgb_ir).mean()#.mul(ir_rgb_ca).mul(rgb_ir_ca) ##


                    # else:
                with torch.no_grad():
                    sim_prob_rgb_rgb = self.wise_memory_rgb_s.features.detach()[index_rgb].mm(self.wise_memory_rgb_s.features.detach().data.t())#F.softmax(F.normalize(f_out_rgb_s, dim=1).mm(self.wise_memory_rgb_s.features.detach().data.t())/0.05,dim=1)#B N
                    sim_rgb_rgb = self.wise_memory_rgb.features.detach()[index_rgb].mm(self.wise_memory_rgb.features.detach().data.t())#F.softmax(F.normalize(f_out_rgb, dim=1).mm(self.wise_memory_rgb.features.detach().data.t())/0.05,dim=1)


                    nearest_rgb_rgb = sim_rgb_rgb.max(dim=1, keepdim=True)[0]
                    nearest_prob_rgb_rgb = sim_prob_rgb_rgb.max(dim=1, keepdim=True)[0]
                    mask_neighbor_rgb_rgb = torch.gt(sim_rgb_rgb, nearest_rgb_rgb * hm_thresh).detach().data#nearest_intra * self.neighbor_eps)self.neighbor_eps
                    mask_neighbor_prob_rgb_rgb = torch.gt(sim_prob_rgb_rgb, nearest_prob_rgb_rgb * hm_thresh)#.cuda()#nearest_intra * self.neighbor_eps)self.neighbor_eps
                    num_neighbor_rgb_rgb = mask_neighbor_rgb_rgb.mul(mask_neighbor_prob_rgb_rgb).sum(dim=1)+1


                sim_prob_rgb_rgb = F.normalize(f_out_rgb_s, dim=1).mm(self.wise_memory_rgb_s.features.detach().data.t())#F.softmax(F.normalize(f_out_rgb_s, dim=1).mm(self.wise_memory_rgb_s.features.detach().data.t())/0.05,dim=1)#B N
                sim_rgb_rgb = F.normalize(f_out_rgb, dim=1).mm(self.wise_memory_rgb.features.detach().data.t())#F.softmax(F.normalize(f_out_rgb, dim=1).mm(self.wise_memory_rgb.features.detach().data.t())/0.05,dim=1)
                sim_rgb_rgb_exp =sim_rgb_rgb /0.05  # 64*13638
                score_intra_rgb_rgb =   F.softmax(sim_rgb_rgb_exp,dim=1)##sim_exp_intra / sim_exp_intra.sum(dim=1, keepdim=True)#
                # print('score_intra',score_intra)
                score_intra_rgb_rgb = score_intra_rgb_rgb.clamp_min(1e-8)
                # count_rgb_ir = (mask_neighbor_rgb_ir).sum(dim=1)
                rgb_rgb_loss = -score_intra_rgb_rgb.log().mul(mask_neighbor_rgb_rgb).mul(mask_neighbor_prob_rgb_rgb).sum(dim=1) #.mul(sim_wise) mul(mask_neighbor_intra) .mul(mask_neighbor_intra)
                rgb_rgb_loss = lamda_d_neibor*rgb_rgb_loss.div(num_neighbor_rgb_rgb).mean()#.mul(rgb_ca).mul(rgb_ca).mul(rgb_ca)..mul(rgb_ca)mul(mask_neighbor_intra_soft) ##

                sim_prob_rgb_rgb_exp =sim_prob_rgb_rgb /0.05  # 64*13638
                score_intra_rgb_rgb_s =  F.softmax(sim_prob_rgb_rgb_exp,dim=1)
                score_intra_rgb_rgb_s = score_intra_rgb_rgb_s.clamp_min(1e-8)
                # count_rgb_ir = (mask_neighbor_rgb_ir).sum(dim=1)
                rgb_rgb_loss_s = -score_intra_rgb_rgb_s.log().mul(mask_neighbor_rgb_rgb).mul(mask_neighbor_prob_rgb_rgb).sum(dim=1)#.mul(sim_wise) mul(mask_neighbor_intra) .mul(mask_neighbor_intra)
                rgb_rgb_loss_s = lamda_s_neibor*rgb_rgb_loss_s.div(num_neighbor_rgb_rgb).mean()#.mul(rgb_ca).mul(rgb_ca).mul(mask_neighbor_intra_soft) ##

                # #################ir-ir

                with torch.no_grad():
                    sim_prob_ir_ir = self.wise_memory_ir_s.features.detach()[index_ir].mm(self.wise_memory_ir_s.features.detach().data.t())#F.softmax(F.normalize(f_out_ir_s, dim=1).mm(self.wise_memory_ir_s.features.detach().data.t())/0.05,dim=1)#B N
                    sim_ir_ir = self.wise_memory_ir.features.detach()[index_ir].mm(self.wise_memory_ir.features.detach().data.t())#F.softmax(F.normalize(f_out_ir, dim=1).mm(self.wise_memory_ir.features.detach().data.t())/0.05,dim=1)



                    nearest_ir_ir = sim_ir_ir.max(dim=1, keepdim=True)[0]
                    nearest_prob_ir_ir = sim_prob_ir_ir.max(dim=1, keepdim=True)[0]
                    mask_neighbor_prob_ir_ir = torch.gt(sim_prob_ir_ir, nearest_prob_ir_ir * hm_thresh)#.cuda()#nearest_intra * self.neighbor_eps)self.neighbor_eps
                    mask_neighbor_ir_ir = torch.gt(sim_ir_ir, nearest_ir_ir * hm_thresh).detach().data#nearest_intra * self.neighbor_eps)self.neighbor_eps
                    num_neighbor_ir_ir = mask_neighbor_ir_ir.mul(mask_neighbor_prob_ir_ir).sum(dim=1)+1#.mul(sim_wise).
                    # print('num_neighbor_ir_ir',num_neighbor_ir_ir)

                sim_prob_ir_ir = F.normalize(f_out_ir_s, dim=1).mm(self.wise_memory_ir_s.features.detach().data.t())#F.softmax(F.normalize(f_out_ir_s, dim=1).mm(self.wise_memory_ir_s.features.detach().data.t())/0.05,dim=1)#B N
                sim_ir_ir = F.normalize(f_out_ir, dim=1).mm(self.wise_memory_ir.features.detach().data.t())#F.softmax(F.normalize(f_out_ir, dim=1).mm(self.wise_memory_ir.features.detach().data.t())/0.05,dim=1)
                sim_ir_ir_exp =sim_ir_ir /0.05  # 64*13638
                score_intra_ir_ir =   F.softmax(sim_ir_ir_exp,dim=1)##sim_exp_intra / sim_exp_intra.sum(dim=1, keepdim=True)#
                # print('score_intra',score_intra)
                score_intra_ir_ir = score_intra_ir_ir.clamp_min(1e-8)
                # count_ir_rgb = (mask_neighbor_ir_rgb).sum(dim=1)
                ir_ir_loss = -score_intra_ir_ir.log().mul(mask_neighbor_ir_ir).mul(mask_neighbor_prob_ir_ir).sum(dim=1)#.mul(sim_wise) mul(mask_neighbor_intra) .mul(mask_neighbor_intra)
                ir_ir_loss = lamda_d_neibor*ir_ir_loss.div(num_neighbor_ir_ir).mean()#.mul(ir_ca).mul(ir_ca).mul(ir_ca).mul(ir_ca).mul(mask_neighbor_intra_soft) ##


                sim_prob_ir_ir_exp =sim_prob_ir_ir /0.05  # 64*13638
                score_intra_ir_ir_s =  F.softmax(sim_prob_ir_ir_exp,dim=1)
                score_intra_ir_ir_s = score_intra_ir_ir_s.clamp_min(1e-8)
                # count_rgb_ir = (mask_neighbor_rgb_ir).sum(dim=1)
                ir_ir_loss_s = -score_intra_ir_ir_s.log().mul(mask_neighbor_ir_ir).mul(mask_neighbor_prob_ir_ir).sum(dim=1)#.mul(sim_wise) mul(mask_neighbor_intra) .mul(mask_neighbor_intra)
                ir_ir_loss_s = lamda_s_neibor*ir_ir_loss_s.div(num_neighbor_ir_ir).mean()#.mul(ir_ca).mul(ir_ca).mul(mask_neighbor_intra_soft) ##



            if epoch >=self.cmlabel:
                loss=(loss_ir+loss_rgb+loss_rgb_s+loss_ir_s)+0.1*(ir_ir_loss+rgb_rgb_loss+ir_ir_loss_s+rgb_rgb_loss_s)+0.1*(rgb_ir_loss+ir_rgb_loss+rgb_ir_loss_s+ir_rgb_loss_s)#+0*(loss_ins_ir+loss_ins_rgb+loss_ins_ir_s+loss_ins_rgb_s)#+0.1*(loss_camera_ir+loss_camera_rgb)
            else:
                loss=(loss_ir+loss_rgb+loss_rgb_s+loss_ir_s)+1*(ir_ir_loss+rgb_rgb_loss+ir_ir_loss_s+rgb_rgb_loss_s)+0.5*(rgb_ir_loss+ir_rgb_loss+rgb_ir_loss_s+ir_rgb_loss_s)#+0*


            with torch.no_grad():
                self.wise_memory_ir.updateEM(f_out_ir, index_ir)
                self.wise_memory_rgb.updateEM(f_out_rgb, index_rgb)
                self.wise_memory_ir_s.updateEM(f_out_ir_s, index_ir)
                self.wise_memory_rgb_s.updateEM(f_out_rgb_s, index_rgb)


            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.update(loss.item())

            loss_ir_log.update(loss_ir.item())
            loss_rgb_log.update(loss_rgb.item())

            ir_rgb_loss_log.update(ir_rgb_loss.item())
            ir_ir_loss_log.update(ir_ir_loss.item())

            # print log
            batch_time.update(time.time() - end)
            end = time.time()

            if (i + 1) % print_freq == 0:
                print('Epoch: [{}][{}/{}]\t'
                      'Time {:.3f} ({:.3f})\t'
                      'Data {:.3f} ({:.3f})\t'
                      'Loss {:.3f} ({:.3f})\t'
                      'Loss ir {:.3f} ({:.3f})\t'
                      'Loss rgb {:.3f} ({:.3f})\t'
                      'camera ir {:.3f} ({:.3f})\t'
                      'camera rgb {:.3f} ({:.3f})\t'
                      'ir_rgb_loss_log {:.3f} ({:.3f})\t'
                      'rgb_ir_loss_log {:.3f} ({:.3f})\t'
                      'ir_ir_loss_log {:.3f} ({:.3f})\t'
                      'rgb_rgb_loss_log {:.3f} ({:.3f})\t'

                      .format(epoch, i + 1, len(data_loader_rgb),
                              batch_time.val, batch_time.avg,
                              data_time.val, data_time.avg,
                              losses.val, losses.avg,loss_ir_log.val,loss_ir_log.avg,loss_rgb_log.val,loss_rgb_log.avg,\
                              loss_camera_ir_log.val,loss_camera_ir_log.avg,loss_camera_rgb_log.val,loss_camera_rgb_log.avg,\
                              ir_rgb_loss_log.val,ir_rgb_loss_log.avg,rgb_ir_loss_log.val,rgb_ir_loss_log.avg,\
                              ir_ir_loss_log.val,ir_ir_loss_log.avg,rgb_rgb_loss_log.val,rgb_rgb_loss_log.avg))
                print('loss_ir_s,loss_rgb_s',loss_ir_s.item(),loss_rgb_s.item())
                print('loss_shared_s,loss_shared',loss_shared_s.item(),loss_shared.item())
                print('ir_rgb_loss_s,rgb_ir_loss_s',ir_rgb_loss_s.item(),rgb_ir_loss_s.item())
                print('ir_ir_loss_s,rgb_rgb_loss_s',ir_ir_loss_s.item(),rgb_rgb_loss_s.item())

    def _parse_data_rgb(self, inputs):
        imgs,imgs1, name, pids, cids, indexes = inputs
        return imgs.cuda(),imgs1.cuda(), pids.cuda(), indexes.cuda(),cids.cuda(),name

    def _parse_data_ir(self, inputs):
        imgs, name, pids, cids, indexes = inputs
        return imgs.cuda(), pids.cuda(), indexes.cuda(),cids.cuda(),name



    def _forward(self, x1, x2, label_1=None,label_2=None,modal=0,cid_rgb=None,cid_ir=None,index_rgb=None,index_ir=None):
        return self.encoder(x1, x2, modal=modal,label_1=label_1,label_2=label_2,cid_rgb=cid_rgb,cid_ir=cid_ir,index_rgb=index_rgb,index_ir=index_ir)



    def init_camera_proxy(self,all_img_cams,all_pseudo_label,intra_id_features):
        all_img_cams = torch.tensor(all_img_cams).cuda()
        unique_cams = torch.unique(all_img_cams)
        # print(self.unique_cams)

        all_pseudo_label = torch.tensor(all_pseudo_label).cuda()
        init_intra_id_feat = intra_id_features
        # print(len(self.init_intra_id_feat))

        # initialize proxy memory
        percam_memory = []
        memory_class_mapper = []
        concate_intra_class = []
        for cc in unique_cams:
            percam_ind = torch.nonzero(all_img_cams == cc).squeeze(-1)
            uniq_class = torch.unique(all_pseudo_label[percam_ind])
            uniq_class = uniq_class[uniq_class >= 0]
            concate_intra_class.append(uniq_class)
            cls_mapper = {int(uniq_class[j]): j for j in range(len(uniq_class))}
            memory_class_mapper.append(cls_mapper)  # from pseudo label to index under each camera

            if len(init_intra_id_feat) > 0:
                # print('initializing ID memory from updated embedding features...')
                proto_memory = init_intra_id_feat[cc]
                proto_memory = proto_memory.cuda()
                percam_memory.append(proto_memory.detach())
            # print(cc,proto_memory.size())
        concate_intra_class = torch.cat(concate_intra_class)

        percam_tempV = []
        for ii in unique_cams:
            percam_tempV.append(percam_memory[ii].detach().clone())
        percam_tempV_ = torch.cat(percam_tempV, dim=0).cuda()
        return concate_intra_class,percam_tempV_,percam_memory#memory_class_mapper,
    def camera_loss(self,f_out_t1,cids,targets,percam_tempV,concate_intra_class,memory_class_mapper):
        beta = 0.07
        bg_knn = 50
        loss_cam = torch.tensor([0.]).cuda()
        for cc in torch.unique(cids):
            # print(cc)
            inds = torch.nonzero(cids == cc).squeeze(-1)
            percam_targets = targets[inds]
            # print(percam_targets)
            percam_feat = f_out_t1[inds]

            associate_loss = 0
            # target_inputs = percam_feat.mm(percam_tempV.t().clone())
            target_inputs = torch.matmul(F.normalize(percam_feat), F.normalize(percam_tempV.t().clone()))
            temp_sims = target_inputs.detach().clone()
            target_inputs /= beta
            for k in range(len(percam_feat)):
                ori_asso_ind = torch.nonzero(concate_intra_class == percam_targets[k]).squeeze(-1)
                if len(ori_asso_ind) == 0:
                    continue
                temp_sims[k, ori_asso_ind] = -10000.0  # mask out positive
                sel_ind = torch.sort(temp_sims[k])[1][-bg_knn:]
                concated_input = torch.cat((target_inputs[k, ori_asso_ind], target_inputs[k, sel_ind]), dim=0)
                concated_target = torch.zeros((len(concated_input)), dtype=concated_input.dtype).to(
                    torch.device('cuda'))

                concated_target[0:len(ori_asso_ind)] = 1.0 / len(ori_asso_ind)
                associate_loss += -1 * (
                        F.log_softmax(concated_input.unsqueeze(0), dim=1) * concated_target.unsqueeze(
                    0)).sum()
            loss_cam +=  associate_loss / len(percam_feat)
        return loss_cam

    @torch.no_grad()
    def generate_cluster_features(self,labels, features):
        centers = collections.defaultdict(list)
        for i, label in enumerate(labels):
            if label == -1:
                continue
            centers[labels[i].item()].append(features[i])

        for idx in sorted(centers.keys()):
            centers[idx] = torch.stack(centers[idx], dim=0).mean(0)

        return centers

    def mask(self,ones, labels,ins_label):
        for i, label in enumerate(labels):
            ones[i,ins_label==label] = 1
        return ones

    def part_sim(self,query_t, key_m):
        
        self.seq_len=5
        q, d_5 = query_t.size() # b d*5,
        k, d_5 = key_m.size()

        z= int(d_5/self.seq_len)
        d = int(d_5/self.seq_len)
        # query_t =  query_t.detach().view(q, -1, z)#self.bn3(tgt.view(q, -1, z))  #B N C
        # key_m = key_m.detach().view(k, -1, d)#self.bn3(memory.view(k, -1, d)) #B N C

        query_t = F.normalize(query_t.view(q, -1, z), dim=-1)  #B N C tgt.view(q, -1, z)#
        key_m = F.normalize(key_m.view(k, -1, d), dim=-1) #Q N C memory.view(k, -1, d)#
        # score = einsum('q t d, k s d -> q k s t', query_t, key_m)#F.softmax(einsum('q t d, k s d -> q k s t', query_t, key_m),dim=-1).view(q,-1) # B Q N N
        score = einsum('q t d, k s d -> q k t s', query_t, key_m)

        score = torch.cat((score.max(dim=2)[0], score.max(dim=3)[0]), dim=-1) #####score.max(dim=3)[0]#q k 10
        score = F.softmax(score.permute(0,2,1)/0.01,dim=-1).reshape(q,-1)

        # score = F.softmax(score,dim=1)
        return score


def pdist_torch(emb1, emb2):
    '''
    compute the eucilidean distance matrix between embeddings1 and embeddings2
    using gpu
    '''
    m, n = emb1.shape[0], emb2.shape[0]
    emb1_pow = torch.pow(emb1, 2).sum(dim = 1, keepdim = True).expand(m, n)
    emb2_pow = torch.pow(emb2, 2).sum(dim = 1, keepdim = True).expand(n, m).t()
    dist_mtx = emb1_pow + emb2_pow
    dist_mtx = dist_mtx.addmm_(1, -2, emb1, emb2.t())
    # dist_mtx = dist_mtx.clamp(min = 1e-12)
    dist_mtx = dist_mtx.clamp(min = 1e-12).sqrt()
    return dist_mtx
def softmax_weights(dist, mask):
    max_v = torch.max(dist * mask, dim=1, keepdim=True)[0]
    diff = dist - max_v
    Z = torch.sum(torch.exp(diff) * mask, dim=1, keepdim=True) + 1e-6 # avoid division by zero
    W = torch.exp(diff) * mask / Z
    return W
def normalize(x, axis=-1):
    """Normalizing to unit length along the specified dimension.
    Args:
      x: pytorch Variable
    Returns:
      x: pytorch Variable, same shape as input
    """
    x = 1. * x / (torch.norm(x, 2, axis, keepdim=True).expand_as(x) + 1e-12)
    return x









