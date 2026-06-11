import collections
import numpy as np
from abc import ABC
import torch
import torch.nn.functional as F
from torch import nn, autograd
from collections import defaultdict
import random
class CM(autograd.Function):

    @staticmethod
    def forward(ctx, inputs, targets, features, momentum):
        ctx.features = features
        ctx.momentum = momentum
        ctx.save_for_backward(inputs, targets)
        outputs = inputs.mm(ctx.features.t())

        return outputs

    @staticmethod
    def backward(ctx, grad_outputs):
        inputs, targets = ctx.saved_tensors
        grad_inputs = None
        if ctx.needs_input_grad[0]:
            grad_inputs = grad_outputs.mm(ctx.features)

        # momentum update
        for x, y in zip(inputs, targets):
            ctx.features[y] = ctx.momentum * ctx.features[y] + (1. - ctx.momentum) * x
            ctx.features[y] /= ctx.features[y].norm()

        return grad_inputs, None, None, None


def cm(inputs, indexes, features, momentum=0.5):
    return CM.apply(inputs, indexes, features, torch.Tensor([momentum]).to(inputs.device))


class CM_Hard(autograd.Function):

    @staticmethod
    def forward(ctx, inputs, targets, features, momentum, num_instances):
        ctx.features = features
        ctx.momentum = momentum
        ctx.num_instances = num_instances
        ctx.save_for_backward(inputs, targets)
        outputs = inputs.mm(ctx.features.t())

        return outputs

    @staticmethod
    def backward(ctx, grad_outputs):
        inputs, targets = ctx.saved_tensors
        nums = len(ctx.features) // ctx.num_instances 
        grad_inputs = None
        if ctx.needs_input_grad[0]:
            grad_inputs = grad_outputs.mm(ctx.features)

        batch_centers = collections.defaultdict(list)
        for instance_feature, index in zip(inputs, targets.tolist()):
            batch_centers[index].append(instance_feature)

        # 更新实例特征
        for index, features in batch_centers.items():
            indexes = [index + nums * i for i in range(0, ctx.num_instances)]
            if len(features) >= 16:
                ids = np.random.choice(len(features), len(indexes), replace=False)
            else:
                ids = np.random.choice(len(features), len(indexes), replace=True)

            ctx.features[indexes] = torch.stack(features)[ids]
            # ctx.features[indexes] /= ctx.features[indexes].norm()
            
        return grad_inputs, None, None, None, None


def cm_hard(inputs, indexes, features, momentum=0.5, num_instances=4):
    return CM_Hard.apply(inputs, indexes, features, torch.Tensor([momentum]).to(inputs.device), num_instances)



class ClusterMemory_all(nn.Module, ABC):
    def __init__(self, num_features, num_samples, temp=0.05, momentum=0.2, use_hard=False):
        super(ClusterMemory_all, self).__init__()
        self.num_features = num_features
        self.num_samples = num_samples

        self.momentum = momentum
        self.temp = temp
        self.use_hard = use_hard

        self.register_buffer('features', torch.zeros(num_samples, num_features))

    def forward(self, inputs, targets):

        inputs = F.normalize(inputs, dim=1).cuda()
        # if self.use_hard:
        outputs1 = cm_hard(inputs, targets, self.features, self.momentum)
        # else:
        outputs2 = cm(inputs, targets, self.features, self.momentum)

        outputs1 /= self.temp
        outputs2 /= self.temp
        loss1 = F.cross_entropy(outputs2, targets) + 0.1*F.cross_entropy(outputs1, targets)
    
        return loss



#ori
class ClusterMemory(nn.Module, ABC):
    def __init__(self, num_features, num_samples, temp=0.05, momentum=0.2, use_hard=False,smooth=0,num_instances=4):
        super(ClusterMemory, self).__init__()
        self.num_features = num_features
        self.num_samples = num_samples

        self.momentum = momentum
        self.temp = temp
        self.use_hard = use_hard
        self.smooth = smooth
        
        self.register_buffer('features', torch.zeros(num_samples, num_features))
        if smooth > 0:
            self.cross_entropy = AdaptiveLabelSmoothLoss(self.num_samples, 0.3, 0.05, 10, 0.02)
            print('>>> Using CrossEntropy with Label Smoothing.')
        else:
            self.cross_entropy = nn.CrossEntropyLoss().cuda()
    def forward(self, inputs, targets,ca=None,training_momentum=None,return_out=False):

        inputs = F.normalize(inputs, dim=1).cuda()
        if training_momentum == None:
            if self.use_hard:
                outputs = cm_hard(inputs, targets, self.features, self.momentum,self.num_instances)
            else:
                outputs = cm(inputs, targets, self.features, self.momentum)
        else:
            if self.use_hard:
                outputs = cm_hard(inputs, targets, self.features, training_momentum,self.num_instances)
            else:
                outputs = cm(inputs, targets, self.features, training_momentum)
        outputs /= self.temp
        
        if return_out:
            return outputs
        
        if ca == None:
            loss = F.cross_entropy(outputs, targets)
        else:
            loss = (F.cross_entropy(outputs, targets,reduction='none')*ca).mean()

        return loss


class EM(autograd.Function):
    @staticmethod
    def forward(ctx, inputs, indexes, features, momentum):
        ctx.features = features
        ctx.momentum = momentum
        ctx.save_for_backward(inputs, indexes)
        outputs = inputs.mm(ctx.features.t())
        return outputs

    @staticmethod
    def backward(ctx, grad_outputs):
        inputs, indexes = ctx.saved_tensors
        grad_inputs = None
        if ctx.needs_input_grad[0]:
            grad_inputs = grad_outputs.mm(ctx.features)

        # momentum update, not applied for meta learning
        for x, y in zip(inputs, indexes):
            ctx.features[y] = ctx.momentum * ctx.features[y] + (1. - ctx.momentum) * x
            ctx.features[y] /= ctx.features[y].norm()

        return grad_inputs, None, None, None


def em(inputs, indexes, features, momentum=0.5):
    return EM.apply(inputs, indexes, features, torch.Tensor([momentum]).to(inputs.device))


class Memory(nn.Module):
    def __init__(self, num_features, num_samples,num_cluster, temp=0.05, momentum=0.2):
        super(Memory, self).__init__()
        self.num_features = num_features
        self.num_samples = num_samples
        self.momentum = momentum
        self.temp = temp
        self.devices = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.register_buffer('features', torch.zeros(num_samples, num_features))
        # features--(source centers+tgt features)
        self.register_buffer('labels', torch.zeros(num_samples).long())
        self.register_buffer('cam', torch.zeros(num_samples).long())
        # labels--(each src and predicted tgt id and outliers), 13638
        self.sce = AdaptiveLabelSmoothLoss(num_cluster)
        self.global_std, self.global_mean = torch.zeros(num_features).to(self.devices), \
                                            torch.zeros(num_features).to(self.devices)
    def updateEM(self, inputs, indexes):
        # momentum update
        for x, y in zip(inputs, indexes):
            self.features[y] = self.momentum * self.features[y] + (1. - self.momentum) * x

    def __update_params(self):
        camSet = set(self.cam.cpu().numpy().tolist())
        temp_std, temp_mean = [], []
        for cam in camSet:
            cam_feat = self.features[self.cam == cam]
            if len(cam_feat) <= 1: continue
            temp_std.append(cam_feat.std(0))
            temp_mean.append(cam_feat.mean(0))
        self.global_std = self.momentum * torch.stack(temp_std).mean(0) + \
                          (1 - self.momentum) * self.global_std
        self.global_mean = self.momentum * torch.stack(temp_mean).mean(0) + \
                           (1 - self.momentum) * self.global_mean

    def forward(self, inputs, indexes,cameras, symmetric=False):
        # self.__update_params()
        # inputs: B*2048, features: L*2048
        # get scores for all samples, inputs--(64*13638)
        inputs = F.normalize(inputs, dim=1).cuda()



        inputs = em(inputs, indexes, self.features, self.momentum)#B N 
        input_forinc=inputs
        inputs /= self.temp  # 64*13638
        B = inputs.size(0)

        targets = self.labels[indexes].clone()
        # print('targets',targets)
        labels = self.labels.clone()  # 16522, whole labels

        # get centroids for each id
        sim = torch.zeros(labels.max() + 1, B).float().cuda() # C B
        # re-arange simi matrix according to labels to find centroids
        sim.index_add_(0, labels[labels != -1], inputs[:, labels != -1].t().contiguous())

        nums = torch.zeros(labels.max() + 1, 1).float().cuda()
        # get counter
        nums.index_add_(0, labels[labels != -1], torch.ones(labels[labels != -1].shape[0], 1).float().cuda())
        sim /= nums.clone().expand_as(sim)  # compute centroids # C B
        sim = sim.t() #B,C
        # loss = self.sce(sim,targets)#torch.tensor([0.]).cuda()#
        # loss = F.cross_entropy(sim, targets)

        # softMask = torch.zeros(sim.t().shape).cuda()
        # softMask.scatter_(1, targets.view(-1, 1), 1)
        # loss = F.cross_entropy(sim, targets)
        ########soft instance
        loss = -(F.softmax(input_forinc/10, 1) * F.log_softmax(inputs, dim=1)).sum(1).mean()


        return loss#,loss_cam 



class AdaptiveLabelSmoothLoss(nn.Module):
    """
    Adaptive Label Smoothing Loss Function, specially designed for unsupervised pedestrian Re-ID clustering
    
    Improvements:1. Dynamic epsilon adjustment based on clustering confidence
                 2. Top-K neighbor aware smoothing strategy
                 3. Curriculum learning style smoothing intensity decay
    """
    def __init__(self, num_classes, max_epsilon=0.3, min_epsilon=0.05, 
                 warmup_epochs=10, topk_ratio=0.02):
        super().__init__()
        self.num_classes = num_classes
        self.max_epsilon = max_epsilon
        self.min_epsilon = min_epsilon
        self.warmup_epochs = warmup_epochs
        self.topk = max(1, int(num_classes * topk_ratio))
        self.logsoftmax = nn.LogSoftmax(dim=1)
        
        # 聚类记忆库
        self.register_buffer('cluster_centers', torch.zeros(num_classes, 512))
        self.register_buffer('cluster_counts', torch.zeros(num_classes))

    def forward(self, inputs, targets, features, epoch):
        """
        Args:
            inputs: 预测logits (batch_size, num_classes)
            targets: 聚类伪标签 (batch_size,)
            features: 当前样本特征 (batch_size, feat_dim)
            epoch: 当前训练轮次
        """
        # 1. 动态计算epsilon
        epsilon = self._get_adaptive_epsilon(features, targets, epoch)
        
        # 2. 获取Top-K预测类别
        log_probs = self.logsoftmax(inputs)
        topk = torch.topk(log_probs, self.topk, dim=1)[1] if self.topk > 1 else None
        
        # 3. 构建平滑标签
        smooth_targets = self._build_smooth_targets(
            log_probs, targets, epsilon, topk)
        
        # 4. 计算损失
        loss = (-smooth_targets * log_probs).sum(dim=1).mean()
        
        # 5. 更新聚类统计量
        self._update_cluster_stats(features.detach(), targets)
        
        return loss

    def _get_adaptive_epsilon(self, features, targets, epoch):
        """三阶段自适应epsilon策略"""
        # 阶段1：课程学习衰减
        curriculum = 1.0 - min(epoch / self.warmup_epochs, 1.0)
        base_epsilon = self.min_epsilon + (self.max_epsilon - self.min_epsilon) * curriculum
        
        # 阶段2：基于聚类距离的调整
        with torch.no_grad():
            centers = self.cluster_centers[targets]
            distances = 1 - F.cosine_similarity(features, centers, dim=1)
            distance_factor = distances / (distances.mean() + 1e-6)
        
        # 阶段3：基于类别频率的平衡
        counts = self.cluster_counts[targets]
        freq_factor = 1.0 / (torch.log(counts.float() + 1.0) + 0.5)
        
        # 综合调整
        epsilon = (base_epsilon * distance_factor * freq_factor).clamp(self.min_epsilon, self.max_epsilon)
        
        return epsilon.mean()

    def _build_smooth_targets(self, log_probs, targets, epsilon, topk):
        """构建平滑标签分布"""
        batch_size = targets.size(0)
        targets = targets.unsqueeze(1)
        
        # 基础one-hot
        smooth_targets = torch.zeros_like(log_probs)
        smooth_targets.scatter_(1, targets, 1.0 - epsilon.unsqueeze(1))
        
        # Top-K平滑
        if topk is not None:
            k = min(self.topk, self.num_classes - 1)
            smooth_value = epsilon.unsqueeze(1) / k
            smooth_targets.scatter_(1, topk, smooth_value)
        else:
            # 常规均匀平滑
            smooth_value = epsilon.unsqueeze(1) / self.num_classes
            smooth_targets += smooth_value
        
        return smooth_targets

    def _update_cluster_stats(self, features, labels):
        """更新聚类中心统计量"""
        for lbl in torch.unique(labels):
            mask = (labels == lbl)
            if mask.sum() == 0:
                continue
                
            # 指数移动平均更新
            center = features[mask].mean(dim=0)
            self.cluster_centers[lbl] = 0.9 * self.cluster_centers[lbl] + 0.1 * center
            self.cluster_counts[lbl] += mask.sum().item()



class CamMemory(nn.Module):
    def __init__(self, num_features, num_samples, temp=0.05, momentum=0.2):
        super(CamMemory, self).__init__()
        self.devices = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.num_features = num_features
        self.num_samples = num_samples

        self.momentum = momentum
        self.temp = temp

        self.register_buffer('features', torch.zeros(num_samples, num_features).to(self.devices))
        # features--(source centers+tgt features)
        self.register_buffer('labels', torch.zeros(num_samples).long().to(self.devices))

        self.register_buffer('cam', torch.zeros(num_samples).long())
        # labels--(each src and predicted tgt id and outliers), 13638

        self.global_std, self.global_mean = torch.zeros(num_features).to(self.devices), \
                                            torch.zeros(num_features).to(self.devices)

    def updateEM(self, inputs, indexes):
        # momentum update
        for x, y in zip(inputs, indexes):
            self.features[y] = self.momentum * self.features[y] + (1. - self.momentum) * x
            self.features[y] /= self.features[y].norm()

    def __update_params(self):
        camSet = set(self.cam.cpu().numpy().tolist())
        temp_std, temp_mean = [], []
        for cam in camSet:
            cam_feat = self.features[self.cam == cam]
            if len(cam_feat) <= 1: continue
            temp_std.append(cam_feat.std(0))
            temp_mean.append(cam_feat.mean(0))
        self.global_std = self.momentum * torch.stack(temp_std).mean(0) + \
                          (1 - self.momentum) * self.global_std
        self.global_mean = self.momentum * torch.stack(temp_mean).mean(0) + \
                           (1 - self.momentum) * self.global_mean

    def forward(self, features, indexes, cameras, symmetric=False):
        # inputs: B*2048, features: L*2048
        # get scores for all samples, inputs--(64*13638)
        self.__update_params()  # update camera-level params
        inputs = em(features, indexes, self.features, self.momentum)
        inputs /= self.temp  # 64*13638
        B = inputs.size(0)

        targets = self.labels[indexes].clone()
        labels = self.labels.clone()  # 13638, whole labels

        # get centroids for each id
        sim = torch.zeros(labels.max() + 1, B).float().cuda()  # 12123(maxID)*64
        # re-arange simi matrix according to labels
        sim.index_add_(0, labels, inputs.t().contiguous())  # labels--13638(centroids+tgt IDs), inputs--13638*64

        nums = torch.zeros(labels.max() + 1, 1).float().cuda()  # 12123(maxID)
        # get counter
        nums.index_add_(0, labels, torch.ones(self.num_samples, 1).float().cuda())

        sim /= nums.clone().expand_as(sim)

        # get camera loss
        num_cams, cam_set, loss_cam = len(set(self.cam)), set(self.cam.cpu().numpy().tolist()), []
        for cur_cam in range(len(cam_set)):
            cam_feat = features[cur_cam == cameras]
            if len(cam_feat) <= 1:
                continue
            temp_mean, temp_std = cam_feat.mean(0), cam_feat.std(0)

            loss_mean = (temp_mean - self.global_mean).pow(2).sum()
            loss_std = (temp_std - self.global_std).pow(2).sum()
            loss_cam.append(loss_mean)
            loss_cam.append(loss_std)
        loss_cam = 0 if len(loss_cam) == 0 else torch.stack(loss_cam).mean()
        softMask = torch.zeros(sim.t().shape).cuda()
        softMask.scatter_(1, targets.view(-1, 1), 1)
        loss = -(softMask * F.log_softmax(sim.t(), dim=1)).sum(1).mean()
        loss_sym = 0
        if symmetric:
            loss_sym = -(F.softmax(sim.t(), 1) * F.log_softmax(softMask, dim=1)).sum(1).mean()



        return loss,loss_sym,loss_cam

#######ori
class Memory_wise(nn.Module):
    def __init__(self, num_features, num_samples,num_cluster, temp=0.05, momentum=0.2):
        super(Memory_wise, self).__init__()
        self.num_features = num_features
        self.num_samples = num_samples
        self.momentum = momentum
        self.temp = temp
        self.devices = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.register_buffer('features', torch.zeros(num_samples, num_features))
        # features--(source centers+tgt features)
        self.register_buffer('labels', torch.zeros(num_samples).long())
        self.register_buffer('cam', torch.zeros(num_samples).long())
        # labels--(each src and predicted tgt id and outliers), 13638
    def cam2uid(self):
        uid2cam = zip(range(self.num_samples), self.cam)
        self.cam2uid = defaultdict(list)
        for uid, cam in uid2cam:
            self.cam2uid[int(cam.cpu().data)].append(uid)
        # print(self.cam2uid)

    def updateEM(self, inputs, indexes):
        # momentum update
        for x, y in zip(inputs, indexes):
            self.features[y] = self.momentum * self.features[y] + (1. - self.momentum) * x
            self.features[y] /= self.features[y].norm()

    def forward(self, inputs, indexes,cameras,neighbor_eps=0.9):
        self.thresh=0.5
        self.neighbor_eps  = neighbor_eps
        inputs = F.normalize(inputs, dim=1).cuda()
        sim = em(inputs, indexes, self.features, self.momentum)#B N 
        sim_exp =sim /self.temp  # 64*13638
        B = inputs.size(0)
        mask_instance, mask_intra, mask_inter = self.compute_mask(sim.size(), indexes, cameras, sim.device)
        # -------------------------- Intra-camera Neighborhood Loss -------------------------- #

        sim_intra = (sim.data + 1) * mask_intra * (1 - mask_instance) - 1
        # print('sim_intra.sum(1)',sim_intra.sum(1))
        nearest_intra = sim_intra.max(dim=1, keepdim=True)[0]
        # print('nearest_intra',nearest_intra)
        mask_neighbor_intra = torch.gt(sim_intra, nearest_intra * self.neighbor_eps)
        num_neighbor_intra = mask_neighbor_intra.sum(dim=1)+1
        # print('num_neighbor_intra',num_neighbor_intra)
        # Activate intra-camera candidates
        sim_exp_intra = sim_exp * mask_intra
        # print('sim_exp_intra',sim_exp_intra)
        score_intra =  F.softmax(sim_exp_intra,dim=1)# sim_exp_intra / sim_exp_intra.sum(dim=1, keepdim=True)# 
        # print('score_intra',score_intra)
        score_intra = score_intra.clamp_min(1e-8)
        # print('score_intra',score_intra)
        intra_loss = -score_intra.log().mul(mask_neighbor_intra).sum(dim=1)
        intra_loss = intra_loss.div(num_neighbor_intra)

        ins_loss = -score_intra.masked_select(mask_instance.bool()).log()
        # -------------------------- Inter-Camera Neighborhood Loss --------------------------#
        # Compute masks for inter-camera neighbors
        sim_inter = (sim.data + 1) * mask_inter - 1
        nearest_inter = sim_inter.max(dim=1, keepdim=True)[0]
        mask_neighbor_inter = torch.gt(sim_inter, nearest_inter * self.neighbor_eps)
        num_neighbor_inter = mask_neighbor_inter.sum(dim=1)+1
        # print('num_neighbor_inter',num_neighbor_inter)
        # Activate inter-camera candidates
        sim_exp_inter = sim_exp * mask_inter
        score_inter = F.softmax(sim_exp_inter,dim=1) #sim_exp_inter / sim_exp_inter.sum(dim=1, keepdim=True) #
        score_inter = score_inter.clamp_min(1e-8)
        inter_loss = -score_inter.log().mul(mask_neighbor_inter).sum(dim=1)
        inter_loss = inter_loss.div(num_neighbor_inter)


        return ins_loss.mean(),intra_loss.mean(),inter_loss.mean()* 0.6#loss#,loss_cam inter_loss.mean()
    def compute_mask(self, size, img_ids, cam_ids, device):
        # print('self.cam2uid',self.cam2uid)
        # print('cam_ids',cam_ids)
        mask_inter = torch.ones(size, device=device)
        for i, cam in enumerate(cam_ids.tolist()):
            intra_cam_ids = self.cam2uid[cam]
            # print(cam_ids)
            
            # print('intra_cam_ids',intra_cam_ids)
            mask_inter[i, intra_cam_ids] = 0

        mask_intra = 1 - mask_inter
        # print(mask_intra)
        mask_instance = torch.zeros(size, device=device)
        mask_instance[torch.arange(size[0]), img_ids] = 1
        return mask_instance, mask_intra, mask_inter

    def compute_mask_camwise(self, size, img_ids, cam_ids, device):
        # print('self.cam2uid',self.cam2uid)
        # print('cam_ids',cam_ids)
        mask_inter = torch.ones(size, device=device)
        for i, cam in enumerate(cam_ids.tolist()):
            intra_cam_ids = self.cam2uid[cam]
            # print(cam_ids)
            
            # print('intra_cam_ids',intra_cam_ids)
            mask_inter[i, intra_cam_ids] = 0

        mask_intra = 1 - mask_inter
        # print(mask_intra)
        mask_instance = torch.zeros(size, device=device)
        mask_instance[torch.arange(size[0]), img_ids] = 1
        return mask_instance, mask_intra, mask_inter



class Memory_wise_v1(nn.Module):
    def __init__(self, num_features, num_samples,num_cluster, temp=0.05, momentum=0.2):
        super(Memory_wise_v1, self).__init__()
        self.num_features = num_features
        self.num_samples = num_samples
        self.momentum = momentum
        self.temp = temp
        self.devices = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.register_buffer('features', torch.zeros(num_samples, num_features))
        # features--(source centers+tgt features)
        self.register_buffer('labels', torch.zeros(num_samples).long())
        self.register_buffer('cam', torch.zeros(num_samples).long())
        # labels--(each src and predicted tgt id and outliers), 13638
    def cam2uid(self):
        uid2cam = zip(range(self.num_samples), self.cam)
        self.cam2uid = defaultdict(list)
        for uid, cam in uid2cam:
            self.cam2uid[int(cam.cpu().data)].append(uid)
        # print(self.cam2uid)
        self.allcam = torch.unique(self.cam).cpu().numpy().tolist()
    def updateEM(self, inputs, indexes):
        # momentum update
        for x, y in zip(inputs, indexes):
            self.features[y] = self.momentum * self.features[y] + (1. - self.momentum) * x
            self.features[y] /= self.features[y].norm()

    def forward(self, inputs, indexes,cameras,neighbor_eps=0.9):
        self.thresh=0.6
        self.neighbor_eps  = neighbor_eps
        inputs = F.normalize(inputs, dim=1).cuda()
        sim = em(inputs, indexes, self.features, self.momentum)#B N 
        sim_exp =sim /self.temp  # 64*13638
        B = inputs.size(0)

        intrawise_loss_total=torch.tensor([0.]).cuda()

        mask_instance, mask_intra, mask_inter = self.compute_mask(sim.size(), indexes, cameras, sim.device)
        # -------------------------- Intra-camera Neighborhood Loss -------------------------- #
        # Compute masks for intra-camera neighbors
        # print('mask_instance.sum(1)',mask_instance.sum(1))
        # print('mask_intra.sum(1)',mask_intra.sum(1))
        # print('mask_inter.sum(1)',mask_inter.sum(1))
        # print('sim',sim)
        sim_intra = (sim.data + 1) * mask_intra * (1 - mask_instance) - 1
        # print('sim_intra.sum(1)',sim_intra.sum(1))
        nearest_intra = sim_intra.max(dim=1, keepdim=True)[0]
        # print('nearest_intra',nearest_intra)
        mask_neighbor_intra = torch.gt(sim_intra, nearest_intra * self.neighbor_eps)
        num_neighbor_intra = mask_neighbor_intra.sum(dim=1)+1
        # print('num_neighbor_intra',num_neighbor_intra)
        # Activate intra-camera candidates
        sim_exp_intra = sim_exp * mask_intra
        # print('sim_exp_intra',sim_exp_intra)
        score_intra =  F.softmax(sim_exp_intra,dim=1)# sim_exp_intra / sim_exp_intra.sum(dim=1, keepdim=True)# 
        # print('score_intra',score_intra)
        score_intra = score_intra.clamp_min(1e-8)
        # print('score_intra',score_intra)
        intra_loss = -score_intra.log().mul(mask_neighbor_intra).sum(dim=1)
        intra_loss = intra_loss.div(num_neighbor_intra)

        ins_loss = -score_intra.masked_select(mask_instance.bool()).log()
        # -------------------------- Inter-Camera Neighborhood Loss --------------------------#
        # Compute masks for inter-camera neighbors
        sim_inter = (sim.data + 1) * mask_inter - 1
        nearest_inter = sim_inter.max(dim=1, keepdim=True)[0]
        mask_neighbor_inter = torch.gt(sim_inter, nearest_inter * self.neighbor_eps)
        num_neighbor_inter = mask_neighbor_inter.sum(dim=1)+1
        # print('num_neighbor_inter',num_neighbor_inter)
        # Activate inter-camera candidates
        sim_exp_inter = sim_exp * mask_inter
        score_inter = F.softmax(sim_exp_inter,dim=1) #sim_exp_inter / sim_exp_inter.sum(dim=1, keepdim=True) #
        score_inter = score_inter.clamp_min(1e-8)
        inter_loss = -score_inter.log().mul(mask_neighbor_inter).sum(dim=1)
        inter_loss = inter_loss.div(num_neighbor_inter)

        for c in self.allcam:
            cam_wise = [int(c) for i in range(inputs.size(0))]
            mask_instance, mask_intra, mask_inter = self.compute_mask_camwise(sim.size(), indexes, cam_wise, sim.device)
            # -------------------------- Intra-camera Neighborhood Loss -------------------------- #
            # Compute masks for intra-camera neighbors
            # print('mask_instance.sum(1)',mask_instance.sum(1))
            # print('mask_intra.sum(1)',mask_intra.sum(1))
            # print('mask_inter.sum(1)',mask_inter.sum(1))
            # print('sim',sim)
            sim_intra = (sim.data + 1) * mask_intra * (1 - mask_instance) - 1
            # print('sim_intra.sum(1)',sim_intra.sum(1))
            nearest_intra = sim_intra.max(dim=1, keepdim=True)[0]
            # print('nearest_intra',nearest_intra)
            mask_neighbor_intra = torch.gt(sim_intra, nearest_intra * self.neighbor_eps)
            num_neighbor_intra = mask_neighbor_intra.sum(dim=1)+1
            # print('num_neighbor_intra',num_neighbor_intra)
            # Activate intra-camera candidates
            sim_exp_intra = sim_exp * mask_intra
            # print('sim_exp_intra',sim_exp_intra)
            score_intra =  F.softmax(sim_exp_intra,dim=1)# sim_exp_intra / sim_exp_intra.sum(dim=1, keepdim=True)# 
            # print('score_intra',score_intra)
            score_intra = score_intra.clamp_min(1e-8)
            # print('score_intra',score_intra)
            intrawise_loss = -score_intra.log().mul(mask_neighbor_intra).sum(dim=1)
            intrawise_loss = intrawise_loss.div(num_neighbor_intra)
            # print('score_intra,intra_loss',score_intra,intra_loss)
            if self.thresh >0:
                # Weighting intra-camera neighborhood consistency
                weight_intra = sim.data * mask_neighbor_intra
                weight_intra = weight_intra.sum(dim=1).div(num_neighbor_intra)
                weight_intra = torch.where(weight_intra > self.thresh, 1, 0)
                intrawise_loss = intrawise_loss.mul(weight_intra)
            intrawise_loss_total = intrawise_loss_total+intrawise_loss.mean()

        return ins_loss.mean(),intra_loss.mean(),inter_loss.mean()* 0.6,intrawise_loss_total* 0.6#ins_loss_total,intra_loss_total,inter_loss_total* 0.6#loss#,loss_cam inter_loss.mean()* 0.6
    def compute_mask(self, size, img_ids, cam_ids, device):
        # print('self.cam2uid',self.cam2uid)
        # print('cam_ids',cam_ids)
        mask_inter = torch.ones(size, device=device)
        for i, cam in enumerate(cam_ids.tolist()):
            intra_cam_ids = self.cam2uid[cam]
            # print(cam_ids)
            
            # print('intra_cam_ids',intra_cam_ids)
            mask_inter[i, intra_cam_ids] = 0

        mask_intra = 1 - mask_inter
        # print(mask_intra)
        mask_instance = torch.zeros(size, device=device)
        mask_instance[torch.arange(size[0]), img_ids] = 1
        return mask_instance, mask_intra, mask_inter

    def compute_mask_camwise(self, size, img_ids, cam_ids, device):
        # print('self.cam2uid',self.cam2uid)
        # print('cam_ids',cam_ids)
        mask_inter = torch.ones(size, device=device)
        for i, cam in enumerate(cam_ids):
            intra_cam_ids = self.cam2uid[cam]
            # print(cam_ids)
            
            # print('intra_cam_ids',intra_cam_ids)
            mask_inter[i, intra_cam_ids] = 0

        mask_intra = 1 - mask_inter
        # print(mask_intra)
        mask_instance = torch.zeros(size, device=device)
        mask_instance[torch.arange(size[0]), img_ids] = 1
        return mask_instance, mask_intra, mask_inter

def pairwise_distance(features_q, features_g):
    x = features_q#torch.from_numpy(features_q)
    y = features_g#torch.from_numpy(features_g)
    m, n = x.size(0), y.size(0)
    x = x.view(m, -1)
    y = y.view(n, -1)
    dist_m = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(m, n) + \
           torch.pow(y, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    dist_m.addmm_(1, -2, x, y.t())
    return dist_m#.numpy()


class Memory_wise_v2(nn.Module):
    def __init__(self, num_features, num_samples,num_cluster, temp=0.05, momentum=0.2):
        super(Memory_wise_v2, self).__init__()
        self.num_features = num_features
        self.num_samples = num_samples
        self.momentum = momentum
        self.temp = temp
        self.devices = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.register_buffer('features', torch.zeros(num_samples, num_features))
        # features--(source centers+tgt features)
        self.register_buffer('labels', torch.zeros(num_samples).int())
        self.register_buffer('cam', torch.zeros(num_samples).int())
        self.cam_mem = defaultdict(list)
        # labels--(each src and predicted tgt id and outliers), 13638
    def cam2uid(self):
        uid2cam = zip(range(self.num_samples), self.cam)
        self.cam2uid = defaultdict(list)
        for uid, cam in uid2cam:
            self.cam2uid[int(cam.cpu().data)].append(uid)
        # print(self.cam2uid)
        self.allcam = torch.unique(self.cam).cpu().numpy().tolist()
        print(self.allcam)
        # self.cam_mem = defaultdict(list)
    def cam_mem_gen(self):
        num_c_total=0
        for c in self.allcam:
            self.cam_mem[c],num_c = self.generate_cluster_features(self.labels,self.features,c)
            num_c_total= num_c_total+num_c
        print(num_c_total)
        # self.cam_mem = torch.cat([self.cam_mem[i] for i in self.allcam], 0).detach().data
        # self.cluster = self.generate_cluster_features_all(self.labels,self.features)


    def updateEM(self, inputs, indexes):
        # momentum update
        for x, y in zip(inputs, indexes):
            self.features[y] = self.momentum * self.features[y] + (1. - self.momentum) * x
            self.features[y] /= self.features[y].norm()

    def forward(self, inputs, indexes,cameras,neighbor_eps=0.9,refine=False,stage3=False):
        self.thresh=-1
        self.neighbor_eps  = neighbor_eps
        inputs = F.normalize(inputs, dim=1)#.cuda()

        # print(indexes)
        sim = em(inputs, indexes, self.features, self.momentum)#B N 
        sim_exp =sim /self.temp  # 64*13638
        B = inputs.size(0)
        mask_instance, mask_intra, mask_inter = self.compute_mask(sim.size(), indexes, cameras, sim.device)
        sim_exp_intra = sim_exp #* mask_intra
        # print('sim_exp_intra',sim_exp_intra)
        score_intra =   F.softmax(sim_exp_intra,dim=1)#sim_exp_intra / sim_exp_intra.sum(dim=1, keepdim=True)# 
        # print('score_intra',score_intra)
        score_intra = score_intra.clamp_min(1e-8)
        ins_loss = -score_intra.masked_select(mask_instance.bool()).log().mean()
        return ins_loss#* 0.6
    def compute_mask(self, size, img_ids, cam_ids, device):
        mask_inter = torch.ones(size, device=device)
        for i, cam in enumerate(cam_ids):
            intra_cam_ids = self.cam2uid[cam]
            mask_inter[i, intra_cam_ids] = 0

        mask_intra = 1 - mask_inter
        mask_instance = torch.zeros(size, device=device)
        mask_instance[torch.arange(size[0]), img_ids] = 1
        return mask_instance, mask_intra, mask_inter



    def compute_mask_camwise(self, size, img_ids, cam_ids, device):
        # print('self.cam2uid',self.cam2uid)
        # print('cam_ids',cam_ids)
        mask_intra = torch.zeros(size, device=device)
        for i, cam in enumerate(cam_ids):
            intra_cam_ids = self.cam2uid[cam]
            # print(cam_ids)
            
            # print('intra_cam_ids',intra_cam_ids)
            mask_intra[i, intra_cam_ids] = 1

        # mask_intra = 1 - mask_inter
        # print(mask_intra)
        mask_instance = torch.zeros(size, device=device)
        mask_instance[torch.arange(size[0]), img_ids] = 1
        return mask_intra,mask_instance

    def generate_cluster_features(self,labels, features,cam_id):
        centers = collections.defaultdict(list)
        for i, label in enumerate(self.labels):
            # print(int(self.cam[i]),int(cam_id))
            if (label == -1) or (int(self.cam[i]) != int(cam_id)):
                continue
            centers[int(label)].append(self.features[i])
            # print('cam label',self.cam[i],label)
        # print(centers)
        centers = [
            torch.stack(centers[idx], dim=0).mean(0) for idx in sorted(centers.keys())
        ]

        centers = torch.stack(centers, dim=0).cuda()
        print('cam cluster',cam_id,centers.size(0))
        return centers, centers.size(0)

    def generate_cluster_features_all(self,labels, features):
        centers = collections.defaultdict(list)
        for i, label in enumerate(labels):
            if (label == -1):
                continue
            centers[labels[i]].append(features[i])

        centers = [
            torch.stack(centers[idx], dim=0).mean(0) for idx in sorted(centers.keys())
        ]

        centers = torch.stack(centers, dim=0).cuda()
        return centers


class Memory_wise_v2_ori(nn.Module):
    def __init__(self, num_features, num_samples,num_cluster, temp=0.05, momentum=0.2):
        super(Memory_wise_v2_ori, self).__init__()
        self.num_features = num_features
        self.num_samples = num_samples
        self.momentum = momentum
        self.temp = temp
        self.devices = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.register_buffer('features', torch.zeros(num_samples, num_features))
        # features--(source centers+tgt features)
        self.register_buffer('labels', torch.zeros(num_samples).int())
        self.register_buffer('cam', torch.zeros(num_samples).int())
        self.cam_mem = defaultdict(list)
        # labels--(each src and predicted tgt id and outliers), 13638
    def cam2uid(self):
        uid2cam = zip(range(self.num_samples), self.cam)
        self.cam2uid = defaultdict(list)
        for uid, cam in uid2cam:
            self.cam2uid[int(cam.cpu().data)].append(uid)
        # print(self.cam2uid)
        self.allcam = torch.unique(self.cam).cpu().numpy().tolist()
        print(self.allcam)
        # self.cam_mem = defaultdict(list)
    def cam_mem_gen(self):
        num_c_total=0
        for c in self.allcam:
            self.cam_mem[c],num_c = self.generate_cluster_features(self.labels,self.features,c)
            num_c_total= num_c_total+num_c
        print(num_c_total)


    def updateEM(self, inputs, indexes):
        # momentum update
        for x, y in zip(inputs, indexes):
            self.features[y] = self.momentum * self.features[y] + (1. - self.momentum) * x
            self.features[y] /= self.features[y].norm()

    def forward(self, inputs, indexes,cameras,neighbor_eps=0.9,refine=False,stage3=False):
        self.thresh=-1
        self.neighbor_eps  = neighbor_eps
        inputs = F.normalize(inputs, dim=1)#.cuda()

        # print(indexes)
        sim = em(inputs, indexes, self.features, self.momentum)#B N 
        sim_exp =sim /self.temp  # 64*13638
        B = inputs.size(0)



        intrawise_loss_total=torch.tensor([0.]).cuda()
        inswise_loss_total =torch.tensor([0.]).cuda()
 
        for c in self.allcam:
            if stage3==True:
                cam_wise = 1-cameras
            else:
                cam_wise = [int(c) for i in range(inputs.size(0))]
            mask_intra,mask_instance = self.compute_mask_camwise(sim.size(), indexes, cam_wise, sim.device)
            # sim_wise = self.features.mm(F.normalize(self.cam_mem.detach().data, dim=1).t()) #N C
            # sim_wise = F.softmax(sim_wise.detach().data/self.temp,dim=1)
            sim_wise = torch.cat([F.softmax(self.features.mm(F.normalize(self.cam_mem[i].detach().data, dim=1).t()),dim=1) for i in self.allcam],dim=1).detach().data  #N C/0.05
            # sim_wise = F.softmax(self.features.mm(F.normalize(self.cam_mem[c].detach().data, dim=1).t())/0.05,dim=1).detach().data  #N C
            sim_wise_B = sim_wise[indexes]#B C
            sim_wise = F.normalize(sim_wise_B, dim=1).mm(F.normalize(sim_wise.t(),dim=1))#B N

            sim_intra = (sim.data + 1) * mask_intra * (1 - mask_instance) - 1
            sim_wise = (sim_wise.data + 1) * mask_intra * (1 - mask_instance) - 1

            # print('sim_intra.sum(1)',sim_intra.sum(1))
            nearest_intra = sim_intra.max(dim=1, keepdim=True)[0]
            sim_wise_max = sim_wise.max(dim=1, keepdim=True)[0]
            # print('nearest_intra',nearest_intra)
            # print('sim_wise_max',sim_wise_max)
            mask_neighbor_intra = torch.gt(sim_intra, nearest_intra * self.neighbor_eps)#nearest_intra * self.neighbor_eps)self.neighbor_eps
            sim_wise = torch.gt(sim_wise, sim_wise_max * self.neighbor_eps)
            num_neighbor_intra = mask_neighbor_intra.sum(dim=1)#.mul(sim_wise).
            num_neighbor_sim_wise = sim_wise.sum(dim=1)#+1

            sim_exp_intra = sim_exp# * mask_intra
            # print('sim_exp_intra',sim_exp_intra)
            score_intra =   F.softmax(sim_exp_intra,dim=1)
            score_intra = score_intra.clamp_min(1e-8)

            cam_id_count = mask_neighbor_intra.mul(sim_wise).sum(dim=1)+1e-8
 
            mask_neighbor_intra_soft = F.softmax(cam_id_count.float(),dim=-1)


            intrawise_loss = -score_intra.log().mul(mask_neighbor_intra).mul(sim_wise).sum(dim=1)
            intrawise_loss = intrawise_loss.div(cam_id_count).mul(mask_neighbor_intra_soft)


            intrawise_loss_total = intrawise_loss_total+intrawise_loss.sum()#.mean()#
            
        inswise_loss = -score_intra.masked_select(mask_instance.bool()).log()
        inswise_loss_total=inswise_loss.mean()#inswise_loss_total/len(self.allcam)
        intrawise_loss_total=intrawise_loss_total/len(self.allcam)

        return inswise_loss_total,intrawise_loss_total#* 0.6
    def compute_mask(self, size, img_ids, cam_ids, device):
        # print('self.cam2uid',self.cam2uid)
        # print('cam_ids',cam_ids)
        mask_inter = torch.ones(size, device=device)
        for i, cam in enumerate(cam_ids.tolist()):
            intra_cam_ids = self.cam2uid[cam]
            # print(cam_ids)
            # print('intra_cam_ids',intra_cam_ids)
            mask_inter[i, intra_cam_ids] = 0

        mask_intra = 1 - mask_inter
        # print(mask_intra)
        mask_instance = torch.zeros(size, device=device)
        mask_instance[torch.arange(size[0]), img_ids] = 1
        return mask_instance, mask_intra, mask_inter



    def compute_mask_camwise(self, size, img_ids, cam_ids, device):
        # print('self.cam2uid',self.cam2uid)
        # print('cam_ids',cam_ids)
        mask_intra = torch.zeros(size, device=device)
        for i, cam in enumerate(cam_ids):
            intra_cam_ids = self.cam2uid[cam]
            # print(cam_ids)
            
            # print('intra_cam_ids',intra_cam_ids)
            mask_intra[i, intra_cam_ids] = 1

        # mask_intra = 1 - mask_inter
        # print(mask_intra)
        mask_instance = torch.zeros(size, device=device)
        mask_instance[torch.arange(size[0]), img_ids] = 1
        return mask_intra,mask_instance

    def generate_cluster_features(self,labels, features,cam_id):
        centers = collections.defaultdict(list)
        for i, label in enumerate(self.labels):
            # print(int(self.cam[i]),int(cam_id))
            if (label == -1) or (int(self.cam[i]) != int(cam_id)):
                continue
            centers[int(label)].append(self.features[i])
            # print('cam label',self.cam[i],label)
        # print(centers)
        centers = [
            torch.stack(centers[idx], dim=0).mean(0) for idx in sorted(centers.keys())
        ]

        centers = torch.stack(centers, dim=0).cuda()
        print('cam cluster',cam_id,centers.size(0))
        return centers, centers.size(0)

    def generate_cluster_features_all(self,labels, features):
        centers = collections.defaultdict(list)
        for i, label in enumerate(labels):
            if (label == -1):
                continue
            centers[labels[i]].append(features[i])

        centers = [
            torch.stack(centers[idx], dim=0).mean(0) for idx in sorted(centers.keys())
        ]

        centers = torch.stack(centers, dim=0).cuda()
        return centers



class Memory_wise_v2_ori(nn.Module):
    def __init__(self, num_features, num_samples,num_cluster, temp=0.05, momentum=0.2):
        super(Memory_wise_v2_ori, self).__init__()
        self.num_features = num_features
        self.num_samples = num_samples
        self.momentum = momentum
        self.temp = temp
        self.devices = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.register_buffer('features', torch.zeros(num_samples, num_features))
        # features--(source centers+tgt features)
        self.register_buffer('labels', torch.zeros(num_samples).int())
        self.register_buffer('cam', torch.zeros(num_samples).int())
        self.cam_mem = defaultdict(list)
        # labels--(each src and predicted tgt id and outliers), 13638
    def cam2uid(self):
        uid2cam = zip(range(self.num_samples), self.cam)
        self.cam2uid = defaultdict(list)
        for uid, cam in uid2cam:
            self.cam2uid[int(cam.cpu().data)].append(uid)
        # print(self.cam2uid)
        self.allcam = torch.unique(self.cam).cpu().numpy().tolist()
        print(self.allcam)
        # self.cam_mem = defaultdict(list)
    def cam_mem_gen(self):
        num_c_total=0
        for c in self.allcam:
            self.cam_mem[c],num_c = self.generate_cluster_features(self.labels,self.features,c)
            num_c_total= num_c_total+num_c
        print(num_c_total)


    def updateEM(self, inputs, indexes):
        # momentum update
        for x, y in zip(inputs, indexes):
            self.features[y] = self.momentum * self.features[y] + (1. - self.momentum) * x
            self.features[y] /= self.features[y].norm()

    def forward(self, inputs, indexes,cameras,neighbor_eps=0.9,refine=False,stage3=False):
        self.thresh=-1
        self.neighbor_eps  = neighbor_eps
        inputs = F.normalize(inputs, dim=1)#.cuda()

        # print(indexes)
        sim = em(inputs, indexes, self.features, self.momentum)#B N 
        sim_exp =sim /self.temp  # 64*13638
        B = inputs.size(0)



        intrawise_loss_total=torch.tensor([0.]).cuda()
        inswise_loss_total =torch.tensor([0.]).cuda()

        for c in self.allcam:
            if stage3==True:
                cam_wise = 1-cameras
            else:
                cam_wise = [int(c) for i in range(inputs.size(0))]
            mask_intra,mask_instance = self.compute_mask_camwise(sim.size(), indexes, cam_wise, sim.device)
            # sim_wise = self.features.mm(F.normalize(self.cam_mem.detach().data, dim=1).t()) #N C
            # sim_wise = F.softmax(sim_wise.detach().data/self.temp,dim=1)
            sim_wise = torch.cat([F.softmax(self.features.mm(F.normalize(self.cam_mem[i].detach().data, dim=1).t()),dim=1) for i in self.allcam],dim=1).detach().data  #N C/0.05
            # sim_wise = F.softmax(self.features.mm(F.normalize(self.cam_mem[c].detach().data, dim=1).t())/0.05,dim=1).detach().data  #N C
            sim_wise_B = sim_wise[indexes]#B C
            sim_wise = F.normalize(sim_wise_B, dim=1).mm(F.normalize(sim_wise.t(),dim=1))#B N
            # sim_wise = pairwise_distance(sim_wise_B,sim_wise)
            # sim_wise = F.softmax(sim_wise.detach().data/self.temp,dim=1)
            # sim_wise = sim_wise[indexes]
            # print('sim_wise',sim_wise.size())
            # -------------------------- Intra-camera Neighborhood Loss -------------------------- #
            # Compute masks for intra-camera neighbors
            # print('mask_instance.sum(1)',mask_instance.sum(1))
            # print('mask_intra.sum(1)',mask_intra.sum(1))
            # print('mask_inter.sum(1)',mask_inter.sum(1))
            # print('sim',sim)
            sim_intra = (sim.data + 1) * mask_intra * (1 - mask_instance) - 1
            sim_wise = (sim_wise.data + 1) * mask_intra * (1 - mask_instance) - 1
#########################
            # topk, indices_nearest_intra = torch.topk(sim_intra, 20)#20
            # mask_neighbor_intra = torch.zeros_like(sim_intra)
            # mask_neighbor_intra = mask_neighbor_intra.scatter(1, indices_nearest_intra, 1)

            # topk, indices_sim_wise = torch.topk(sim_wise, 20)#20
            # mask_sim_wise = torch.zeros_like(sim_intra)
            # sim_wise = mask_sim_wise.scatter(1, indices_sim_wise, 1)
#########################
            # print('sim_intra.sum(1)',sim_intra.sum(1))
            nearest_intra = sim_intra.max(dim=1, keepdim=True)[0]
            sim_wise_max = sim_wise.max(dim=1, keepdim=True)[0]
            # print('nearest_intra',nearest_intra)
            # print('sim_wise_max',sim_wise_max)
            mask_neighbor_intra = torch.gt(sim_intra, nearest_intra * self.neighbor_eps)#nearest_intra * self.neighbor_eps)self.neighbor_eps
            sim_wise = torch.gt(sim_wise, sim_wise_max * self.neighbor_eps)#sim_wise_max * self.neighbor_eps)
            ####################
            num_neighbor_intra = mask_neighbor_intra.sum(dim=1)#.mul(sim_wise).
            num_neighbor_sim_wise = sim_wise.sum(dim=1)#+1
            # print('ori num_neighbor_intra',num_neighbor_intra)
            # print('num_neighbor_sim_wise',num_neighbor_sim_wise)
            # Activate intra-camera candidates
            # num_neighbor_intra = mask_neighbor_intra.mul(sim_wise).sum(dim=1)+1#sim_wise.sum(dim=1)+1#
            sim_exp_intra = sim_exp# * mask_intra
            # print('sim_exp_intra',sim_exp_intra)
            score_intra =   F.softmax(sim_exp_intra,dim=1)##sim_exp_intra / sim_exp_intra.sum(dim=1, keepdim=True)# 
            # print('score_intra',score_intra)
            score_intra = score_intra.clamp_min(1e-8)
            # print('score_intra',score_intra)
            # print('sim_wise',sim_wise.size())
            # print('mask_neighbor_intra',mask_neighbor_intra.size())
            # print('mask_neighbor_intra',mask_neighbor_intra.sum(dim=1).view(-1))
            # print('sim_wise',sim_wise.sum(dim=1).view(-1))
            cam_id_count = mask_neighbor_intra.mul(sim_wise).sum(dim=1)+1e-8
            # print('cameras',cameras)
            # print('c',c,cam_id_count)
            # print()
            # intrawise_loss = -score_intra.log().mul(mask_neighbor_intra).sum(dim=1)#.mul(sim_wise)
            # print('cam_id_count',cam_id_count)
            mask_neighbor_intra_soft = F.softmax(cam_id_count.float(),dim=-1)
            # print('mask_neighbor_intra_soft',mask_neighbor_intra_soft)

            intrawise_loss = -score_intra.log().mul(mask_neighbor_intra).mul(sim_wise).sum(dim=1)#.mul(sim_wise) mul(mask_neighbor_intra)
            intrawise_loss = intrawise_loss.div(cam_id_count).mul(mask_neighbor_intra_soft) ##

##################
            # intrcam_mask = cameras.eq(cam_wise).float()
            # intercam_mask = 1-intrcam_mask

            # intrawise_loss_inter = -score_intra.log().mul(mask_neighbor_intra).mul(sim_wise).sum(dim=1)#.mul(sim_wise) mul(mask_neighbor_intra)
            # intrawise_loss_inter = intrawise_loss.div(num_neighbor_intra)#.mul(mask_neighbor_intra_soft) 

            # intrawise_loss_intra = -score_intra.log().mul(mask_neighbor_intra).sum(dim=1)#.mul(sim_wise) mul(mask_neighbor_intra)
            # intrawise_loss_intra = intrawise_loss.div(cam_id_count)#.mul(mask_neighbor_intra_soft) 

            # intrawise_loss = intrawise_loss_intra*intrcam_mask+intrawise_loss_inter*intrcam_mask
#######################
            # intrawise_loss = intrawise_loss.mul(mask_neighbor_intra_soft) 
            # intrawise_loss = -score_intra.log().mul(sim_wise).sum(dim=1)
            # intrawise_loss = intrawise_loss.div(num_neighbor_sim_wise)
            # print('score_intra,intra_loss',score_intra,intra_loss)
            # if self.thresh >0:
            #     # Weighting intra-camera neighborhood consistency
            #     weight_intra = sim.data * mask_neighbor_intra
            #     weight_intra = weight_intra.sum(dim=1).div(num_neighbor_intra)
            #     weight_intra = torch.where(weight_intra > self.thresh, 1, 0)
            #     intrawise_loss = intrawise_loss.mul(weight_intra)
            # intrawise_loss = intrawise_loss
            intrawise_loss_total = intrawise_loss_total+intrawise_loss.sum()#.mean()#

        inswise_loss = -score_intra.masked_select(mask_instance.bool()).log()
        inswise_loss_total=inswise_loss.mean()#inswise_loss_total/len(self.allcam)
        intrawise_loss_total=intrawise_loss_total/len(self.allcam)
        # inter_loss_total=inter_loss_total/len(self.allcam)
        # if refine==True:
        #     return inswise_loss_total,intrawise_loss_total* 0.6,pseudo_labels_rgb_cm
        # else:
        return inswise_loss_total,intrawise_loss_total#* 0.6
    def compute_mask(self, size, img_ids, cam_ids, device):
        # print('self.cam2uid',self.cam2uid)
        # print('cam_ids',cam_ids)
        mask_inter = torch.ones(size, device=device)
        for i, cam in enumerate(cam_ids.tolist()):
            intra_cam_ids = self.cam2uid[cam]
            # print(cam_ids)
            # print('intra_cam_ids',intra_cam_ids)
            mask_inter[i, intra_cam_ids] = 0

        mask_intra = 1 - mask_inter
        # print(mask_intra)
        mask_instance = torch.zeros(size, device=device)
        mask_instance[torch.arange(size[0]), img_ids] = 1
        return mask_instance, mask_intra, mask_inter


    def compute_mask_camwise(self, size, img_ids, cam_ids, device):
        # print('self.cam2uid',self.cam2uid)
        # print('cam_ids',cam_ids)
        mask_intra = torch.zeros(size, device=device)
        for i, cam in enumerate(cam_ids):
            intra_cam_ids = self.cam2uid[cam]
            # print(cam_ids)
            
            # print('intra_cam_ids',intra_cam_ids)
            mask_intra[i, intra_cam_ids] = 1

        # mask_intra = 1 - mask_inter
        # print(mask_intra)
        mask_instance = torch.zeros(size, device=device)
        mask_instance[torch.arange(size[0]), img_ids] = 1
        return mask_intra,mask_instance

    def generate_cluster_features(self,labels, features,cam_id):
        centers = collections.defaultdict(list)
        for i, label in enumerate(self.labels):
            # print(int(self.cam[i]),int(cam_id))
            if (label == -1) or (int(self.cam[i]) != int(cam_id)):
                continue
            centers[int(label)].append(self.features[i])
            # print('cam label',self.cam[i],label)
        # print(centers)
        centers = [
            torch.stack(centers[idx], dim=0).mean(0) for idx in sorted(centers.keys())
        ]

        centers = torch.stack(centers, dim=0).cuda()
        print('cam cluster',cam_id,centers.size(0))
        return centers, centers.size(0)

    def generate_cluster_features_all(self,labels, features):
        centers = collections.defaultdict(list)
        for i, label in enumerate(labels):
            if (label == -1):
                continue
            centers[labels[i]].append(features[i])

        centers = [
            torch.stack(centers[idx], dim=0).mean(0) for idx in sorted(centers.keys())
        ]

        centers = torch.stack(centers, dim=0).cuda()
        return centers


class Memory_wise_v3(nn.Module):
    def __init__(self, num_features, num_samples,num_cluster, temp=0.05, momentum=0.2):
        super(Memory_wise_v3, self).__init__()
        self.num_features = num_features
        self.num_samples = num_samples
        self.momentum = momentum
        self.temp = temp
        self.devices = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.register_buffer('features', torch.zeros(num_samples, num_features))
        # features--(source centers+tgt features)
        self.register_buffer('labels', torch.zeros(num_samples).int())
        self.register_buffer('cam', torch.zeros(num_samples).int())
        self.cam_mem = defaultdict(list)
        # labels--(each src and predicted tgt id and outliers), 13638
    def cam2uid_(self):
        uid2cam = zip(range(self.num_samples), self.cam)
        self.cam2uid = defaultdict(list)
        for uid, cam in uid2cam:
            self.cam2uid[int(cam.cpu().data)].append(uid)
        # print(self.cam2uid)
        self.allcam = torch.unique(self.cam).cpu().numpy().tolist()
        print(self.allcam)
        # self.cam_mem = defaultdict(list)
    def cam_mem_gen(self):
        num_c_total=0
        for c in self.allcam:
            self.cam_mem[c],num_c = self.generate_cluster_features(self.labels,self.features,c)
            num_c_total= num_c_total+num_c
        print(num_c_total)
        # self.cam_mem = torch.cat([self.cam_mem[i] for i in self.allcam], 0).detach().data
        # self.cluster = self.generate_cluster_features_all(self.labels,self.features)


    def updateEM(self, inputs, indexes):
        # momentum update
        for x, y in zip(inputs, indexes):
            self.features[y] = self.momentum * self.features[y] + (1. - self.momentum) * x
            self.features[y] = self.features[y]/self.features[y].norm()


    def forward(self, inputs, indexes,cameras,neighbor_eps=0.9):
        self.thresh=-1
        self.neighbor_eps  = neighbor_eps
        inputs = F.normalize(inputs, dim=1)#.cuda()

        # print(indexes)
        sim = em(inputs, indexes, self.features, self.momentum)#B N 
        sim_exp =sim /self.temp  # 64*13638
        B = inputs.size(0)
        mask_instance = self.compute_mask(sim.size(), indexes,device=sim.device)
        sim_exp_intra = sim_exp #* mask_intra
        # print('sim_exp_intra',sim_exp_intra)
        score_intra =   F.softmax(sim_exp_intra,dim=1)#sim_exp_intra / sim_exp_intra.sum(dim=1, keepdim=True)# 
        # print('score_intra',score_intra)
        score_intra = score_intra.clamp_min(1e-8)
        ins_loss = -score_intra.masked_select(mask_instance.bool()).log().mean()
        return ins_loss#* 0.6
    def compute_mask(self, size, img_ids,device=None):
        # mask_inter = torch.ones(size, device=device)
        # for i, cam in enumerate(cam_ids):
        #     intra_cam_ids = self.cam2uid[cam]
        #     mask_inter[i, intra_cam_ids] = 0

        # mask_intra = 1 - mask_inter
        mask_instance = torch.zeros(size, device=device)
        mask_instance[torch.arange(size[0]), img_ids] = 1
        return mask_instance#, mask_intra, mask_inter



    def compute_mask_camwise(self, size, img_ids, cam_ids, device):
        # print('self.cam2uid',self.cam2uid)
        # print('cam_ids',cam_ids)
        mask_intra = torch.ones(size, device=device)#zeros
        for i, cam in enumerate(cam_ids):
            intra_cam_ids = self.cam2uid[cam]
            # print(cam_ids)
            
            # print('intra_cam_ids',intra_cam_ids)
            mask_intra[i, intra_cam_ids] = 1

        # mask_intra = 1 - mask_inter
        # print(mask_intra)
        mask_instance = torch.zeros(size, device=device)
        mask_instance[torch.arange(size[0]), img_ids] = 1
        return mask_intra,mask_instance

    def generate_cluster_features(self,labels, features,cam_id):
        centers = collections.defaultdict(list)
        for i, label in enumerate(self.labels):
            # print(int(self.cam[i]),int(cam_id))
            if (label == -1) or (int(self.cam[i]) != int(cam_id)):
                continue
            centers[int(label)].append(self.features[i])
            # print('cam label',self.cam[i],label)
        # print(centers)
        centers = [
            torch.stack(centers[idx], dim=0).mean(0) for idx in sorted(centers.keys())
        ]

        centers = torch.stack(centers, dim=0).cuda()
        print('cam cluster',cam_id,centers.size(0))
        return centers, centers.size(0)

    def generate_cluster_features_all(self,labels, features):
        centers = collections.defaultdict(list)
        for i, label in enumerate(labels):
            if (label == -1):
                continue
            centers[labels[i]].append(features[i])

        centers = [
            torch.stack(centers[idx], dim=0).mean(0) for idx in sorted(centers.keys())
        ]

        centers = torch.stack(centers, dim=0).cuda()
        return centers

class Memory_wise_vbatch(nn.Module):
    def __init__(self, num_features, num_samples,num_cluster, temp=0.05, momentum=0.2):
        super(Memory_wise_vbatch, self).__init__()
        self.num_features = num_features
        self.num_samples = num_samples
        self.momentum = momentum
        self.temp = temp
        self.devices = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.register_buffer('features', torch.zeros(num_samples, num_features))
        # features--(source centers+tgt features)
        self.register_buffer('labels', torch.zeros(num_samples).int())
        self.register_buffer('cam', torch.zeros(num_samples).int())
        self.cam_mem = defaultdict(list)
        # num_encoder_layers=4
        # trans_layer = nn.TransformerEncoderLayer(d_model=2048,nhead=8) 
        # self.encoder=nn.TransformerEncoder(trans_layer,num_layers=num_encoder_layers)
        # labels--(each src and predicted tgt id and outliers), 13638
    def cam2uid(self):
        uid2cam = zip(range(self.num_samples), self.cam)
        self.cam2uid = defaultdict(list)
        for uid, cam in uid2cam:
            self.cam2uid[int(cam.cpu().data)].append(uid)
        # print(self.cam2uid)
        self.allcam = torch.unique(self.cam).cpu().numpy().tolist()
        print(self.allcam)
        # self.cam_mem = defaultdict(list)
    def cam_mem_gen(self):
        num_c_total=0
        for c in self.allcam:
            self.cam_mem[c],num_c = self.generate_cluster_features(self.labels,self.features,c)
            num_c_total= num_c_total+num_c
        print(num_c_total)
        # self.cam_mem = torch.cat([self.cam_mem[i] for i in self.allcam], 0).detach().data
        # self.cluster = self.generate_cluster_features_all(self.labels,self.features)


    def updateEM(self, inputs, indexes):
        # momentum update
        for x, y in zip(inputs, indexes):
            self.features[y] = self.momentum * self.features[y] + (1. - self.momentum) * x
            self.features[y] /= self.features[y].norm()

    def forward(self, inputs, indexes,cameras,neighbor_eps=0.9,refine=False,stage3=False):
        self.thresh=-1
        self.neighbor_eps  = neighbor_eps
        inputs = F.normalize(inputs, dim=1)#.cuda()

        # print(indexes)
        sim = em(inputs, indexes, self.features, self.momentum)#B N 
        sim_exp =sim /self.temp  # 64*13638
        B = inputs.size(0)


        intrawise_loss_total=torch.tensor([0.]).cuda()
        inswise_loss_total =torch.tensor([0.]).cuda()
        topk_list=[]
        for c in self.allcam:
            if stage3==True:
                cam_wise = 1-cameras
            else:
                cam_wise = [int(c) for i in range(inputs.size(0))]
            mask_intra,mask_instance = self.compute_mask_camwise(sim.size(), indexes, cam_wise, sim.device)
            # sim_wise = self.features.mm(F.normalize(self.cam_mem.detach().data, dim=1).t()) #N C
            # sim_wise = F.softmax(sim_wise.detach().data/self.temp,dim=1)
            sim_wise = torch.cat([F.softmax(self.features.mm(F.normalize(self.cam_mem[i].detach().data, dim=1).t())/self.temp,dim=1) for i in self.allcam],dim=1).detach().data  #N C
            sim_wise_B = sim_wise[indexes]#B C
            sim_wise = sim_wise_B.mm(sim_wise.t())#B N

            sim_intra = (sim.data + 1) * mask_intra * (1 - mask_instance) - 1


            topk, indices_sim_wise = torch.topk(sim_wise, 1)#20
            topk_list.append(indices_sim_wise.view(-1))

            nearest_intra = sim_intra.max(dim=1, keepdim=True)[0]
            sim_wise_max = sim_wise.max(dim=1, keepdim=True)[0]
            # print('nearest_intra',nearest_intra)
            mask_neighbor_intra = torch.gt(sim_intra, nearest_intra * self.neighbor_eps)
            sim_wise = torch.gt(sim_wise, sim_wise_max * self.neighbor_eps)
            ####################
            num_neighbor_intra = mask_neighbor_intra.mul(sim_wise).sum(dim=1)+1
            sim_exp_intra = sim_exp * mask_intra

            score_intra =   sim_exp_intra / sim_exp_intra.sum(dim=1, keepdim=True)# 
            score_intra = score_intra.clamp_min(1e-8)

            
            intrawise_loss = -score_intra.log().mul(mask_neighbor_intra).mul(sim_wise).sum(dim=1)
            intrawise_loss = intrawise_loss.div(num_neighbor_intra)
            intrawise_loss_total = intrawise_loss_total+intrawise_loss.mean()
 
        inswise_loss = -score_intra.masked_select(mask_instance.bool()).log()
        inswise_loss_total=inswise_loss.mean()#inswise_loss_total/len(self.allcam)
        intrawise_loss_total=intrawise_loss_total/len(self.allcam)

        return inswise_loss_total,intrawise_loss_total#,trans_feat#,self.labels[indexes]#* 0.6
    def compute_mask(self, size, img_ids, cam_ids, device):
        # print('self.cam2uid',self.cam2uid)
        # print('cam_ids',cam_ids)
        mask_inter = torch.ones(size, device=device)
        for i, cam in enumerate(cam_ids.tolist()):
            intra_cam_ids = self.cam2uid[cam]
            # print(cam_ids)
            # print('intra_cam_ids',intra_cam_ids)
            mask_inter[i, intra_cam_ids] = 0

        mask_intra = 1 - mask_inter
        # print(mask_intra)
        mask_instance = torch.zeros(size, device=device)
        mask_instance[torch.arange(size[0]), img_ids] = 1
        return mask_instance, mask_intra, mask_inter





    def compute_mask_camwise(self, size, img_ids, cam_ids, device):
        # print('self.cam2uid',self.cam2uid)
        # print('cam_ids',cam_ids)
        mask_intra = torch.ones(size, device=device)
        for i, cam in enumerate(cam_ids):
            intra_cam_ids = self.cam2uid[cam]
            # print(cam_ids)
            
            # print('intra_cam_ids',intra_cam_ids)
            mask_intra[i, intra_cam_ids] = 1

        # mask_intra = 1 - mask_inter
        # print(mask_intra)
        mask_instance = torch.zeros(size, device=device)
        mask_instance[torch.arange(size[0]), img_ids] = 1
        return mask_intra,mask_instance

    def generate_cluster_features(self,labels, features,cam_id):
        centers = collections.defaultdict(list)
        for i, label in enumerate(self.labels):
            # print(int(self.cam[i]),int(cam_id))
            if (label == -1) or (int(self.cam[i]) != int(cam_id)):
                continue
            centers[int(label)].append(self.features[i])
            # print('cam label',self.cam[i],label)
        # print(centers)
        centers = [
            torch.stack(centers[idx], dim=0).mean(0) for idx in sorted(centers.keys())
        ]

        centers = torch.stack(centers, dim=0).cuda()
        print('cam cluster',cam_id,centers.size(0))
        return centers, centers.size(0)

    def generate_cluster_features_all(self,labels, features):
        centers = collections.defaultdict(list)
        for i, label in enumerate(labels):
            if (label == -1):
                continue
            centers[labels[i]].append(features[i])

        centers = [
            torch.stack(centers[idx], dim=0).mean(0) for idx in sorted(centers.keys())
        ]

        centers = torch.stack(centers, dim=0).cuda()
        return centers