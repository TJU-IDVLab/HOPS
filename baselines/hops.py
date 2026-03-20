# Copyright (c) Facebook, Inc. and its affiliates.
from typing import Tuple
import gc
import torch
from torch import nn
from torch.nn import functional as F
from detectron2.config import configurable
from detectron2.data import MetadataCatalog
from detectron2.modeling import META_ARCH_REGISTRY, build_backbone, build_sem_seg_head
from detectron2.modeling.backbone import Backbone
from detectron2.modeling.postprocessing import sem_seg_postprocess
from detectron2.structures import ImageList
from collections import defaultdict
import cv2
import numpy as np


@META_ARCH_REGISTRY.register()
class Hops(nn.Module):
    @configurable
    def __init__(
        self,
        *,
        backbone: Backbone,
        sem_seg_head: nn.Module,
        size_divisibility: int,
        pixel_mean: Tuple[float],
        pixel_std: Tuple[float],
        clip_pixel_mean: Tuple[float],
        clip_pixel_std: Tuple[float],
        clip_finetune: str,
        clip_pretrained: str,
        train_dataset: str,
        test_dataset: str,
    ):
        """
        Args:
            backbone: a backbone module, must follow detectron2's backbone interface
            sem_seg_head: a module that predicts semantic segmentation from backbone features
        """
        super().__init__()
        
        self.backbone = backbone
        self.sem_seg_head = sem_seg_head
        if size_divisibility < 0:
            size_divisibility = self.backbone.size_divisibility
        self.size_divisibility = size_divisibility

        self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1), False)
        self.register_buffer("clip_pixel_mean", torch.Tensor(clip_pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("clip_pixel_std", torch.Tensor(clip_pixel_std).view(-1, 1, 1), False)

        self.ignore_label = MetadataCatalog.get(test_dataset).ignore_label
        self.init_metadata(train_dataset, "train")
        self.init_metadata(test_dataset, "test")
        self.device = "cuda"

        self.clip_finetune = clip_finetune
        for name, params in self.sem_seg_head.predictor.clip_model.named_parameters():
            if "visual" in name:
                if clip_finetune == "prompt":
                    params.requires_grad = True if "prompt" in name else False
                elif clip_finetune == "attention":
                    # if "attn" in name or "position" in name:
                    #     print(name,params.shape)
                    params.requires_grad = True if "attn" in name or "position" in name else False
                elif clip_finetune == "full":
                    params.requires_grad = True
                else:
                    params.requires_grad = False
            else:
                params.requires_grad = False

        self.sem_seg_head.predictor.transformer.conv2.load_state_dict(self.sem_seg_head.predictor.transformer.conv1.state_dict())
        self.sem_seg_head.predictor.transformer.layers_object.load_state_dict(self.sem_seg_head.predictor.transformer.layers.state_dict())
        self.sem_seg_head.predictor.transformer.layers_specific_part.load_state_dict(self.sem_seg_head.predictor.transformer.layers.state_dict())

        self.sem_seg_head.predictor.transformer.conv2.requires_grad_(True)
        self.sem_seg_head.predictor.transformer.layers_object.requires_grad_(False)
        self.sem_seg_head.predictor.transformer.layers_specific_part.requires_grad_(True)

        self.clip_resolution = (384, 384) if clip_pretrained == "ViT-B/16" else (336, 336)
        self.sequential = False
        self.lambda_part = 1.0
        self.lambda_obj = 1.0
        self.lambda_jsd = 1.0

    @classmethod
    def from_config(cls, cfg):
        backbone = build_backbone(cfg)
        sem_seg_head = build_sem_seg_head(cfg, backbone.output_shape())

        return {
            "backbone": backbone,
            "sem_seg_head": sem_seg_head,
            "size_divisibility": cfg.MODEL.MASK_FORMER.SIZE_DIVISIBILITY,
            "pixel_mean": cfg.MODEL.PIXEL_MEAN,
            "pixel_std": cfg.MODEL.PIXEL_STD,
            "clip_pixel_mean": cfg.MODEL.CLIP_PIXEL_MEAN,
            "clip_pixel_std": cfg.MODEL.CLIP_PIXEL_STD,
            "clip_finetune": cfg.MODEL.SEM_SEG_HEAD.CLIP_FINETUNE,
            "clip_pretrained": cfg.MODEL.SEM_SEG_HEAD.CLIP_PRETRAINED,
            "train_dataset": cfg.DATASETS.TRAIN[0],
            "test_dataset": cfg.DATASETS.TEST[0],
        }

    def init_metadata(self, dataset_name, prefix):
        text_classes = MetadataCatalog.get(dataset_name).stuff_classes
        setattr(self, f"{prefix}_ori_text_classes", text_classes)

        # Create generalized and object-specific classes
        setattr(self, f"{prefix}_text_classes", [c.replace("'s", "") for c in text_classes])
        setattr(self, f"{prefix}_obj_classes", MetadataCatalog.get(dataset_name).obj_classes)
        part_classes = sorted(list(set([c.split("'s")[1].strip() for c in text_classes])))
        setattr(self, f"{prefix}_part_classes", part_classes)
        obj_in_part_classes = sorted(list(set([c.split("'s")[0].strip() for c in text_classes])))
        setattr(self, f"{prefix}_obj_in_part_classes", obj_in_part_classes)

        # Maps for text to part, object to object-in-part, etc.
        text_to_part_map = torch.full(
            (self.ignore_label + 1,),
            self.ignore_label,
            dtype=torch.long,
            device=self.device
        )
        setattr(self, f"{prefix}_text_to_part_map", text_to_part_map)

        obj_to_obj_in_part_map = torch.full(
            (self.ignore_label + 1,),
            self.ignore_label,
            dtype=torch.long,
            device=self.device
        )
        setattr(self, f"{prefix}_obj_to_obj_in_part_map", obj_to_obj_in_part_map)

        text_to_obj_in_part_map = torch.full(
            (len(getattr(self, f"{prefix}_text_classes")),),
            len(getattr(self, f"{prefix}_text_classes")),
            dtype=torch.long,
            device=self.device
        )
        setattr(self, f"{prefix}_text_to_obj_in_part_map", text_to_obj_in_part_map)

        # Create class to part and object to object-in-part mappings
        class_to_part = {
            index: part_classes.index(class_text.split("'s")[1].strip())
            for index, class_text in enumerate(text_classes)
        }
        setattr(self, f"{prefix}_class_to_part", class_to_part)
        for index, part_index in class_to_part.items():
            getattr(self, f"{prefix}_text_to_part_map")[index] = part_index

        obj_to_obj_in_part = {}
        for index, class_text in enumerate(getattr(self, f"{prefix}_obj_classes")):
            if class_text in obj_in_part_classes:
                obj_to_obj_in_part[index] = obj_in_part_classes.index(class_text)
            else:
                obj_to_obj_in_part[index] = self.ignore_label
        setattr(self, f"{prefix}_obj_to_obj_in_part", obj_to_obj_in_part)
        for index, obj_in_part_index in obj_to_obj_in_part.items():
            getattr(self, f"{prefix}_obj_to_obj_in_part_map")[index] = obj_in_part_index

        # Create object-in-part to text mapping
        obj_in_part_to_text = defaultdict(list)
        for index, class_text in enumerate(text_classes):
            obj_class, _ = class_text.split("'s", maxsplit=1)
            obj_in_part_index = obj_in_part_classes.index(obj_class)
            obj_in_part_to_text[obj_in_part_index].append(index)
            getattr(self, f"{prefix}_text_to_obj_in_part_map")[index] = obj_in_part_index
        setattr(self, f"{prefix}_obj_in_part_to_text", obj_in_part_to_text)

        return None

    @property
    def device(self):
        return self.pixel_mean.device

    @device.setter
    def device(self, value):
        self.pixel_mean = self.pixel_mean.to(value)
        self.pixel_std = self.pixel_std.to(value)
        self.clip_pixel_mean = self.clip_pixel_mean.to(value)
        self.clip_pixel_std = self.clip_pixel_std.to(value)
        for prefix in ["train", "test"]:
            setattr(self, f"{prefix}_text_to_part_map", getattr(self, f"{prefix}_text_to_part_map").to(value))
            setattr(self, f"{prefix}_obj_to_obj_in_part_map", getattr(self, f"{prefix}_obj_to_obj_in_part_map").to(value))
            setattr(self, f"{prefix}_text_to_obj_in_part_map", getattr(self, f"{prefix}_text_to_obj_in_part_map").to(value))

    def jensen_shannon_divergence(self, p, q, dim, eps=1e-10):
        p = p + eps
        q = q + eps
        m = 0.5 * (p + q)
        jsd = 0.5 * (p * (p.log() - m.log())).sum(dim=dim) + 0.5 * (q * (q.log() - m.log())).sum(dim=dim)
        return jsd.mean()
    

    def postprocess_obj_predictions(self, output_obj, batched_inputs, H_1, W_1):
        """
        对模型预测的output_obj进行后处理
        Args:
            output_obj: 模型预测的obj类别logits,维度为(b, c, H_1, W_1)，已经调整好分辨率
            batched_inputs: 批次输入数据
            H_1: 原始图像高度
            W_1: 原始图像宽度
        Returns:
            list: 包含每个类别的mask、矩形框坐标和裁剪图像的结果
        """
        results = []
        
        # 在GPU上进行概率转换和预测
        obj_probs = F.softmax(output_obj, dim=1)  # (b, c, H_1, W_1) torch.Size([b, 12, 384, 384])
        pred_classes = torch.argmax(obj_probs, dim=1)  # (b, H_1, W_1) torch.Size([b, 384, 384])
        
        # 对每个batch进行处理
        for batch_idx in range(output_obj.shape[0]):
            batch_results = []
            
            # 获取原始图像（RGB图像，不需要调整分辨率）
            original_image = batched_inputs[batch_idx]["image"]  # (C, H, W) tensor
            if original_image.device != self.device:
                original_image = original_image.to(self.device)
            
            # 检查原始图像尺寸与预测结果尺寸的一致性
            if original_image.shape[1] != H_1 or original_image.shape[2] != W_1:
                # 如果尺寸不一致，需要调整原始图像尺寸
                original_image = F.interpolate(
                    original_image.unsqueeze(0), 
                    size=(H_1, W_1), 
                    mode='bilinear', 
                    align_corners=False
                ).squeeze(0)
            
            # 保持tensor格式，转换为(H, W, C)格式
            # original_image_hwc = original_image.permute(1, 2, 0)  # (H, W, C) tensor
            
            # 获取该batch的预测结果
            batch_pred_classes = pred_classes[batch_idx]  # (H_1, W_1)
            
            # 对每个类别进行处理(去除背景类别)
            num_classes = output_obj.shape[1] - 1
            for class_id in range(num_classes):
                # 在GPU上提取该类别的mask
                class_mask_gpu = (batch_pred_classes == class_id)  # (H_1, W_1) bool tensor
                
                # 如果该类别没有预测区域，跳过
                if class_mask_gpu.sum() == 0:
                    continue
                
                # 在GPU上计算边界框坐标
                y_coords, x_coords = torch.where(class_mask_gpu)
                
                if len(y_coords) == 0:
                    continue
                
                # 在GPU上计算包含所有该类别像素的最小矩形
                x_min = torch.min(x_coords)
                y_min = torch.min(y_coords)
                x_max = torch.max(x_coords)
                y_max = torch.max(y_coords)
                
                w = x_max - x_min + 1
                h = y_max - y_min + 1
                
                # 矩形框的四个顶点坐标（左上角顺时针）- 保持tensor格式
                bbox_coords = torch.stack([
                    torch.stack([x_min, y_min]),         # 左上角
                    torch.stack([x_max, y_min]),         # 右上角
                    torch.stack([x_max, y_max]),         # 右下角
                    torch.stack([x_min, y_max])          # 左下角
                ])
                
                # # 确保坐标在图像范围内
                # x_start = torch.clamp(x_min, 0, W_1-1).long()
                # y_start = torch.clamp(y_min, 0, H_1-1).long()
                # x_end = torch.clamp(x_min + w, 0, W_1).long()
                # y_end = torch.clamp(y_min + h, 0, H_1).long()
                x_start = x_min
                y_start = y_min
                x_end = x_max + 1
                y_end = y_max + 1
                
                # 从原始图像中裁剪矩形区域（保持tensor格式）
                cropped_image = original_image* class_mask_gpu
                cropped_image = cropped_image.permute(1, 2, 0)
                cropped_image = cropped_image[y_start:y_end, x_start:x_end]  # (crop_h, crop_w, C)
                cropped_image = cropped_image.permute(2, 0, 1)
                # 使用torch的插值操作调整分辨率到384x384
                if cropped_image.numel() > 0:
                    # 转换为(C, H, W)格式用于插值
                    cropped_image_chw = cropped_image.unsqueeze(0).float()  # (1, C, crop_h, crop_w)
                    resized_image = F.interpolate(
                        cropped_image_chw, 
                        size=(384, 384), 
                        mode='bilinear', 
                        align_corners=False
                    ).squeeze(0)  # (C, 384, 384)


                # 保存结果 - 全部保持tensor格式
                class_result = {
                    'class_id': class_id, #类别id
                    'mask': class_mask_gpu,  # mask
                    'bbox_coords': bbox_coords,  # 矩形框坐标 左上开始顺时针
                    'cropped_image': cropped_image,  # 裁剪后的图像
                    'resized_image': resized_image,  # 调整分辨率后的图像
                    'bbox_xywh': torch.stack([x_min, y_min, w, h])  # 矩形框坐标(x_min, y_min, w, h)
                }
                batch_results.append(class_result)
            
            results.append(batch_results)
        
        return results
    
    def AFM_A(self, attn_clip, attn_dino ,alpha=0.5, y = 6, mask_open=False):
        sfmx_dino = F.softmax(attn_dino, dim=-2)
        A_dino_to_clip = torch.mul(attn_clip, sfmx_dino) # A_clip<-dino
        # 假设 tensor: [12, 1, 576, 576]
        B, H, N, _ = attn_clip.shape

        # 1. 计算每张矩阵的 95% 分位数
        threshold = attn_clip.reshape(B, H, -1).kthvalue(int(576*576*0.95), dim=-1).values.unsqueeze(-1).unsqueeze(-1)

        # 2. 得到二值 mask（>= 阈值）
        mask = attn_clip >= threshold  # [12, 1, 576, 576]  bool
        A_clip_to_dino = attn_dino * mask #A_dino<-clip
        A = alpha * A_dino_to_clip + (1 - alpha) * A_clip_to_dino
        A = A.permute(1,0,2,3) # [1, 12, 576, 576] B,L,N,N
        A_avg = A.mean(dim=(-2, -1), keepdim=True)
        # 1. 比较：True/False -> 1/0
        over = (A > A_avg).int()          # shape (1,12,576,576)
        # 2. 沿层维度（dim=1）求和，就是“超过均值的个数”
        count = over.sum(dim=1, keepdim=False)  # shape (1,576,576)
        Mask_A = (count >= y).float()
        
        return A*Mask_A if mask_open else A
    def forward(self, batched_inputs):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper`.
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:
                   * "image": Tensor, image in (C, H, W) format.
                   * "instances": per-region ground truth
                   * Other information that's included in the original dicts, such as:
                     "height", "width" (int): the output resolution of the model (may be different
                     from input resolution), used in inference.
        Returns:
            list[dict]:
                each dict has the results for one image. The dict contains the following keys:

                * "sem_seg":
                    A Tensor that represents the
                    per-pixel segmentation prediced by the head.
                    The prediction has shape KxHxW that represents the logits of
                    each class for each pixel.
        """
        images = [x["image"].to(self.device) for x in batched_inputs]
        gts = [x["obj_part_sem_seg"].to(self.device) for x in batched_inputs]
        obj_gts = [x["sem_seg"].to(self.device) for x in batched_inputs]

        clip_images = [(x - self.clip_pixel_mean) / self.clip_pixel_std for x in images]
        clip_images = ImageList.from_tensors(clip_images, self.size_divisibility)
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(images, self.size_divisibility)  # [[3, 384, 384], ...]
        clip_images = F.interpolate(clip_images.tensor, size=self.clip_resolution, mode='bilinear', align_corners=False, )

        #gs
        h, w = clip_images.shape[-2], clip_images.shape[-1]
        clip_features, attn_clip = self.sem_seg_head.predictor.clip_model.encode_image_with_attn(clip_images, dense=True, need_weights=True) #gs

        attn_clip = torch.stack(attn_clip,dim=0)

        attn_clip = attn_clip[...,1:,1:] # torch.Size([12, 4, 577, 577])
        attn_clip_origin = attn_clip.permute(1,0,2,3)
        attn_clip_h, attn_clip_w = attn_clip.shape[-2:]

        # ------
        # DINOv2
        # ------
        features_dino, attn_dino = self.backbone(images, need_weights=True)
        attn_dino = torch.stack(attn_dino,dim=0) # torch.Size([12, 4, 6, 1025, 1025])
        attn_dino = attn_dino[...,1:,1:].mean(dim=2, keepdim=False)
        dino_b,dino_c,dino_h,dino_w = attn_dino.shape

        attn_dino = F.interpolate(attn_dino.view(dino_b*dino_c,1,dino_h,dino_w),
                         size=(attn_clip_h, attn_clip_w), mode='bilinear', align_corners=False
                         ).view(dino_b,dino_c,attn_clip_h, attn_clip_w) # torch.Size([12, 4, 576, 576])
        
        A = self.AFM_A(attn_clip, attn_dino, alpha =1,mask_open=False)
        AFM_ON = True
        
        if self.training:
            num_text_classes = len(self.train_text_classes)
            num_part_classes = len(self.train_part_classes)
            num_obj_classes = len(self.train_obj_in_part_classes)
            # num_part_obj_classes = num_part_classes + num_obj_classes

            targets = torch.stack([gt for gt in gts], dim=0).long().squeeze(1).squeeze(1)
            obj_targets = torch.stack([gt for gt in obj_gts], dim=0).long().squeeze(1).squeeze(1)
            part_targets = self.train_text_to_part_map[targets]
            obj_in_part_targets = self.train_obj_to_obj_in_part_map[obj_targets]

            obj_outputs = self.sem_seg_head(clip_features, features_dino, AFM_A = attn_clip_origin, AFM=AFM_ON ,conf = False) # torch.Size([1, 12, 96, 96])

            obj_outputs = F.interpolate(obj_outputs, size=(targets.shape[-2], targets.shape[-1]), mode="bilinear", align_corners=False)

            """gs"""
            #每个obj里面的所有part对应的obj_part编号
            category_to_part_idx = {} #0: tensor([ 1, 34, 40, 35,  7, 38] 0号obj里面所有的part
            for cat_id, obj_ids in self.train_obj_in_part_to_text.items(): #{0: [0, 1, 2, 3, 4, 5] 0号obj里面所有的specific part
                category_to_part_idx[cat_id] = self.train_text_to_part_map[torch.tensor(obj_ids, device=self.train_text_to_part_map.device)]
            
            bs, num_classes, h, w = obj_outputs.shape

            # 获取原始图像尺寸
            H_1, W_1 = targets.shape[-2], targets.shape[-1]
            #用于后续多个框中区域拼出每个类别logit

            # 对obj_outputs进行后处理，提取不同类别的矩形区域
            obj_regions = self.postprocess_obj_predictions(obj_outputs, batched_inputs, H_1, W_1)
            
            #具体类别的logits
            specific_part_outputs = torch.zeros(bs, num_text_classes , h, w).to(self.device) #obj's part类别的数目
            
            for batch_idx, batch_regions in enumerate(obj_regions):
                
                
                # 逐个处理每个区域图像
                for region in batch_regions:
                    # 获取调整到384x384的图像
                    resized_image = region['resized_image']  # (3, 384, 384) 
                    
                    # (3, 384, 384)
                    region_tensor = resized_image.to(self.device)
                    
                    # 模仿原始CLIP处理方式，单张图像处理
                    clip_region_image = (region_tensor - self.clip_pixel_mean) / self.clip_pixel_std
                    clip_region_image = ImageList.from_tensors([clip_region_image], self.size_divisibility)
                    clip_region_image = F.interpolate(clip_region_image.tensor, size=self.clip_resolution, mode='bilinear', align_corners=False)
                    
                    # 获取CLIP图像的尺寸
                    h_clip, w_clip = clip_region_image.shape[-2], clip_region_image.shape[-1]
                    
                    # 使用CLIP编码单张区域图像
                    region_clip_feature = self.sem_seg_head.predictor.clip_model.encode_image(clip_region_image, dense=True)
                    
                    # 标准化图像 (使用与原始图像相同的标准化)
                    dino_region_image = (region_tensor - self.pixel_mean) / self.pixel_std
                    dino_region_image = ImageList.from_tensors([dino_region_image], self.size_divisibility)
                    
                    # 使用backbone直接获取DINO特征

                    region_dino_features = self.backbone(dino_region_image)
                    #预测出物体中的所有part的编号
                    part_in_my_obj = category_to_part_idx[region['class_id']] #该obj 中所有part的编号 0：[ 1, 34, 40, 35,  7, 38]
    
                    #预测出物体中的所有specific-part的编号
                    my_specific_part = self.train_obj_in_part_to_text[region['class_id']] #该obj 中所有obj's part的编号 0：[0, 1, 2, 3, 4, 5]
                    AEM =1
                    part_outputs = self.sem_seg_head.forward_second(region_clip_feature, region_dino_features, part_in_my_obj,AEM)
                    # 图像裁剪的大小
                    x_region, y_region, w_region, h_region = region["bbox_xywh"]
                    
                    part_outputs = F.interpolate(part_outputs, size=(h_region, w_region), mode="bilinear", align_corners=False)
                    

                    my_obj_mask = region["mask"][y_region : y_region + h_region, x_region : x_region + w_region]

                    part_outputs = torch.sigmoid(part_outputs)
                    my_obj_mask_float = my_obj_mask.float()
                    part_outputs = part_outputs * my_obj_mask_float  # 每个part通道背景区域概率置为0

                    # specific_part_outputs[batch_idx, my_specific_part, y_region : y_region + h_region, x_region : x_region + w_region] = specific_part_outputs[batch_idx, my_specific_part, y_region : y_region + h_region, x_region : x_region + w_region] + part_outputs
                    specific_part_outputs[batch_idx, my_specific_part, y_region : y_region + h_region, x_region : x_region + w_region] = part_outputs
                    #将该类别的logit贴到part总的logit上
                       
            #--------- 在这之上得到了obj(含背景)的logit, obj's part的logit
            
            #bs, num_classes, h, w = obj_outputs.shape
            #计算obj损失 和 part损失

            mask = targets != self.sem_seg_head.ignore_value # mask是gt中前景区域
            part_mask = part_targets != self.ignore_label # part_mask是part_gt中前景区域
            obj_mask = obj_in_part_targets != self.ignore_label # obj_mask是第一阶段obj_gt的前景区域

            obj_outputs = obj_outputs.permute(0,2,3,1)
            specific_part_outputs = specific_part_outputs.permute(0,2,3,1)

            obj_targets = torch.zeros(obj_outputs.shape, device=self.device)
            specific_part_targets = torch.zeros((bs, h, w, num_text_classes), device=self.device)
            

            # part_onehot = F.one_hot(part_targets[part_mask], num_classes=num_part_classes).float()
            obj_onehot = F.one_hot(obj_in_part_targets[obj_mask], num_classes=num_obj_classes).float()
            specific_part_onehot = F.one_hot(targets[mask], num_classes=num_text_classes).float()

            # part_obj_targets[..., :num_part_classes][part_mask] = part_onehot
            obj_targets[...,:num_classes-1][obj_mask] = obj_onehot
            specific_part_targets[..., :num_text_classes][mask] = specific_part_onehot

            obj_weight = torch.ones(num_classes).cuda()
            specific_part_weight = torch.ones(num_text_classes).cuda()

            if self.sem_seg_head.predictor.bg_on:
                obj_targets[..., -1][~obj_mask] = 1

                obj_weight[-1] = 0.05

            obj_loss ,specific_part_loss= 0.0, 0.0

            specific_part_loss = F.binary_cross_entropy(
                specific_part_outputs,
                specific_part_targets,
                weight=specific_part_weight,
            )
            

            obj_loss = F.binary_cross_entropy_with_logits(
                obj_outputs,
                obj_targets,
                weight=obj_weight,
            )

            loss = obj_loss + specific_part_loss

            losses = {"loss_sem_seg" : loss}
            return losses
        else:
            num_text_classes = len(self.test_text_classes)
            num_part_classes = len(self.test_part_classes)
            num_obj_classes = len(self.test_obj_in_part_classes)
            # num_part_obj_classes = num_part_classes + num_obj_classes

            category_to_part_idx = {}
            for cat_id, obj_ids in self.test_obj_in_part_to_text.items():
                category_to_part_idx[cat_id] = self.test_text_to_part_map[torch.tensor(obj_ids, device=self.test_text_to_part_map.device)]

            with torch.no_grad():
                # _, outputs, costs = self.sem_seg_head(clip_features, features_dino)
                # costs = {k: v.cpu() for k, v in costs.items()}
                C_1, H_1, W_1 = batched_inputs[0]['image'].shape
                #用于后续多个框中区域拼出每个类别logit

                obj_outputs = self.sem_seg_head(clip_features, features_dino, AFM_A = attn_clip_origin, AFM=AFM_ON ,conf = False)
           
                obj_outputs = F.interpolate(obj_outputs, size=(H_1, W_1), mode="bilinear", align_corners=False)

                # 对obj_outputs进行后处理，提取不同类别的矩形区域
                obj_regions = self.postprocess_obj_predictions(obj_outputs, batched_inputs, H_1, W_1)
                bs = obj_outputs.shape[0]
                specific_part_outputs = torch.zeros(bs, num_text_classes + 1 , H_1, W_1).to(self.device)
                specific_part_outputs[:,-1,:,:] = 1.0
                for batch_idx, batch_regions in enumerate(obj_regions):
                    
                    # 逐个处理每个区域图像
                    for region in batch_regions:
                        # 获取调整到384x384的图像
                        resized_image = region['resized_image']  # (3, 384, 384) 
                        # (3, 384, 384)
                        region_tensor = resized_image.to(self.device)
                        
                        # 模仿原始CLIP处理方式，单张图像处理
                        clip_region_image = (region_tensor - self.clip_pixel_mean) / self.clip_pixel_std
                        clip_region_image = ImageList.from_tensors([clip_region_image], self.size_divisibility)
                        clip_region_image = F.interpolate(clip_region_image.tensor, size=self.clip_resolution, mode='bilinear', align_corners=False)
                        
                        # 获取CLIP图像的尺寸
                        h_clip, w_clip = clip_region_image.shape[-2], clip_region_image.shape[-1]
                        
                        # 使用CLIP编码单张区域图像
                        region_clip_feature, region_attn_clip = self.sem_seg_head.predictor.clip_model.encode_image_with_attn(clip_region_image, dense=True, need_weights=True)               
                        
                        region_attn_clip = torch.stack(region_attn_clip,dim=0)
                        region_attn_clip = region_attn_clip[...,1:,1:] # torch.Size([12, 4, 577, 577])
                        region_attn_clip = region_attn_clip.permute(1,0,2,3)
                        # 标准化图像 (使用与原始图像相同的标准化)
                        dino_region_image = (region_tensor - self.pixel_mean) / self.pixel_std
                        dino_region_image = ImageList.from_tensors([dino_region_image], self.size_divisibility)
                        
                        # 使用backbone直接获取DINO特征
                        region_dino_features = self.backbone(dino_region_image)

                        
                        #预测出物体中的所有part的编号
                        part_in_my_obj = category_to_part_idx[region['class_id']]
                        #预测出物体中的所有specific-part的编号
                        my_specific_part = self.test_obj_in_part_to_text[region['class_id']]

                        part_outputs = self.sem_seg_head.forward_second(region_clip_feature, region_dino_features, part_in_my_obj,region_attn_clip,AEM=False)
                        # 图像裁剪的大小
                        x_region, y_region, w_region, h_region = region["bbox_xywh"]
                        
                        part_outputs = F.interpolate(part_outputs, size=(h_region, w_region), mode="bilinear", align_corners=False)

                        # specific_part_outputs[batch_idx, my_specific_part, y_region : y_region + h_region, x_region : x_region + w_region] = part_outputs
                        # my_obj_mask = region["mask"]
                        my_obj_mask = region["mask"][y_region : y_region + h_region, x_region : x_region + w_region]
                         
                        part_outputs = torch.sigmoid(part_outputs)

                        part_outputs = part_outputs * my_obj_mask  

                        specific_part_outputs[batch_idx, my_specific_part, y_region : y_region + h_region, x_region : x_region + w_region] = part_outputs
                        mask_3d = region["mask"].unsqueeze(0).expand_as(specific_part_outputs[:, -1, :, :])
                        specific_part_outputs[:,-1,:,:][mask_3d]  = 0.0
                        #将该类别的logit贴到logit上
   

                obj_instances = [x["instances"].to(self.device) for x in batched_inputs]
                obj_class = self.sem_seg_head.predictor.test_obj_classes[obj_instances[0].gt_classes[0].item()]
                obj_part_classes = self.sem_seg_head.predictor.test_ori_text_classes
                select_mask = [i for i, name in enumerate(obj_part_classes) if obj_class not in name] 

                outputs = specific_part_outputs
                if self.sem_seg_head.predictor.bg_on:
                    outputs_all = outputs.sigmoid()
                    outputs = outputs.sigmoid()[:,:-1,:,:].cpu()
                else:
                    outputs_all = outputs.sigmoid()
                    outputs = outputs.sigmoid().cpu()

                outputs[:,select_mask,:,:] = -1.0

                image_size = images.image_sizes[0]
                height = batched_inputs[0].get("height", image_size[0])
                width = batched_inputs[0].get("width", image_size[1])
                output = sem_seg_postprocess(outputs[0].cpu(), image_size, height, width)
                output_all = sem_seg_postprocess(outputs_all[0].cpu(), image_size, height, width)
                processed_results = [
                    {'sem_seg': output.detach().cpu(), 'sem_seg_all': output_all.detach().cpu()}
                ]

            gc.collect()
            torch.cuda.empty_cache()

            return processed_results