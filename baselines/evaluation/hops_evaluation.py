# Copyright (c) Facebook, Inc. and its affiliates.
import itertools
import numpy as np
import os
import PIL.Image as Image
import torch
from torch.nn import functional as F
from detectron2.data import MetadataCatalog
from detectron2.utils.comm import all_gather, is_main_process, synchronize
from detectron2.utils.file_io import PathManager
from detectron2.evaluation import SemSegEvaluator
from detectron2.utils.visualizer import ColorMode

# Copyright (c) Facebook, Inc. and its affiliates.
import itertools
import numpy as np
import os
from collections import OrderedDict, defaultdict
import PIL.Image as Image
import torch

from baselines.utils.visualizer import CustomVisualizer

class HopsEvaluator(SemSegEvaluator):
    """
    Evaluate semantic segmentation metrics.
    """

    def __init__(
        self,
        dataset_name,
        distributed=True,
        output_dir=None,
        *,
        num_classes=None,
        ignore_label=None,
        post_process_func=None,
        visualize=True,
    ):
        super().__init__(
            dataset_name,
            distributed=distributed,
            output_dir=output_dir,
            num_classes=num_classes,
            ignore_label=ignore_label,
        )
        meta = MetadataCatalog.get(dataset_name)
        try:
            self._evaluation_set = meta.evaluation_set
        except AttributeError:
            self._evaluation_set = None
        self.post_process_func = (
            post_process_func
            if post_process_func is not None
            else lambda x, **kwargs: x
        )
        self.visualize = visualize
        
        if self._output_dir:
            PathManager.mkdirs(self._output_dir)

        
        if self.visualize:
            self.vis_path = os.path.join(self._output_dir, "visualization")
            PathManager.mkdirs(self.vis_path)
        
        self.meta = meta
        self.ignore_label = meta.ignore_label
        self.device = "cuda"

        text_classes = meta.stuff_classes
        self.ori_text_classes = text_classes

        self.text_classes = [c.replace("'s", "") for c in text_classes]
        self.obj_classes = meta.obj_classes
        self.part_classes = sorted(list(set([c.split("'s")[1].strip() for c in text_classes])))
        self.obj_in_part_classes = sorted(list(set([c.split("'s")[0].strip() for c in text_classes])))

        self.obj_to_obj_in_part = {}
        self.obj_to_obj_in_part_map = torch.full(
            (self.ignore_label + 1,), self.ignore_label, dtype=torch.long, device=self.device
        )

        self.obj_in_part_to_text = defaultdict(list)
        self.text_to_obj_in_part_map = torch.full(
            (len(self.text_classes),), len(self.text_classes), dtype=torch.long, device=self.device
        )

        self.text_to_part = {}
        for index, class_text in enumerate(text_classes):
            self.text_to_part[index] = self.part_classes.index(class_text.split("'s")[1].strip())

        for index, class_text in enumerate(self.obj_classes):
            if class_text in self.obj_in_part_classes:
                obj_in_part_index = self.obj_in_part_classes.index(class_text)
                self.obj_to_obj_in_part[index] = obj_in_part_index
                self.obj_to_obj_in_part_map[index] = obj_in_part_index
            else:
                self.obj_to_obj_in_part[index] = self.ignore_label

        for index, class_text in enumerate(text_classes):
            obj_class, _ = class_text.split("'s", maxsplit=1)
            obj_in_part_index = self.obj_in_part_classes.index(obj_class)
            self.obj_in_part_to_text[obj_in_part_index].append(index)
            self.text_to_obj_in_part_map[index] = obj_in_part_index

    def reset(self):
        super().reset()
        self._conf_matrix = np.zeros(
            (self._num_classes + 1, self._num_classes + 1), dtype=np.int64
        )
        self._conf_matrix_pred_all = np.zeros(
            (self._num_classes + 1, self._num_classes + 1), dtype=np.int64
        )
        if self._num_classes==116:
            self._conf_matrix_obj = np.zeros(
                (17, 17), dtype=np.int64
            )

    def process(self, inputs, outputs):
        """
        Args:
            inputs: the inputs to a model.
                It is a list of dicts. Each dict corresponds to an image and
                contains keys like "height", "width", "file_name".
            outputs: the outputs of a model. It is either list of semantic segmentation predictions
                (Tensor [H, W]) or list of dicts with key "sem_seg" that contains semantic
                segmentation prediction in the same format.
        """
        for input, output_org in zip(inputs, outputs):            
            sem_seg_output, sem_seg_output_all = output_org["sem_seg"], output_org["sem_seg_all"]

            output = self.post_process_func(sem_seg_output, image=np.array(Image.open(input["file_name"])))
            output_all = self.post_process_func(sem_seg_output_all, image=np.array(Image.open(input["file_name"])))
            
            output = output.argmax(dim=0).to(self._cpu_device)
            output_all = output_all.argmax(dim=0).to(self._cpu_device)
            
            gt_classes = input["obj_part_instances"].gt_classes


            gt_masks = input["obj_part_instances"].gt_masks
            eval_image_size = tuple(output.shape[-2:])
            


            if len(gt_masks) == 0:
                gt = np.zeros_like(pred) + self._ignore_label
                obj_gt = np.zeros_like(pred) + self._ignore_label
            else:
                gt = np.zeros_like(gt_masks[0], dtype=np.float) + self._ignore_label

                for i in range(len(gt_classes)):
                    gt[gt_masks[i] == True] = gt_classes[i]
                # pascal116使用objgt
                if self._num_classes==116:
                    obj_gt = np.zeros_like(gt_masks[0], dtype=np.float) + self._ignore_label       
                    obj_to_obj_dict = {0: 0, 1: 1, 2: 2, 3: 255, 4: 3, 5: 4, 6: 5, 7: 6, 8: 255, 9: 7, 10: 255, 11: 8, 12: 9, 13: 10, 14: 11, 15: 12, 16: 13, 17: 255, 18: 14, 19: 15}
                    obj_gt_classes = input["instances"].gt_classes
                    obj_gt_masks = input["instances"].gt_masks                    
                    for i in range(len(obj_gt_classes)):
                        obj_gt[obj_gt_masks[i] == True] = obj_gt_classes[i]
                    obj_gt = F.interpolate(torch.tensor(obj_gt).unsqueeze(0).unsqueeze(0), size=eval_image_size, mode='nearest').squeeze()
                    obj_gt = obj_gt.int().numpy()
                    for i in range(len(obj_to_obj_dict)):
                        obj_gt[obj_gt==i] =obj_to_obj_dict[i] 
                gt = F.interpolate(torch.tensor(gt).unsqueeze(0).unsqueeze(0), size=eval_image_size, mode='nearest').squeeze()
                gt = gt.int().numpy()
                
            

            output[gt == self._ignore_label] = self.meta.ignore_label

            # pred = np.array(output, dtype=np.int)
            pred = np.array(output_all, dtype=np.int)
            pred_all = np.array(output_all, dtype=np.int)
                
            pred[pred == self._ignore_label] = self._num_classes
            pred_all[pred_all == self._ignore_label] = self._num_classes
            # pred_all[(gt == self._ignore_label)] = self._num_classes
            pred[(gt == self._ignore_label)] = self._num_classes

            gt[gt == self._ignore_label] = self._num_classes 

            if self._num_classes==116:
                pred_obj = pred_all.copy()
                s_part_to_obj=[ 0,  0,  0,  0,  0,  0,  1,  1,  1,  1,  1,  2,  2,  2,  2,  2,  2,  2,
                    2,  2,  3,  3,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  5,  5,  5,  5,
                    5,  5,  5,  5,  5,  5,  6,  6,  6,  6,  6,  6,  6,  6,  6,  7,  7,  7,
                    7,  7,  7,  7,  7,  7,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  9,  9,
                    9,  9,  9,  9,  9,  9,  9, 10, 10, 10, 10, 11, 11, 11, 11, 11, 11, 11,
                    11, 11, 11, 11, 11, 11, 11, 12, 12, 13, 13, 13, 13, 13, 13, 13, 13, 13,
                    14, 14, 14, 14, 14, 14, 14, 15, 16]
                for i in range(self._num_classes+1):
                    pred_obj[pred_all==i] = s_part_to_obj[i]
                obj_gt[obj_gt == self._ignore_label]=16
                self._conf_matrix_obj += np.bincount(
                    (17) * pred_obj.reshape(-1) + obj_gt.reshape(-1),
                    minlength=self._conf_matrix_obj.size,
                ).reshape(self._conf_matrix_obj.shape)

            self._conf_matrix += np.bincount(
                (self._num_classes + 1) * pred.reshape(-1) + gt.reshape(-1),
                minlength=self._conf_matrix.size,
            ).reshape(self._conf_matrix.shape)

            self._conf_matrix_pred_all += np.bincount(
                (self._num_classes + 1) * pred_all.reshape(-1) + gt.reshape(-1),
                minlength=self._conf_matrix_pred_all.size,
            ).reshape(self._conf_matrix_pred_all.shape)

            self._predictions.extend(self.encode_json_sem_seg(pred, input["file_name"]))
            
            if self.visualize:
                ext = os.path.splitext(input["file_name"])[1]
                input_img_tensor = F.interpolate(input["image"].unsqueeze(0), size=eval_image_size, mode='bilinear').squeeze()
                input_img_npy = input_img_tensor.permute(1, 2, 0).int().numpy()

                visualizer_pred = CustomVisualizer(input_img_npy, self.meta, instance_mode=ColorMode.SEGMENTATION)
                visualizer_pred_all = CustomVisualizer(input_img_npy, self.meta, instance_mode=ColorMode.SEGMENTATION)
                visualizer_gt = CustomVisualizer(input_img_npy, self.meta, instance_mode=ColorMode.SEGMENTATION)
                
                vis_pred = visualizer_pred.draw_sem_seg(pred)
                vis_pred.save(os.path.join(self.vis_path, os.path.basename(input["file_name"])))
                
                vis_pred_all = visualizer_pred_all.draw_sem_seg(np.array(output_all, dtype=np.int))
                vis_pred_all.save(os.path.join(self.vis_path, os.path.basename(input["file_name"]).replace(ext, "_all.jpg")))

                vis_gt = visualizer_gt.draw_sem_seg(gt)
                vis_gt.save(os.path.join(self.vis_path, os.path.basename(input["file_name"]).replace(ext, "_gt.jpg")))

    def cal_osi(self, conf_matrix, num_classes, seen): 
        
        if num_classes==147:
            s_part_to_obj=[21, 21, 21, 21,  7,  7,  7,  7,  4,  4,  4,  4,  4, 11, 11, 11, 25, 25,
                25, 29, 29, 29, 29, 29, 15, 15, 15, 15, 17, 17, 17, 17, 17,  5,  5,  5,
                27, 27, 27, 27, 39, 39, 18, 18, 18, 18, 14, 14, 14, 14, 19, 19,  8,  8,
                34, 34, 34, 34, 35, 35, 35, 35, 24, 24, 24, 24, 22, 22, 22, 10, 10, 10,
                10, 10, 16, 16, 16, 16, 16, 37, 37,  0,  0,  0,  0, 13, 13, 13, 13, 33,
                33, 33, 33, 38, 38, 20, 20, 20, 20, 30, 30,  9,  9,  9,  9, 36, 36, 36,
                36, 36, 32, 32,  2,  2,  2,  2,  1,  1, 12, 12, 12, 12, 28, 28, 28, 28,
                31, 31, 31, 23, 23, 23, 23, 26, 26, 26, 26,  3,  3,  3,  3,  3,  6,  6,
                6,  6,  6]
        elif num_classes==116:
            s_part_to_obj=[ 0,  0,  0,  0,  0,  0,  1,  1,  1,  1,  1,  2,  2,  2,  2,  2,  2,  2,
                2,  2,  3,  3,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  5,  5,  5,  5,
                5,  5,  5,  5,  5,  5,  6,  6,  6,  6,  6,  6,  6,  6,  6,  7,  7,  7,
                7,  7,  7,  7,  7,  7,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  9,  9,
                9,  9,  9,  9,  9,  9,  9, 10, 10, 10, 10, 11, 11, 11, 11, 11, 11, 11,
                11, 11, 11, 11, 11, 11, 11, 12, 12, 13, 13, 13, 13, 13, 13, 13, 13, 13,
                14, 14, 14, 14, 14, 14, 14, 15]
        else:
            s_part_to_obj=[27, 27, 27, 27, 27, 27, 27, 27, 27, 17, 17, 17, 17, 11, 11, 38, 38, 38,
                6,  6,  6,  6,  6,  6,  6, 32, 32, 32, 32, 32, 21, 21, 21, 21, 21, 21,
                21, 21, 21, 30, 30, 30, 30, 30,  8,  8,  8,  8,  8,  8,  8,  8,  8, 10,
                10, 10, 10, 10,  9,  9,  9,  9,  9,  9,  9,  9,  2,  2,  2,  2, 36, 36,
                36, 36, 36, 36,  1,  1,  1,  1,  1,  1,  1,  1, 25, 25, 25, 31, 31, 31,
                31, 35, 35, 35, 35, 18, 18, 18, 12, 12, 34, 34, 33, 33, 33, 33, 33, 33,
                33, 13, 13, 13, 13, 15, 15, 15, 15, 15, 15, 42, 42, 42, 42, 42, 42,  7,
                7,  7,  7,  7,  7,  7,  7,  7,  7,  5,  5,  5,  5,  5,  5,  5,  5,  5,
                26, 26, 26, 26, 14, 14, 14, 14, 14, 14, 23, 23, 23, 23, 23, 23, 29, 29,
                29, 29, 20, 20, 20, 20, 20, 16, 16, 16,  4,  4,  4,  4, 37, 37, 37, 37,
                37, 37, 19, 19, 19, 19, 28, 28, 28, 41, 41, 41, 41, 41, 41, 41, 41, 41,
                41,  0,  0,  0,  0,  0,  0,  0, 40, 40, 40, 40, 40, 40, 40, 40, 24, 24,
                24, 24, 43, 43, 43, 43,  3,  3,  3,  3, 39, 39, 22, 22, 22, 22, 22, 22]
        obj_classes = np.max(s_part_to_obj)+1
        part2obj = np.array(s_part_to_obj + [obj_classes], dtype=np.int32)  # 147→obj, 第148维(背景)→40

        # ---------- 2. 初始化 41×41 obj 混淆矩阵 ---------- 备注都以partimagenet147类为基准
        obj_cm = np.zeros((obj_classes+1, obj_classes+1), dtype=conf_matrix.dtype)

        # ---------- 3. 聚合：先按行（真实），再按列（预测） ----------
        # 行索引 = 真实 part-id，列索引 = 预测 part-id
        rows, cols = np.indices(conf_matrix.shape)  # 148×148 的坐标网格
        # 把每个计数累加到对应的 obj 格子
        np.add.at(obj_cm, (part2obj[rows], part2obj[cols]), conf_matrix)

        tp = np.diag(obj_cm).astype(np.float64)
        gt_pix = np.sum(obj_cm, axis=0).astype(np.float64)   # |G|
        pred_pix = np.sum(obj_cm, axis=1).astype(np.float64) # |P|
        fp = pred_pix - tp
        osi_valid = gt_pix > 0
        osi = np.full(obj_classes+1, np.nan, dtype=np.float64)
        osi[osi_valid] = fp[osi_valid] / pred_pix[osi_valid]

        osi = osi[:obj_classes]
        mask_base = np.zeros(obj_classes)
        for idx in seen:
            mask_base[s_part_to_obj[idx]] = 1
        osi_base = np.nanmean(osi[mask_base==1])
        osi_unbase = np.nanmean(osi[mask_base==0])
        mOSI= np.nanmean(osi)
        return osi_base*100, osi_unbase*100,mOSI*100
        
    def calculate_metrics(self, conf_matrix, num_classes, class_names, include_last_class=False):
        acc = np.full(num_classes, np.nan, dtype=np.float64)
        iou = np.full(num_classes, np.nan, dtype=np.float64)
        recall = np.full(num_classes, np.nan, dtype=np.float64)

        if include_last_class:
            tp = conf_matrix.diagonal().astype(np.float64)
            pos_gt = np.sum(conf_matrix, axis=0).astype(np.float64)
            pos_pred = np.sum(conf_matrix, axis=1).astype(np.float64)
        else:
            tp = conf_matrix.diagonal()[:-1].astype(np.float64)
            pos_gt = np.sum(conf_matrix[:-1, :-1], axis=0).astype(np.float64)
            pos_pred = np.sum(conf_matrix[:-1, :-1], axis=1).astype(np.float64)
        # 1. 计算 USI
        usi = np.full(num_classes, np.nan, dtype=np.float64)
        fn = pos_gt - tp                 # 漏检数
        usi_valid = pos_gt > 0
        usi[usi_valid] = fn[usi_valid] / pos_gt[usi_valid]
        class_weights = pos_gt / np.sum(pos_gt)

        recall_valid = pos_gt > 0
        acc[recall_valid] = tp[recall_valid] / pos_gt[recall_valid]

        union = pos_gt + pos_pred - tp
        iou_valid = (pos_gt + pos_pred) > 0

        iou[iou_valid] = tp[iou_valid] / union[iou_valid]
        recall[recall_valid] = tp[recall_valid] / pos_gt[recall_valid]

        macc = np.nanmean(acc)
        miou = np.nanmean(iou)
        fiou = np.nansum(iou * class_weights)
        pacc = np.nansum(tp) / np.nansum(pos_gt)
        mRecall = np.nanmean(recall)
        mUSI = np.nanmean(usi)
        res = {
            "mIoU": 100 * miou,
            "fwIoU": 100 * fiou,
            "mACC": 100 * macc,
            "pACC": 100 * pacc,
            "mRecall": 100 * mRecall,
            "mUSI": 100 * mUSI,
        }
        
        for i, name in enumerate(class_names):
            res[f"IoU-{name}"] = 100 * iou[i]
            res[f"ACC-{name}"] = 100 * acc[i]
            res[f"Recall-{name}"] = 100 * recall[i]
            res[f"USI-{name}"] = 100 * usi[i]
        return res

    def evaluate(self):
        if self._distributed:
            synchronize()
            conf_matrix_list = all_gather(self._conf_matrix)
            conf_matrix_pred_all_list = all_gather(self._conf_matrix_pred_all)
            self._predictions = all_gather(self._predictions)
            self._predictions = list(itertools.chain(*self._predictions))
            if not is_main_process():
                return

            self._conf_matrix = np.zeros_like(self._conf_matrix)
            for conf_matrix in conf_matrix_list:
                self._conf_matrix += conf_matrix

            self._conf_matrix_pred_all = np.zeros_like(self._conf_matrix_pred_all)
            for conf_matrix_pred_all in conf_matrix_pred_all_list:
                self._conf_matrix_pred_all += conf_matrix_pred_all

        # res = self.calculate_metrics(self._conf_matrix, self._num_classes, self._class_names, include_last_class=False)
        res = self.calculate_metrics(self._conf_matrix, self._num_classes + 1, self._class_names, include_last_class=True)
        res_pred_all = self.calculate_metrics(self._conf_matrix_pred_all, self._num_classes + 1, self._class_names, include_last_class=True)

        if self._evaluation_set is not None:
            for set_name, set_inds in self._evaluation_set.items():
                set_inds = np.array(set_inds, dtype=int)
                mask = np.zeros(len(self._class_names), dtype=bool)
                mask[set_inds] = True

                subset_iou_valid = mask & np.array([res[f"IoU-{self._class_names[i]}"] > 0 for i in range(len(self._class_names))])

                if np.any(subset_iou_valid):
                    miou = np.nanmean([res[f"IoU-{self._class_names[i]}"] for i in set_inds if subset_iou_valid[i]])
                    mrecall = np.nanmean([res[f"Recall-{self._class_names[i]}"] for i in set_inds if subset_iou_valid[i]])
                    pacc = np.nansum([res[f"ACC-{self._class_names[i]}"] for i in set_inds if subset_iou_valid[i]]) / np.sum(subset_iou_valid)

                    res[f"mIoU-{set_name}"] = miou
                    res[f"mRecall-{set_name}"] = mrecall
                    res[f"pACC-{set_name}"] = pacc

                # Calculate for inverse mask (unbase classes)
                inv_mask = ~mask
                subset_iou_valid_inv = inv_mask & np.array([res[f"IoU-{self._class_names[i]}"] > 0 for i in range(len(self._class_names))])

                if np.any(subset_iou_valid_inv):
                    miou_inv = np.nanmean([res[f"IoU-{self._class_names[i]}"] for i in range(len(self._class_names)) if subset_iou_valid_inv[i]])
                    mrecall_inv = np.nanmean([res[f"Recall-{self._class_names[i]}"] for i in range(len(self._class_names)) if subset_iou_valid_inv[i]])
                    pacc_inv = np.nansum([res[f"ACC-{self._class_names[i]}"] for i in range(len(self._class_names)) if subset_iou_valid_inv[i]]) / np.sum(subset_iou_valid_inv)

                    res[f"mIoU-un{set_name}"] = miou_inv
                    res[f"mRecall-un{set_name}"] = mrecall_inv
                    res[f"pACC-un{set_name}"] = pacc_inv

                # Repeat the same for res_pred_all
                subset_iou_valid_pred_all = mask & np.array([res_pred_all[f"IoU-{self._class_names[i]}"] > 0 for i in range(len(self._class_names))])

                if np.any(subset_iou_valid_pred_all):
                    miou_pred_all = np.nanmean([res_pred_all[f"IoU-{self._class_names[i]}"] for i in set_inds if subset_iou_valid_pred_all[i]])
                    mrecall_pred_all = np.nanmean([res_pred_all[f"Recall-{self._class_names[i]}"] for i in set_inds if subset_iou_valid_pred_all[i]])
                    pacc_pred_all = np.nansum([res_pred_all[f"ACC-{self._class_names[i]}"] for i in set_inds if subset_iou_valid_pred_all[i]]) / np.sum(subset_iou_valid_pred_all)
                    musi_pred_all = np.nanmean([res_pred_all[f"USI-{self._class_names[i]}"] for i in set_inds if subset_iou_valid_pred_all[i]])

                    res_pred_all[f"mIoU-{set_name}"] = miou_pred_all
                    res_pred_all[f"mRecall-{set_name}"] = mrecall_pred_all
                    res_pred_all[f"pACC-{set_name}"] = pacc_pred_all
                    res_pred_all[f"mUSI-{set_name}"] = musi_pred_all

                subset_iou_valid_pred_all_inv = inv_mask & np.array([res_pred_all[f"IoU-{self._class_names[i]}"] > 0 for i in range(len(self._class_names))])

                if np.any(subset_iou_valid_pred_all_inv):
                    miou_pred_all_inv = np.nanmean([res_pred_all[f"IoU-{self._class_names[i]}"] for i in range(len(self._class_names)) if subset_iou_valid_pred_all_inv[i]])
                    mrecall_pred_all_inv = np.nanmean([res_pred_all[f"Recall-{self._class_names[i]}"] for i in range(len(self._class_names)) if subset_iou_valid_pred_all_inv[i]])
                    pacc_pred_all_inv = np.nansum([res_pred_all[f"ACC-{self._class_names[i]}"] for i in range(len(self._class_names)) if subset_iou_valid_pred_all_inv[i]]) / np.sum(subset_iou_valid_pred_all_inv)
                    musi_pred_all_inv = np.nanmean([res_pred_all[f"USI-{self._class_names[i]}"] for i in range(len(self._class_names)) if subset_iou_valid_pred_all_inv[i]])

                    res_pred_all[f"mIoU-un{set_name}"] = miou_pred_all_inv
                    res_pred_all[f"mRecall-un{set_name}"] = mrecall_pred_all_inv
                    res_pred_all[f"pACC-un{set_name}"] = pacc_pred_all_inv
                    res_pred_all[f"mUSI-un{set_name}"] = musi_pred_all_inv

        if 'mIoU-base' in res and 'mIoU-unbase' in res:
            res['h-IoU'] = 2 * (res['mIoU-base'] * res['mIoU-unbase']) / (res['mIoU-base'] + res['mIoU-unbase']) if (res['mIoU-base'] + res['mIoU-unbase']) != 0 else np.nan
        if 'mRecall-base' in res and 'mRecall-unbase' in res:
            res['h-Recall'] = 2 * (res['mRecall-base'] * res['mRecall-unbase']) / (res['mRecall-base'] + res['mRecall-unbase']) if (res['mRecall-base'] + res['mRecall-unbase']) != 0 else np.nan

        if 'mIoU-base' in res_pred_all and 'mIoU-unbase' in res_pred_all:
            res_pred_all['h-IoU'] = 2 * (res_pred_all['mIoU-base'] * res_pred_all['mIoU-unbase']) / (res_pred_all['mIoU-base'] + res_pred_all['mIoU-unbase']) if (res_pred_all['mIoU-base'] + res_pred_all['mIoU-unbase']) != 0 else np.nan
        if 'mRecall-base' in res_pred_all and 'mRecall-unbase' in res_pred_all:
            res_pred_all['h-Recall'] = 2 * (res_pred_all['mRecall-base'] * res_pred_all['mRecall-unbase']) / (res_pred_all['mRecall-base'] + res_pred_all['mRecall-unbase']) if (res_pred_all['mRecall-base'] + res_pred_all['mRecall-unbase']) != 0 else np.nan
        if 'mUSI-base' in res_pred_all and 'mUSI-unbase' in res_pred_all:
            res_pred_all['h-USI'] = 2 * (res_pred_all['mUSI-base'] * res_pred_all['mUSI-unbase']) / (res_pred_all['mUSI-base'] + res_pred_all['mUSI-unbase']) if (res_pred_all['mUSI-base'] + res_pred_all['mUSI-unbase']) != 0 else np.nan
        
        
        #屎一样优美的代码
        osi_base, osi_unbase,mOSI = self.cal_osi(self._conf_matrix_pred_all, self._num_classes, self._evaluation_set['base'])
 
        if self._num_classes == 116:
            obj_cm = self._conf_matrix_obj
            obj_classes = 16
            tp = np.diag(obj_cm).astype(np.float64)
            gt_pix = np.sum(obj_cm, axis=0).astype(np.float64)   # |G|
            pred_pix = np.sum(obj_cm, axis=1).astype(np.float64) # |P|
            fp = pred_pix - tp
            osi_valid = gt_pix > 0
            osi = np.full(obj_classes+1, np.nan, dtype=np.float64)
            # osi[osi_valid] = fp[osi_valid] / gt_pix[osi_valid] #0 ~ 无穷
            osi[osi_valid] = fp[osi_valid] / pred_pix[osi_valid] #0 ~ 1
            osi = osi[:obj_classes]

            seen = np.array(self._evaluation_set['base'])
            mask_base = np.zeros(obj_classes)
            s_part_to_obj=[ 0,  0,  0,  0,  0,  0,  1,  1,  1,  1,  1,  2,  2,  2,  2,  2,  2,  2,
                2,  2,  3,  3,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  5,  5,  5,  5,
                5,  5,  5,  5,  5,  5,  6,  6,  6,  6,  6,  6,  6,  6,  6,  7,  7,  7,
                7,  7,  7,  7,  7,  7,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  9,  9,
                9,  9,  9,  9,  9,  9,  9, 10, 10, 10, 10, 11, 11, 11, 11, 11, 11, 11,
                11, 11, 11, 11, 11, 11, 11, 12, 12, 13, 13, 13, 13, 13, 13, 13, 13, 13,
                14, 14, 14, 14, 14, 14, 14, 15]
            for idx in seen:
                mask_base[s_part_to_obj[idx]] = 1
            osi_base = np.nanmean(osi[mask_base==1])*100
            osi_unbase = np.nanmean(osi[mask_base==0])*100
            mOSI=np.nanmean(osi)*100
        res_pred_all["osi-base"] = osi_base
        res_pred_all["osi-unbase"] = osi_unbase
        res_pred_all["h-osi"] = 2*osi_base*osi_unbase/(osi_base+osi_unbase) if (osi_base + osi_unbase) != 0 else np.nan
        res_pred_all["mOSI"] = mOSI
        if self._output_dir:
            file_path = os.path.join(self._output_dir, "sem_seg_evaluation.pth")
            file_path_pred_all = os.path.join(self._output_dir, "sem_seg_evaluation_all.pth")

            with PathManager.open(file_path, "wb") as f:
                torch.save(res, f)

            with PathManager.open(file_path_pred_all, "wb") as f:
                torch.save(res_pred_all, f)

        # results = OrderedDict({"oracle_obj": res, "pred_all": res_pred_all})
        results = OrderedDict({"pred_with_gt": res, "pred_without_gt": res_pred_all})
        self._logger.info(results)
        return results
