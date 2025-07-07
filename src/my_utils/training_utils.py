import os
import random
import argparse
import json
import torch
from PIL import Image
from torchvision import transforms
import torchvision.transforms.functional as F
from torchvision.transforms import InterpolationMode
from glob import glob

import cv2
import numpy as np

def random_crop_same_3(gt, lr, lr_gray, crop_size):
    """
    使用同一个随机区域，对 gt, lr, lr_gray 三张图进行裁剪
    """
    # 从 gt 中获取随机裁剪参数（也可以从 lr 获取，如果它们大小相同）
    i, j, h, w = transforms.RandomCrop.get_params(gt, output_size=crop_size)
    
    # 对三张图应用同一个裁剪参数
    gt_cropped = F.crop(gt, i, j, h, w)
    lr_cropped = F.crop(lr, i, j, h, w)
    lr_gray_cropped = F.crop(lr_gray, i, j, h, w)

    return gt_cropped, lr_cropped, lr_gray_cropped

def batch_calculate_optical_flow(frames1, frames2, normalize=True):
    """
    批量计算光流图 (支持GPU Tensor输入)
    参数:
        frames1: 前一帧 [B,C,H,W] 范围[0,1]
        frames2: 后一帧 [B,C,H,W]
        normalize: 是否对光流进行归一化
    返回:
        flows: 光流图 [B,2,H,W] 的Tensor
    """
    assert frames1.dim() == 4 and frames2.dim() == 4, "输入必须是[B,C,H,W]格式"
    
    b, c, h, w = frames1.shape
    device = frames1.device
    flows = torch.zeros(b, 2, h, w, device='cpu')  # 先用CPU处理
    
    # 批量处理每对帧
    for i in range(b):
        # 转换为numpy (H,W,3)
        frame1 = frames1[i].permute(1, 2, 0).detach().cpu().numpy() * 255
        frame2 = frames2[i].permute(1, 2, 0).detach().cpu().numpy() * 255
        
        # 计算灰度图
        frame1_gray = cv2.cvtColor(frame1.astype(np.uint8), cv2.COLOR_RGB2GRAY)
        frame2_gray = cv2.cvtColor(frame2.astype(np.uint8), cv2.COLOR_RGB2GRAY)
        
        # Farneback光流
        flow = cv2.calcOpticalFlowFarneback(
            frame1_gray, frame2_gray, None,
            pyr_scale=0.5, levels=3, winsize=15,
            iterations=3, poly_n=5, poly_sigma=1.2, flags=0
        )  # (H,W,2)
        
        if normalize:
            flow[..., 0] /= w  # u分量归一化
            flow[..., 1] /= h  # v分量归一化
        
        flows[i] = torch.from_numpy(flow).permute(2, 0, 1).float()
    
    return flows.to(device)

### rotate and flip
class Augment_RGB_torch:
    def __init__(self):
        pass
    def transform0(self, torch_tensor):
        return torch_tensor
    def transform1(self, torch_tensor):
        torch_tensor = torch.rot90(torch_tensor, k=1, dims=[-1,-2])
        return torch_tensor
    def transform2(self, torch_tensor):
        torch_tensor = torch.rot90(torch_tensor, k=2, dims=[-1,-2])
        return torch_tensor
    def transform3(self, torch_tensor):
        torch_tensor = torch.rot90(torch_tensor, k=3, dims=[-1,-2])
        return torch_tensor
    def transform4(self, torch_tensor):
        torch_tensor = torch_tensor.flip(-2)
        return torch_tensor
    def transform5(self, torch_tensor):
        torch_tensor = (torch.rot90(torch_tensor, k=1, dims=[-1,-2])).flip(-2)
        return torch_tensor
    def transform6(self, torch_tensor):
        torch_tensor = (torch.rot90(torch_tensor, k=2, dims=[-1,-2])).flip(-2)
        return torch_tensor
    def transform7(self, torch_tensor):
        torch_tensor = (torch.rot90(torch_tensor, k=3, dims=[-1,-2])).flip(-2)
        return torch_tensor


augment = Augment_RGB_torch()
transforms_aug = [method for method in dir(augment) if callable(getattr(augment, method)) if not method.startswith('_')]


def parse_args(input_args=None):

    parser = argparse.ArgumentParser()
   
    # args for the loss function
    parser.add_argument("--lambda_lpips", default=2, type=float)
    parser.add_argument("--lambda_l2", default=1.0, type=float)
    parser.add_argument("--lambda_consistency", default=10.0, type=float)
    parser.add_argument("--lambda_highfreq", default=0.1, type=float)
    parser.add_argument("--lambda_moe", default=2, type=float)
    parser.add_argument("--pretrained_from_checkpoint", action="store_true", )
    parser.add_argument('--pretrained_path', type=str, default=None, help='path to a model state dict to be used')

    # dataset options
    parser.add_argument("--lsdir_txt_path", default=None, type=str)
    parser.add_argument("--pexel_txt_path", default=None, type=str)
    parser.add_argument("--dataset_txt_paths", default=None, type=str)
    parser.add_argument("--null_text_ratio", default=0., type=float)
    parser.add_argument("--tracker_project_name", type=str, default="osediff")

    # details about the model architecture
    parser.add_argument("--pretrained_model_name_or_path", default=None, type=str)
    parser.add_argument("--revision", type=str, default=None,)
    parser.add_argument("--variant", type=str, default=None,)
    parser.add_argument("--tokenizer_name", type=str, default=None)

    ## pretrained_path
    parser.add_argument("--osediff_pretrained_path", default=None, type=str)

    # tile setting
    parser.add_argument("--vae_decoder_tiled_size", type=int, default=224)
    parser.add_argument("--vae_encoder_tiled_size", type=int, default=1024)
    parser.add_argument("--latent_tiled_size", type=int, default=96)
    parser.add_argument("--latent_tiled_overlap", type=int, default=32)

    # training details
    parser.add_argument("--output_dir", default='experience/oup')
    parser.add_argument("--seed", type=int, default=123, help="A seed for reproducible training.")
    parser.add_argument("--resolution", type=int, default=512,)
    parser.add_argument("--frames", type=int, default=14, )
    parser.add_argument("--train_batch_size", type=int, default=2, help="Batch size (per device) for the training dataloader.")
    parser.add_argument("--num_training_epochs", type=int, default=10000)
    parser.add_argument("--max_train_steps", type=int, default=100000,)
    parser.add_argument("--checkpointing_steps", type=int, default=500,)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4, help="Number of updates steps to accumulate before performing a backward/update pass.",)
    parser.add_argument("--gradient_checkpointing", action="store_true",)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--lr_scheduler", type=str, default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument("--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler.")
    parser.add_argument("--lr_num_cycles", type=int, default=1,
        help="Number of hard resets of the lr in cosine_with_restarts scheduler.",
    )
    parser.add_argument("--lr_power", type=float, default=1.0, help="Power factor of the polynomial scheduler.")

    parser.add_argument("--dataloader_num_workers", type=int, default=0,)
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--allow_tf32", action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument("--report_to", type=str, default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument("--mixed_precision", type=str, default="fp16", choices=["no", "fp16", "bf16"],)
    parser.add_argument("--enable_xformers_memory_efficient_attention", action="store_true", help="Whether or not to use xformers.")
    parser.add_argument("--set_grads_to_none", action="store_true",)

    parser.add_argument("--logging_dir", type=str, default="logs")
    parser.add_argument("--use_online_deg", action="store_true",)
    parser.add_argument("--need_upscale", action="store_true", )
    parser.add_argument("--deg_file_path", default="params_pasd.yml", type=str)
    parser.add_argument("--align_method", type=str, choices=['wavelet', 'adain', 'nofix'], default='adain')

    # args for the vsd training
    parser.add_argument("--pretrained_model_name_or_path_vsd", default='/home/notebook/data/group/LowLevelLLM/models/diffusion_models/stable-diffusion-2-1-base', type=str)
    parser.add_argument("--snr_gamma_vsd", default=None)
    parser.add_argument("--lambda_vsd", default=1.0, type=float)
    parser.add_argument("--lambda_vsd_lora", default=1.0, type=float)
    parser.add_argument("--min_dm_step_ratio", default=0.02, type=float)
    parser.add_argument("--max_dm_step_ratio", default=0.5, type=float)
    parser.add_argument("--neg_prompt_vsd", default="", type=str)
    parser.add_argument("--pos_prompt_vsd", default="", type=str)
    parser.add_argument("--cfg_vsd", default=7.5, type=float)
    parser.add_argument("--change_max_ratio_iter", default=100000, type=int)
    parser.add_argument("--change_max_dm_step_ratio", default=0.50, type=float)

    # # unet lora setting
    parser.add_argument("--use_unet_encode_lora", action="store_true",)
    parser.add_argument("--use_unet_decode_lora", action="store_true",)
    parser.add_argument("--use_unet_middle_lora", action="store_true",)
    parser.add_argument("--lora_rank_unet", default=4, type=int)
    parser.add_argument("--lora_rank_vae", default=4, type=int)

    parser.add_argument("--lora_rank_unet_quality", default=4, type=int)
    parser.add_argument("--lora_rank_unet_consistency", default=4, type=int)

    ## resume ckpt
    parser.add_argument("--resume_ckpt", default=None, type=str)
    parser.add_argument("--resume_from_lora2", default=None, type=int)
    parser.add_argument("--quality_iter", default=10, type=int)
    parser.add_argument('--quality_iter_1_final', type=int, help='完成第一次quality训练的迭代次数')
    parser.add_argument('--quality_iter_2', type=int, help='第二次consistency训练的迭代次数')

    parser.add_argument("--cfg_csd", default=7.5, type=float)
    parser.add_argument("--load_cfr", action="store_true", )
    parser.add_argument("--pretrained_model_path_csd", default='pretrained_model_path_csd', type=str)

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    return args


def build_transform(image_prep):
    """
    Constructs a transformation pipeline based on the specified image preparation method.

    Parameters:
    - image_prep (str): A string describing the desired image preparation

    Returns:
    - torchvision.transforms.Compose: A composable sequence of transformations to be applied to images.
    """
    if image_prep == "resized_crop_512":
        T = transforms.Compose([
            transforms.Resize(512, interpolation=transforms.InterpolationMode.LANCZOS),
            transforms.CenterCrop(512),
        ])
    elif image_prep == "resize_286_randomcrop_256x256_hflip":
        T = transforms.Compose([
            transforms.Resize((286, 286), interpolation=Image.LANCZOS),
            transforms.RandomCrop((256, 256)),
            transforms.RandomHorizontalFlip(),
        ])
    elif image_prep in ["resize_256", "resize_256x256"]:
        T = transforms.Compose([
            transforms.Resize((256, 256), interpolation=Image.LANCZOS)
        ])
    elif image_prep in ["resize_512", "resize_512x512"]:
        T = transforms.Compose([
            transforms.Resize((512, 512), interpolation=Image.LANCZOS)
        ])
    elif image_prep == "no_resize":
        T = transforms.Lambda(lambda x: x)
    return T

            

import sys
import numpy as np
from src.datasets.realesrgan import RealESRGAN_degradation
class PairedSROnlineTxtDataset(torch.utils.data.Dataset):
    def __init__(self, args=None):
        super().__init__()

        self.args = args
        self.degradation = RealESRGAN_degradation(args.deg_file_path, device='cpu')
        self.crop_preproc = transforms.Compose([
            transforms.RandomCrop((args.resolution, args.resolution)),
            transforms.RandomHorizontalFlip(),
        ])

        with open(args.dataset_txt_paths, 'r') as f:
            self.gt_list = [line.strip() for line in f.readlines()]

    def __len__(self):
        return len(self.gt_list)

    def __getitem__(self, idx):

        gt_img = Image.open(self.gt_list[idx]).convert('RGB')
        gt_img = self.crop_preproc(gt_img)

        output_t, img_t = self.degradation.degrade_process(np.asarray(gt_img)/255., resize_bak=True)
        output_t, img_t = output_t.squeeze(0), img_t.squeeze(0)

        # input images scaled to -1,1
        img_t = F.normalize(img_t, mean=[0.5], std=[0.5])
        # output images scaled to -1,1
        output_t = F.normalize(output_t, mean=[0.5], std=[0.5])

        example = {}
        example["neg_prompt"] = self.args.neg_prompt_vsd
        example["null_prompt"] = ""
        example["output_pixel_values"] = output_t
        example["conditioning_pixel_values"] = img_t

        return example

from torchvision.transforms import transforms

class PairedSROnlineTxtDataset_multiframe(torch.utils.data.Dataset):
    def __init__(self, args=None):
        super().__init__()

        self.args = args
        self.degradation = RealESRGAN_degradation(args.deg_file_path, device='cpu')
        # self.crop_preproc = transforms.Compose([
        #     transforms.RandomCrop((args.resolution, args.resolution)),
        #     transforms.RandomHorizontalFlip(),
        # ])
        self.crop_preproc = transforms.Compose([
            transforms.CenterCrop((args.resolution, args.resolution))
        ])
        if args.need_upscale:
            self.crop_preproc_lr = transforms.Compose([
                transforms.CenterCrop((args.resolution // 4, args.resolution // 4))
            ])
        else:
            self.crop_preproc_lr = transforms.Compose([
                transforms.CenterCrop((args.resolution, args.resolution))
            ])
        self.transform = transforms.Compose([transforms.ToTensor()])
        self.frame_number = args.frames
        self.need_upscale = args.need_upscale

        with open(args.dataset_txt_paths, 'r') as f:
            self.gt_list = [line.strip() for line in f.readlines()]

    def __len__(self):
        return len(self.gt_list)

    def __getitem__(self, idx):
        gt_img = Image.open(self.gt_list[idx])
        gt_img = self.transform(gt_img)
        gt_img = self.crop_preproc(gt_img)
        # output images scaled to -1,1
        gt_img = F.normalize(gt_img, mean=[0.5], std=[0.5])

        ori_h, ori_w = gt_img.size()[1:3]

        img_t_dir = self.gt_list[idx].replace("gt", "lr")
        img_ts = []

        if self.frame_number == 1:
            ids = [0]

        else:
            ids = random.sample(range(1, self.frame_number), k=(self.frame_number - 1))
            ids = [0, ] + ids

        for lr_index in ids:
            img_t = Image.open('{}/{}_{}.png'.format(img_t_dir.split('.png')[0], img_t_dir.split('.png')[0].split('/')[-1], lr_index))
            img_t = self.transform(img_t)
            img_t = self.crop_preproc_lr(img_t)
            if self.need_upscale:
                img_t = (torch.nn.functional.interpolate(img_t.unsqueeze(0), size=(ori_h, ori_w), mode='bicubic', align_corners=False).squeeze(0))
            # input images scaled to -1,1
            img_t = F.normalize(img_t, mean=[0.5], std=[0.5])

            img_ts.append(img_t)

        apply_trans = transforms_aug[random.getrandbits(3)]
        gt_img = getattr(augment, apply_trans)(gt_img)
        img_ts = [getattr(augment, apply_trans)(im) for im in img_ts]

        img_t_final = torch.stack(img_ts)

        example = {}
        example["neg_prompt"] = self.args.neg_prompt_vsd
        example["null_prompt"] = ""
        example["output_pixel_values"] = gt_img
        example["conditioning_pixel_values"] = img_t_final
        example["seq_name"] = img_t_dir.split('.png')[0].split('/')[-1]

        return example

class PairedSROnlineTxtDataset_multiframe_lownoise(torch.utils.data.Dataset):
    def __init__(self, args=None):
        super().__init__()

        self.args = args
        self.degradation = RealESRGAN_degradation(args.deg_file_path, device='cpu')
        # self.crop_preproc = transforms.Compose([
        #     transforms.RandomCrop((args.resolution, args.resolution)),
        #     transforms.RandomHorizontalFlip(),
        # ])
        self.crop_preproc = transforms.Compose([
            transforms.CenterCrop((args.resolution, args.resolution))
        ])
        self.crop_preproc_lr = transforms.Compose([
            transforms.CenterCrop((args.resolution, args.resolution))
        ])
        self.transform = transforms.Compose([transforms.ToTensor()])
        self.frame_number = args.frames
        self.need_upscale = args.need_upscale

        with open(args.dataset_txt_paths, 'r') as f:
            self.gt_list = [line.strip() for line in f.readlines()]

    def __len__(self):
        return len(self.gt_list)

    def __getitem__(self, idx):
        gt_img = Image.open(self.gt_list[idx])
        gt_img = self.transform(gt_img)
        gt_img = self.crop_preproc(gt_img)
        # output images scaled to -1,1
        gt_img = F.normalize(gt_img, mean=[0.5], std=[0.5])

        ori_h, ori_w = gt_img.size()[1:3]

        img_t_dir = self.gt_list[idx].replace("gt", "LR_frames_moderate_sisr")

        img_t = Image.open(img_t_dir)
        img_t = self.transform(img_t)
        img_t = self.crop_preproc_lr(img_t)
        if self.need_upscale:
            img_t = (torch.nn.functional.interpolate(img_t.unsqueeze(0), size=(ori_h, ori_w), mode='bicubic', align_corners=False).squeeze(0))
        # input images scaled to -1,1
        img_t = F.normalize(img_t, mean=[0.5], std=[0.5])
        img_ts = []
        img_ts.append(img_t)

        apply_trans = transforms_aug[random.getrandbits(3)]
        gt_img = getattr(augment, apply_trans)(gt_img)
        img_ts = [getattr(augment, apply_trans)(im) for im in img_ts]

        img_t_final = torch.stack(img_ts)

        example = {}
        example["neg_prompt"] = self.args.neg_prompt_vsd
        example["null_prompt"] = ""
        example["output_pixel_values"] = gt_img
        example["conditioning_pixel_values"] = img_t_final

        return example


class PairedSROnlineTxtDataset_multiframe_YouHQ(torch.utils.data.Dataset):
    def __init__(self, args=None):
        super().__init__()

        self.args = args
        self.degradation = RealESRGAN_degradation(args.deg_file_path, device='cpu')
        # self.crop_preproc = transforms.Compose([
        #     transforms.RandomCrop((args.resolution, args.resolution)),
        #     transforms.RandomHorizontalFlip(),
        # ])
        self.crop_preproc = transforms.Compose([
            transforms.CenterCrop((args.resolution, args.resolution))
        ])
        self.crop_preproc_lr = transforms.Compose([
            transforms.CenterCrop((args.resolution, args.resolution))
        ])
        self.transform = transforms.Compose([transforms.ToTensor()])
        self.frame_number = args.frames
        self.need_upscale = args.need_upscale

        with open(args.dataset_txt_paths, 'r') as f:
            self.gt_list = [line.strip() for line in f.readlines()]

    def __len__(self):
        return len(self.gt_list)

    def __getitem__(self, idx):
        gt_img = Image.open(self.gt_list[idx])
        gt_img = self.transform(gt_img)
        gt_img = self.crop_preproc(gt_img)
        # output images scaled to -1,1
        gt_img = F.normalize(gt_img, mean=[0.5], std=[0.5])

        ori_h, ori_w = gt_img.size()[1:3]

        img_t_dir = self.gt_list[idx].replace("GT_frames", "LR_moderate_frames")

        img_t = Image.open(img_t_dir)
        img_t = self.transform(img_t)
        img_t = self.crop_preproc_lr(img_t)
        if self.need_upscale:
            img_t = (torch.nn.functional.interpolate(img_t.unsqueeze(0), size=(ori_h, ori_w), mode='bicubic', align_corners=False).squeeze(0))
        # input images scaled to -1,1
        img_t = F.normalize(img_t, mean=[0.5], std=[0.5])
        img_ts = []
        img_ts.append(img_t)

        apply_trans = transforms_aug[random.getrandbits(3)]
        gt_img = getattr(augment, apply_trans)(gt_img)
        img_ts = [getattr(augment, apply_trans)(im) for im in img_ts]

        img_t_final = torch.stack(img_ts)

        example = {}
        example["neg_prompt"] = self.args.neg_prompt_vsd
        example["null_prompt"] = ""
        example["output_pixel_values"] = gt_img
        example["conditioning_pixel_values"] = img_t_final

        return example


class PairedSROnlineTxtDataset_multiframe_YouHQ_crossattn(torch.utils.data.Dataset):
    def __init__(self, args=None):
        super().__init__()

        self.args = args
        self.degradation = RealESRGAN_degradation(args.deg_file_path, device='cpu')
        # self.crop_preproc = transforms.Compose([
        #     transforms.RandomCrop((args.resolution, args.resolution)),
        #     transforms.RandomHorizontalFlip(),
        # ])
        self.crop_preproc = transforms.Compose([
            transforms.CenterCrop((args.resolution, args.resolution))
        ])
        self.crop_preproc_lr = transforms.Compose([
            transforms.CenterCrop((args.resolution, args.resolution))
        ])
        self.transform = transforms.Compose([transforms.ToTensor()])
        self.to_pil = transforms.Compose([transforms.ToPILImage()])
        self.frame_number = args.frames
        self.need_upscale = args.need_upscale

        with open(args.dataset_txt_paths, 'r') as f:
            self.gt_list = [line.strip() for line in f.readlines()]

    def compute_frame_difference_mask(self, frames):
        ambi_matrix = frames.var(dim=0)
        threshold = ambi_matrix.mean().item()
        mask_id = torch.where(ambi_matrix >= threshold, ambi_matrix, torch.zeros_like(ambi_matrix))
        frame_mask = torch.where(mask_id == 0, mask_id, torch.ones_like(mask_id))
        return frame_mask

    def visualize_and_save_mask(self, reference_frame, mask, save_path, file_name):
        overlay = reference_frame * mask
        overlay_pil = self.to_pil(overlay)
        gt_pil = self.to_pil(reference_frame)
        os.makedirs(save_path, exist_ok=True)
        overlay_pil.save(os.path.join(save_path, file_name))
        gt_pil.save(os.path.join(save_path, file_name.split('.png')[0]+'_gt.png'))

    def __len__(self):
        return len(self.gt_list)

    def __getitem__(self, idx):
        gt_img_path = self.gt_list[idx]
        gt_img = Image.open(gt_img_path)
        gt_img = self.transform(gt_img)
        gt_img = self.crop_preproc(gt_img)
        # output images scaled to -1,1
        gt_img = F.normalize(gt_img, mean=[0.5], std=[0.5])

        ori_h, ori_w = gt_img.size()[1:3]

        gt_frames_dir = os.path.dirname(gt_img_path)  # .../GT_frames
        parent_dir = os.path.dirname(gt_frames_dir)  # .../xxx
        img_t_dir = os.path.join(parent_dir, "LR_lowparam_nonoise_frames")

        gt_frame_name = os.path.basename(gt_img_path)  # e.g., frame_0002.png
        gt_frame_num_str = gt_frame_name.split('.')[0].split('_')[-1]  # '0002'
        gt_frame_num = int(gt_frame_num_str)  # 2

        # lr_frame_nums = list(range(gt_frame_num, gt_frame_num + self.args.frames))
        lr_frame_nums = list(range(gt_frame_num + 1, gt_frame_num + self.args.frames))
        # random.shuffle(lr_frame_nums)
        random.shuffle(lr_frame_nums)

        lr_frame_nums.insert(0, gt_frame_num)

        img_ts = []
        img_ts_gray = []

        for lr_num in lr_frame_nums:
            lr_frame_name = f"frame_{lr_num:04d}.png"  # e.g., frame_0002.png
            lr_frame_path = os.path.join(img_t_dir, lr_frame_name)
            img_t = Image.open(lr_frame_path)
            img_t_gray = img_t.convert("L")
            img_t = self.transform(img_t)
            img_t_gray = self.transform(img_t_gray)
            img_t = self.crop_preproc_lr(img_t)
            img_t_gray = self.crop_preproc_lr(img_t_gray)
            if self.need_upscale:
                img_t = (torch.nn.functional.interpolate(img_t.unsqueeze(0), size=(ori_h, ori_w), mode='bicubic', align_corners=False).squeeze(0))
                img_t_gray = (torch.nn.functional.interpolate(img_t_gray.unsqueeze(0), size=(ori_h, ori_w), mode='bicubic', align_corners=False).squeeze(0))

            # input images scaled to -1,1
            img_t = F.normalize(img_t, mean=[0.5], std=[0.5])

            img_ts.append(img_t)
            img_ts_gray.append(img_t_gray)

        apply_trans = transforms_aug[random.getrandbits(3)]
        gt_img = getattr(augment, apply_trans)(gt_img)
        img_ts = [getattr(augment, apply_trans)(im) for im in img_ts]
        img_ts_gray = [getattr(augment, apply_trans)(im_gray) for im_gray in img_ts_gray]

        img_t_final = torch.stack(img_ts)
        img_t_final_gray = torch.stack(img_ts_gray)

        frame_difference_mask = self.compute_frame_difference_mask(img_t_final_gray)

        # self.visualize_and_save_mask(
        #     img_t_final[0]*0.5+0.5,
        #     frame_difference_mask,
        #     save_path="visualized_masks",
        #     file_name=f"mask_overlay_{idx}.png"
        # )

        example = {}
        example["neg_prompt"] = self.args.neg_prompt_vsd
        example["null_prompt"] = ""
        example["output_pixel_values"] = gt_img
        example["conditioning_pixel_values"] = img_t_final
        example["seq_name"] = parent_dir.split('/')[-1]
        example["difference_mask"] = frame_difference_mask

        return example


class PairedSROnlineTxtDataset_multiframe_YouHQ_crossattn_videos(torch.utils.data.Dataset):
    def __init__(self, args=None):
        super().__init__()

        self.args = args
        self.degradation = RealESRGAN_degradation(args.deg_file_path, device='cpu')
        # self.crop_preproc = transforms.Compose([
        #     transforms.RandomCrop((args.resolution, args.resolution)),
        #     transforms.RandomHorizontalFlip(),
        # ])
        # self.crop_preproc = transforms.Compose([
        #     transforms.CenterCrop((args.resolution, args.resolution))
        # ])
        # self.crop_preproc_lr = transforms.Compose([
        #     transforms.CenterCrop((args.resolution, args.resolution))
        # ])

        self.transform = transforms.Compose([transforms.ToTensor()])
        self.to_pil = transforms.Compose([transforms.ToPILImage()])
        self.frame_number = args.frames
        self.need_upscale = args.need_upscale

        with open(args.dataset_txt_paths, 'r') as f:
            self.gt_list = [line.strip() for line in f.readlines()]

    def compute_frame_difference_mask(self, frames):
        ambi_matrix = frames.var(dim=0)
        threshold = ambi_matrix.mean().item()
        mask_id = torch.where(ambi_matrix >= threshold, ambi_matrix, torch.zeros_like(ambi_matrix))
        frame_mask = torch.where(mask_id == 0, mask_id, torch.ones_like(mask_id))
        return frame_mask

    def visualize_and_save_mask(self, reference_frame, mask, save_path, file_name):
        overlay = reference_frame * mask
        overlay_pil = self.to_pil(overlay)
        gt_pil = self.to_pil(reference_frame)
        os.makedirs(save_path, exist_ok=True)
        overlay_pil.save(os.path.join(save_path, file_name))
        gt_pil.save(os.path.join(save_path, file_name.split('.png')[0]+'_gt.png'))

    def __len__(self):
        return len(self.gt_list)

    def __getitem__(self, idx):
        gt_img_path = self.gt_list[idx]

        gt_frames_dir = os.path.dirname(gt_img_path)  # .../GT_frames
        parent_dir = os.path.dirname(gt_frames_dir)  # .../xxx
        # img_t_dir = os.path.join(parent_dir, "LR_lowparam_nonoise_frames")
        img_t_dir = os.path.join(parent_dir, "LR_frames")

        gt_frame_name = os.path.basename(gt_img_path)  # e.g., frame_0002.png
        gt_frame_num_str = gt_frame_name.split('.')[0].split('_')[-1]  # '0002'
        gt_frame_num = int(gt_frame_num_str)  # 2

        lr_frame_nums = list(range(gt_frame_num, gt_frame_num + self.args.frames))

        img_ts = []
        img_ts_gt = []
        img_ts_gray = []

        for lr_num in lr_frame_nums:
            lr_frame_name = f"frame_{lr_num:04d}.png"  # e.g., frame_0002.png
            lr_frame_path = os.path.join(img_t_dir, lr_frame_name)
            gt_frame_path = os.path.join(gt_frames_dir, lr_frame_name)

            # read image
            img_t = Image.open(lr_frame_path)
            img_t_gt = Image.open(gt_frame_path)
            img_t_gray = img_t.convert("L")
            # to tensor
            img_t = self.transform(img_t)
            img_t_gt = self.transform(img_t_gt)
            img_t_gray = self.transform(img_t_gray)

            print("img_t name: {}, img_t.shape: {}".format(gt_frame_path, img_t.shape))
            # crop
            img_t_gt, img_t, img_t_gray = random_crop_same_3(
                img_t_gt, img_t, img_t_gray, (self.args.resolution, self.args.resolution)
            )
            # img_t = self.crop_preproc_lr(img_t)
            # img_t_gt = self.crop_preproc(img_t_gt)
            # img_t_gray = self.crop_preproc_lr(img_t_gray)

            # upscale
            ori_h, ori_w = img_t_gt.size()[1:3]
            if self.need_upscale:
                img_t = (torch.nn.functional.interpolate(img_t.unsqueeze(0), size=(ori_h, ori_w), mode='bicubic', align_corners=False).squeeze(0))
            img_t_gray = (torch.nn.functional.interpolate(img_t_gray.unsqueeze(0), size=(ori_h//8, ori_w//8), mode='bicubic', align_corners=False).squeeze(0))

            # input images scaled to -1,1
            img_t = F.normalize(img_t, mean=[0.5], std=[0.5])
            img_t_gt = F.normalize(img_t_gt, mean=[0.5], std=[0.5])
            # save
            img_ts.append(img_t)
            img_ts_gt.append(img_t_gt)
            img_ts_gray.append(img_t_gray)


        apply_trans = transforms_aug[random.getrandbits(3)]
        img_ts_gt = [getattr(augment, apply_trans)(im_gt) for im_gt in img_ts_gt]
        img_ts = [getattr(augment, apply_trans)(im) for im in img_ts]
        img_ts_gray = [getattr(augment, apply_trans)(im_gray) for im_gray in img_ts_gray]

        img_t_final = torch.stack(img_ts)
        img_t_final_gt = torch.stack(img_ts_gt)
        img_t_final_gray = torch.stack(img_ts_gray)

        frame_difference_mask = []
        for frame_counter in range(self.args.frames):
            if frame_counter != 0:
                current_frame = img_t_final_gray[frame_counter]
                prev_frame = img_t_final_gray[frame_counter - 1]
                frames_compute = torch.stack([current_frame, prev_frame])
                frame_difference_mask_each = self.compute_frame_difference_mask(frames_compute)
                frame_difference_mask.append(frame_difference_mask_each)

        frame_difference_mask = torch.stack(frame_difference_mask)

        # self.visualize_and_save_mask(
        #     img_t_final[0]*0.5+0.5,
        #     frame_difference_mask,
        #     save_path="visualized_masks",
        #     file_name=f"mask_overlay_{idx}.png"
        # )

        example = {}
        example["neg_prompt"] = self.args.neg_prompt_vsd
        example["null_prompt"] = ""
        example["output_pixel_values"] = img_t_final_gt
        example["conditioning_pixel_values"] = img_t_final
        example["seq_name"] = parent_dir.split('/')[-1]
        example["difference_mask"] = frame_difference_mask

        return example


class PairedSROnlineTxtDataset_multiframe_VideoLQ(torch.utils.data.Dataset):
    def __init__(self, args=None):
        super().__init__()

        self.args = args
        self.degradation = RealESRGAN_degradation(args.deg_file_path, device='cpu')
        # self.crop_preproc = transforms.Compose([
        #     transforms.RandomCrop((args.resolution//4, args.resolution//4))
        # ])
        # self.crop_preproc_lr = transforms.Compose([
        #     transforms.RandomCrop((args.resolution//4, args.resolution//4))
        # ])
        self.crop_preproc = transforms.Compose([
            transforms.CenterCrop((args.resolution//4, args.resolution//4))
        ])
        self.crop_preproc_lr = transforms.Compose([
            transforms.CenterCrop((args.resolution//4, args.resolution//4))
        ])
        self.transform = transforms.Compose([transforms.ToTensor()])
        self.to_pil = transforms.Compose([transforms.ToPILImage()])
        self.frame_number = args.frames
        self.need_upscale = args.need_upscale

        with open(args.dataset_txt_paths, 'r') as f:
            self.gt_list = [line.strip() for line in f.readlines()]

    def compute_frame_difference_mask(self, frames):
        ambi_matrix = frames.var(dim=0)
        threshold = ambi_matrix.mean().item()
        mask_id = torch.where(ambi_matrix >= threshold, ambi_matrix, torch.zeros_like(ambi_matrix))
        frame_mask = torch.where(mask_id == 0, mask_id, torch.ones_like(mask_id))
        return frame_mask

    def visualize_and_save_mask(self, reference_frame, mask, save_path, file_name):
        overlay = reference_frame * mask
        overlay_pil = self.to_pil(overlay)
        gt_pil = self.to_pil(reference_frame)
        os.makedirs(save_path, exist_ok=True)
        overlay_pil.save(os.path.join(save_path, file_name))
        gt_pil.save(os.path.join(save_path, file_name.split('.png')[0]+'_gt.png'))

    def __len__(self):
        return len(self.gt_list)

    def __getitem__(self, idx):
        gt_img_path = self.gt_list[idx]

        # /home/notebook/data/group/syj/dataset/VideoLQ/015/00000000.png
        gt_frames_dir = os.path.dirname(gt_img_path)  # /home/notebook/data/group/syj/dataset/VideoLQ/015
        parent_dir = os.path.dirname(gt_frames_dir)  # /home/notebook/data/group/syj/dataset/VideoLQ
        # img_t_dir = os.path.join(parent_dir, "LR_lowparam_nonoise_frames")
        # img_t_dir = os.path.join(parent_dir, "LR_frames")
        img_t_dir = gt_frames_dir

        gt_frame_name = os.path.basename(gt_img_path)  # e.g., 00000000.png
        gt_frame_num_str = gt_frame_name.split('.')[0]  # '00000000'
        gt_frame_num = int(gt_frame_num_str)  # 0
        lr_frame_nums = list(range(gt_frame_num, gt_frame_num + self.frame_number))

        img_ts = []
        img_ts_gt = []
        img_ts_gray = []

        for lr_num in lr_frame_nums:
            lr_frame_name = f"{lr_num:08d}.png"  # e.g., frame_0002.png
            lr_frame_path = os.path.join(img_t_dir, lr_frame_name)
            gt_frame_path = os.path.join(gt_frames_dir, lr_frame_name)

            # read image
            img_t = Image.open(lr_frame_path)
            img_t_gt = Image.open(gt_frame_path)
            img_t_gray = img_t.convert("L")
            # crop
            img_t = self.crop_preproc_lr(img_t)
            img_t_gt = self.crop_preproc_lr(img_t_gt)
            img_t_gray = self.crop_preproc_lr(img_t_gray)
            # to tensor
            img_t = self.transform(img_t)
            img_t_gt = self.transform(img_t_gt)
            img_t_gt = torch.nn.functional.interpolate(img_t_gt.unsqueeze(0), scale_factor=4, mode='bicubic', align_corners=False).squeeze(0)
            img_t_gray = self.transform(img_t_gray)

            # upscale
            ori_h, ori_w = img_t_gt.size()[1:3]
            if self.need_upscale:
                img_t = (torch.nn.functional.interpolate(img_t.unsqueeze(0), size=(ori_h, ori_w), mode='bicubic', align_corners=False).squeeze(0))
            img_t_gray = (torch.nn.functional.interpolate(img_t_gray.unsqueeze(0), size=(ori_h//8, ori_w//8), mode='bicubic', align_corners=False).squeeze(0))

            # input images scaled to -1,1
            img_t = F.normalize(img_t, mean=[0.5], std=[0.5])
            img_t_gt = F.normalize(img_t_gt, mean=[0.5], std=[0.5])
            # save
            img_ts.append(img_t)
            img_ts_gt.append(img_t_gt)
            img_ts_gray.append(img_t_gray)


        # apply_trans = transforms_aug[random.getrandbits(3)]
        # img_ts_gt = [getattr(augment, apply_trans)(im_gt) for im_gt in img_ts_gt]
        # img_ts = [getattr(augment, apply_trans)(im) for im in img_ts]
        # img_ts_gray = [getattr(augment, apply_trans)(im_gray) for im_gray in img_ts_gray]

        img_t_final = torch.stack(img_ts)
        img_t_final_gt = torch.stack(img_ts_gt)
        img_t_final_gray = torch.stack(img_ts_gray)

        frame_difference_mask = []
        for frame_counter in range(self.frame_number):
            if frame_counter != 0:
                current_frame = img_t_final_gray[frame_counter]
                prev_frame = img_t_final_gray[frame_counter - 1]
                frames_compute = torch.stack([current_frame, prev_frame])
                frame_difference_mask_each = self.compute_frame_difference_mask(frames_compute)
                frame_difference_mask.append(frame_difference_mask_each)

        frame_difference_mask = torch.stack(frame_difference_mask)

        # self.visualize_and_save_mask(
        #     img_t_final[0]*0.5+0.5,
        #     frame_difference_mask,
        #     save_path="visualized_masks",
        #     file_name=f"mask_overlay_{idx}.png"
        # )

        example = {}
        example["neg_prompt"] = self.args.neg_prompt_vsd
        example["null_prompt"] = ""
        example["output_pixel_values"] = img_t_final_gt
        example["conditioning_pixel_values"] = img_t_final
        example["seq_name"] = gt_frames_dir.replace('/', '-')
        example["difference_mask"] = frame_difference_mask

        return example




class PairedSROnlineTxtDataset_YouHQ_and_LSDIR(torch.utils.data.Dataset):
    def __init__(self, args=None):
        super().__init__()

        self.args = args
        self.degradation = RealESRGAN_degradation(args.deg_file_path, device='cpu')
        # self.crop_preproc = transforms.Compose([
        #     transforms.RandomCrop((args.resolution, args.resolution)),
        #     transforms.RandomHorizontalFlip(),
        # ])
        self.crop_preproc = transforms.Compose([
            transforms.CenterCrop((args.resolution, args.resolution))
        ])
        self.crop_preproc_lr = transforms.Compose([
            transforms.CenterCrop((args.resolution, args.resolution))
        ])
        self.transform = transforms.Compose([transforms.ToTensor()])
        self.to_pil = transforms.Compose([transforms.ToPILImage()])
        self.frame_number = args.frames
        self.need_upscale = args.need_upscale

        with open(args.dataset_txt_paths, 'r') as f:
            self.gt_list = [line.strip() for line in f.readlines()]

    def compute_frame_difference_mask(self, frames):
        ambi_matrix = frames.var(dim=0)
        threshold = ambi_matrix.mean().item()
        mask_id = torch.where(ambi_matrix >= threshold, ambi_matrix, torch.zeros_like(ambi_matrix))
        frame_mask = torch.where(mask_id == 0, mask_id, torch.ones_like(mask_id))
        return frame_mask

    def visualize_and_save_mask(self, reference_frame, mask, save_path, file_name):
        overlay = reference_frame * mask
        overlay_pil = self.to_pil(overlay)
        gt_pil = self.to_pil(reference_frame)
        os.makedirs(save_path, exist_ok=True)
        overlay_pil.save(os.path.join(save_path, file_name))
        gt_pil.save(os.path.join(save_path, file_name.split('.png')[0]+'_gt.png'))

    def __len__(self):
        return len(self.gt_list)

    def __getitem__(self, idx):
        gt_img_path = self.gt_list[idx]

        if "LSDIR" in gt_img_path:
            gt_img = Image.open(self.gt_list[idx])
            gt_img = self.transform(gt_img)
            gt_img = self.crop_preproc(gt_img)
            # output images scaled to -1,1
            gt_img = F.normalize(gt_img, mean=[0.5], std=[0.5])

            ori_h, ori_w = gt_img.size()[1:3]

            img_t_dir = self.gt_list[idx].replace("gt", "lr")
            img_ts = []
            img_ts_gray = []

            if self.frame_number == 1:
                ids = [0]

            else:
                ids = random.sample(range(1, self.frame_number), k=(self.frame_number - 1))
                ids = [0, ] + ids

            for lr_index in ids:
                img_t = Image.open(
                    '{}/{}_{}.png'.format(img_t_dir.split('.png')[0], img_t_dir.split('.png')[0].split('/')[-1],
                                          lr_index))
                img_t_gray = img_t.convert("L")
                img_t = self.transform(img_t)
                img_t = self.crop_preproc_lr(img_t)
                img_t_gray = self.transform(img_t_gray)
                img_t_gray = self.crop_preproc_lr(img_t_gray)
                img_t = (torch.nn.functional.interpolate(img_t.unsqueeze(0), size=(ori_h, ori_w), mode='bicubic',
                                                         align_corners=False).squeeze(0))
                img_t_gray = (torch.nn.functional.interpolate(img_t_gray.unsqueeze(0), size=(ori_h//8, ori_w//8), mode='bicubic',
                                                              align_corners=False).squeeze(0))

                # input images scaled to -1,1
                img_t = F.normalize(img_t, mean=[0.5], std=[0.5])

                img_ts.append(img_t)
                img_ts_gray.append(img_t_gray)

            gt_img = gt_img.unsqueeze(0).repeat(self.args.frames, 1, 1, 1)

            apply_trans = transforms_aug[random.getrandbits(3)]
            gt_img = [getattr(augment, apply_trans)(gt_im) for gt_im in gt_img]
            img_ts = [getattr(augment, apply_trans)(im) for im in img_ts]
            img_ts_gray = [getattr(augment, apply_trans)(im_gray) for im_gray in img_ts_gray]

            img_t_final = torch.stack(img_ts)
            gt_img_final = torch.stack(gt_img)
            img_t_final_gray = torch.stack(img_ts_gray)

            frame_difference_mask = []
            for frame_counter in range(self.args.frames):
                if frame_counter != 0:
                    current_frame = img_t_final_gray[frame_counter]
                    prev_frame = img_t_final_gray[frame_counter - 1]
                    frames_compute = torch.stack([current_frame, prev_frame])
                    frame_difference_mask_each = self.compute_frame_difference_mask(frames_compute)
                    frame_difference_mask.append(frame_difference_mask_each)

            frame_difference_mask = torch.stack(frame_difference_mask)

            example = {}
            example["neg_prompt"] = self.args.neg_prompt_vsd
            example["null_prompt"] = ""
            example["output_pixel_values"] = gt_img_final
            example["conditioning_pixel_values"] = img_t_final
            example["seq_name"] = img_t_dir.split('.png')[0].split('/')[-1]
            example["difference_mask"] = frame_difference_mask
            example["is_lsdir"] = 1

        elif "YouHQ" in gt_img_path:
            gt_frames_dir = os.path.dirname(gt_img_path)  # .../GT_frames
            parent_dir = os.path.dirname(gt_frames_dir)  # .../xxx
            # img_t_dir = os.path.join(parent_dir, "LR_lowparam_nonoise_frames")
            img_t_dir = os.path.join(parent_dir, "LR_frames")

            gt_frame_name = os.path.basename(gt_img_path)  # e.g., frame_0002.png
            gt_frame_num_str = gt_frame_name.split('.')[0].split('_')[-1]  # '0002'
            gt_frame_num = int(gt_frame_num_str)  # 2

            lr_frame_nums = list(range(gt_frame_num, gt_frame_num + self.args.frames))

            img_ts = []
            img_ts_gt = []
            img_ts_gray = []

            for lr_num in lr_frame_nums:
                lr_frame_name = f"frame_{lr_num:04d}.png"  # e.g., frame_0002.png
                lr_frame_path = os.path.join(img_t_dir, lr_frame_name)
                gt_frame_path = os.path.join(gt_frames_dir, lr_frame_name)

                # read image
                img_t = Image.open(lr_frame_path)
                img_t_gt = Image.open(gt_frame_path)
                img_t_gray = img_t.convert("L")
                # to tensor
                img_t = self.transform(img_t)
                img_t_gt = self.transform(img_t_gt)
                img_t_gray = self.transform(img_t_gray)
                # crop
                img_t = self.crop_preproc_lr(img_t)
                img_t_gt = self.crop_preproc(img_t_gt)
                img_t_gray = self.crop_preproc_lr(img_t_gray)
                # upscale
                ori_h, ori_w = img_t_gt.size()[1:3]
                if self.need_upscale:
                    img_t = (torch.nn.functional.interpolate(img_t.unsqueeze(0), size=(ori_h, ori_w), mode='bicubic', align_corners=False).squeeze(0))
                img_t_gray = (torch.nn.functional.interpolate(img_t_gray.unsqueeze(0), size=(ori_h//8, ori_w//8), mode='bicubic', align_corners=False).squeeze(0))

                # input images scaled to -1,1
                img_t = F.normalize(img_t, mean=[0.5], std=[0.5])
                img_t_gt = F.normalize(img_t_gt, mean=[0.5], std=[0.5])
                # save
                img_ts.append(img_t)
                img_ts_gt.append(img_t_gt)
                img_ts_gray.append(img_t_gray)

            apply_trans = transforms_aug[random.getrandbits(3)]
            img_ts_gt = [getattr(augment, apply_trans)(im_gt) for im_gt in img_ts_gt]
            img_ts = [getattr(augment, apply_trans)(im) for im in img_ts]
            img_ts_gray = [getattr(augment, apply_trans)(im_gray) for im_gray in img_ts_gray]

            img_t_final = torch.stack(img_ts)
            img_t_final_gt = torch.stack(img_ts_gt)
            img_t_final_gray = torch.stack(img_ts_gray)

            frame_difference_mask = []
            for frame_counter in range(self.args.frames):
                if frame_counter != 0:
                    current_frame = img_t_final_gray[frame_counter]
                    prev_frame = img_t_final_gray[frame_counter - 1]
                    frames_compute = torch.stack([current_frame, prev_frame])
                    frame_difference_mask_each = self.compute_frame_difference_mask(frames_compute)
                    frame_difference_mask.append(frame_difference_mask_each)

            frame_difference_mask = torch.stack(frame_difference_mask)

            # self.visualize_and_save_mask(
            #     img_t_final[0]*0.5+0.5,
            #     frame_difference_mask,
            #     save_path="visualized_masks",
            #     file_name=f"mask_overlay_{idx}.png"
            # )

            example = {}
            example["neg_prompt"] = self.args.neg_prompt_vsd
            example["null_prompt"] = ""
            example["output_pixel_values"] = img_t_final_gt
            example["conditioning_pixel_values"] = img_t_final
            example["seq_name"] = parent_dir.replace('/', '-')
            example["difference_mask"] = frame_difference_mask
            example["is_lsdir"] = 0

        return example



def resize_and_crop(image, size):
    """
    Resize 图像的短边到目标 size，然后进行中心裁剪到指定尺寸。
    使用 Box 模式以避免锯齿。
    
    :param image: PIL.Image 对象
    :param size: 最终输出的裁剪尺寸 (height, width)
    :return: 裁剪后的图像 (PIL.Image)
    """
    # 获取图像的宽高
    width, height = image.size
    
    # 计算 resize 的新高度和宽度
    if width < height:
        new_width = size[0]
        new_height = int(height * (size[0] / width))
    else:
        new_height = size[1]
        new_width = int(width * (size[1] / height))
    
    # Resize 使用 Box 模式
    image = image.resize((new_width, new_height), Image.BOX)
    
    # CenterCrop
    left = (image.width - size[0]) // 2
    top = (image.height - size[1]) // 2
    right = left + size[0]
    bottom = top + size[1]
    
    image = image.crop((left, top, right, bottom))
    return image

class PairedSROnlineTxtDataset_Pexel_and_LSDIR(torch.utils.data.Dataset):
    def __init__(self, args=None):
        super().__init__()

        self.args = args
        self.degradation = RealESRGAN_degradation(args.deg_file_path, device='cpu')
        # self.crop_preproc = transforms.Compose([
        #     transforms.RandomCrop((args.resolution, args.resolution)),
        #     transforms.RandomHorizontalFlip(),
        # ])
        # self.crop_preproc = transforms.Compose([
        #     transforms.CenterCrop((args.resolution, args.resolution))
        # ])
        # self.crop_preproc_lr = transforms.Compose([
        #     transforms.CenterCrop((args.resolution, args.resolution))
        # ])
        self.crop_preproc = transforms.Compose([
            transforms.Lambda(lambda img: resize_and_crop(img, (args.resolution, args.resolution)))
        ])
        self.crop_preproc_lr = transforms.Compose([
            transforms.Lambda(lambda img: resize_and_crop(img, (args.resolution, args.resolution)))
        ])
        self.transform = transforms.Compose([transforms.ToTensor()])
        self.to_pil = transforms.Compose([transforms.ToPILImage()])
        self.frame_number = args.frames
        self.need_upscale = args.need_upscale

        with open(args.dataset_txt_paths, 'r') as f:
            self.gt_list = [line.strip() for line in f.readlines()]

    def compute_frame_difference_mask(self, frames):
        ambi_matrix = frames.var(dim=0)
        threshold = ambi_matrix.mean().item()
        mask_id = torch.where(ambi_matrix >= threshold, ambi_matrix, torch.zeros_like(ambi_matrix))
        frame_mask = torch.where(mask_id == 0, mask_id, torch.ones_like(mask_id))
        return frame_mask

    def visualize_and_save_mask(self, reference_frame, mask, save_path, file_name):
        overlay = reference_frame * mask
        overlay_pil = self.to_pil(overlay)
        gt_pil = self.to_pil(reference_frame)
        os.makedirs(save_path, exist_ok=True)
        overlay_pil.save(os.path.join(save_path, file_name))
        gt_pil.save(os.path.join(save_path, file_name.split('.png')[0]+'_gt.png'))

    def __len__(self):
        return len(self.gt_list)

    def __getitem__(self, idx):
        gt_img_path = self.gt_list[idx]

        if "LSDIR" in gt_img_path:
            gt_img = Image.open(self.gt_list[idx])
            gt_img = self.crop_preproc(gt_img)
            gt_img = self.transform(gt_img)
            # gt_img = self.crop_preproc(gt_img)
            # output images scaled to -1,1
            gt_img = F.normalize(gt_img, mean=[0.5], std=[0.5])

            ori_h, ori_w = gt_img.size()[1:3]

            img_t_dir = self.gt_list[idx].replace("gt", "lr")
            img_ts = []
            img_ts_gray = []

            if self.frame_number == 1:
                ids = [0]

            else:
                ids = random.sample(range(1, self.frame_number), k=(self.frame_number - 1))
                ids = [0, ] + ids

            for lr_index in ids:
                img_t = Image.open(
                    '{}/{}_{}.png'.format(img_t_dir.split('.png')[0], img_t_dir.split('.png')[0].split('/')[-1],
                                          lr_index))
                img_t_gray = img_t.convert("L")
                img_t = self.crop_preproc_lr(img_t)
                img_t = self.transform(img_t)
                # img_t = self.crop_preproc_lr(img_t)
                img_t_gray = self.crop_preproc_lr(img_t_gray)
                img_t_gray = self.transform(img_t_gray)
                # img_t_gray = self.crop_preproc_lr(img_t_gray)
                img_t = (torch.nn.functional.interpolate(img_t.unsqueeze(0), size=(ori_h, ori_w), mode='bicubic',
                                                         align_corners=False).squeeze(0))
                img_t_gray = (torch.nn.functional.interpolate(img_t_gray.unsqueeze(0), size=(ori_h//8, ori_w//8), mode='bicubic',
                                                              align_corners=False).squeeze(0))

                # input images scaled to -1,1
                img_t = F.normalize(img_t, mean=[0.5], std=[0.5])

                img_ts.append(img_t)
                img_ts_gray.append(img_t_gray)

            gt_img = gt_img.unsqueeze(0).repeat(self.args.frames, 1, 1, 1)

            apply_trans = transforms_aug[random.getrandbits(3)]
            gt_img = [getattr(augment, apply_trans)(gt_im) for gt_im in gt_img]
            img_ts = [getattr(augment, apply_trans)(im) for im in img_ts]
            img_ts_gray = [getattr(augment, apply_trans)(im_gray) for im_gray in img_ts_gray]

            img_t_final = torch.stack(img_ts)
            gt_img_final = torch.stack(gt_img)
            img_t_final_gray = torch.stack(img_ts_gray)

            frame_difference_mask = []
            for frame_counter in range(self.args.frames):
                if frame_counter != 0:
                    current_frame = img_t_final_gray[frame_counter]
                    prev_frame = img_t_final_gray[frame_counter - 1]
                    frames_compute = torch.stack([current_frame, prev_frame])
                    frame_difference_mask_each = self.compute_frame_difference_mask(frames_compute)
                    frame_difference_mask.append(frame_difference_mask_each)

            frame_difference_mask = torch.stack(frame_difference_mask)

            example = {}
            example["neg_prompt"] = self.args.neg_prompt_vsd
            example["null_prompt"] = ""
            example["output_pixel_values"] = gt_img_final
            example["conditioning_pixel_values"] = img_t_final
            example["seq_name"] = img_t_dir.split('.png')[0].split('/')[-1]
            example["difference_mask"] = frame_difference_mask
            example["is_lsdir"] = 1

        elif "Pexel" in gt_img_path:
            gt_frames_dir = os.path.dirname(gt_img_path)  # .../GT_frames
            parent_dir = os.path.dirname(gt_frames_dir)  # .../560818491_1
            img_t_dir = os.path.join(parent_dir, "LR_frames")

            gt_frame_name = os.path.basename(gt_img_path)  # 例如，frame_1.png
            gt_frame_num_str = gt_frame_name.split('.')[0].split('_')[-1]  # '1'
            gt_frame_num = int(gt_frame_num_str)  # 1

            lr_frame_nums = list(range(gt_frame_num, gt_frame_num + self.args.frames))

            img_ts = []
            img_ts_gt = []
            img_ts_gray = []

            for lr_num in lr_frame_nums:
                lr_frame_name = f"frame_{lr_num}.png"  # 例如，frame_1.png
                lr_frame_path = os.path.join(img_t_dir, lr_frame_name)
                gt_frame_path = os.path.join(gt_frames_dir, lr_frame_name)

                # 读取图像
                img_t = Image.open(lr_frame_path)
                img_t_gt = Image.open(gt_frame_path)
                img_t_gray = img_t.convert("L")

                # 裁剪
                img_t = self.crop_preproc_lr(img_t)
                img_t_gt = self.crop_preproc(img_t_gt)
                img_t_gray = self.crop_preproc_lr(img_t_gray)
                
                # 转换为tensor
                img_t = self.transform(img_t)
                img_t_gt = self.transform(img_t_gt)
                img_t_gray = self.transform(img_t_gray)
                
                # # 裁剪
                # img_t = self.crop_preproc_lr(img_t)
                # img_t_gt = self.crop_preproc(img_t_gt)
                # img_t_gray = self.crop_preproc_lr(img_t_gray)
                
                # 上采样（如果需要）
                ori_h, ori_w = img_t_gt.size()[1:3]
                if self.need_upscale:
                    img_t = (torch.nn.functional.interpolate(img_t.unsqueeze(0), size=(ori_h, ori_w), mode='bicubic', align_corners=False).squeeze(0))
                img_t_gray = (torch.nn.functional.interpolate(img_t_gray.unsqueeze(0), size=(ori_h//8, ori_w//8), mode='bicubic', align_corners=False).squeeze(0))

                # 标准化到 [-1, 1]
                img_t = F.normalize(img_t, mean=[0.5], std=[0.5])
                img_t_gt = F.normalize(img_t_gt, mean=[0.5], std=[0.5])
                
                # 添加到列表
                img_ts.append(img_t)
                img_ts_gt.append(img_t_gt)
                img_ts_gray.append(img_t_gray)

            # 应用数据增强
            apply_trans = transforms_aug[random.getrandbits(3)]
            img_ts_gt = [getattr(augment, apply_trans)(im_gt) for im_gt in img_ts_gt]
            img_ts = [getattr(augment, apply_trans)(im) for im in img_ts]
            img_ts_gray = [getattr(augment, apply_trans)(im_gray) for im_gray in img_ts_gray]

            # 堆叠帧
            img_t_final = torch.stack(img_ts)
            img_t_final_gt = torch.stack(img_ts_gt)
            img_t_final_gray = torch.stack(img_ts_gray)

            frame_difference_mask = []
            for frame_counter in range(self.args.frames):
                if frame_counter != 0:
                    current_frame = img_t_final_gray[frame_counter]
                    prev_frame = img_t_final_gray[frame_counter - 1]
                    frames_compute = torch.stack([current_frame, prev_frame])
                    frame_difference_mask_each = self.compute_frame_difference_mask(frames_compute)
                    frame_difference_mask.append(frame_difference_mask_each)

            frame_difference_mask = torch.stack(frame_difference_mask)

            example = {}
            example["neg_prompt"] = self.args.neg_prompt_vsd
            example["null_prompt"] = ""
            example["output_pixel_values"] = img_t_final_gt
            example["conditioning_pixel_values"] = img_t_final
            example["seq_name"] = parent_dir.replace('/', '-')
            example["difference_mask"] = frame_difference_mask
            example["is_lsdir"] = 0

        return example


class PairedSROnlineTxtDataset_Pexel_and_REDS_and_LSDIR(torch.utils.data.Dataset):
    def __init__(self, args=None):
        super().__init__()

        self.args = args
        self.degradation = RealESRGAN_degradation(args.deg_file_path, device='cpu')
        # self.crop_preproc = transforms.Compose([
        #     transforms.RandomCrop((args.resolution, args.resolution)),
        #     transforms.RandomHorizontalFlip(),
        # ])
        # self.crop_preproc = transforms.Compose([
        #     transforms.CenterCrop((args.resolution, args.resolution))
        # ])
        # self.crop_preproc_lr = transforms.Compose([
        #     transforms.CenterCrop((args.resolution, args.resolution))
        # ])
        self.crop_preproc = transforms.Compose([
            transforms.Lambda(lambda img: resize_and_crop(img, (args.resolution, args.resolution)))
        ])
        self.crop_preproc_lr = transforms.Compose([
            transforms.Lambda(lambda img: resize_and_crop(img, (args.resolution, args.resolution)))
        ])
        self.transform = transforms.Compose([transforms.ToTensor()])
        self.to_pil = transforms.Compose([transforms.ToPILImage()])
        self.frame_number = args.frames
        self.need_upscale = args.need_upscale

        with open(args.dataset_txt_paths, 'r') as f:
            self.gt_list = [line.strip() for line in f.readlines()]

    def compute_frame_difference_mask(self, frames):
        ambi_matrix = frames.var(dim=0)
        threshold = ambi_matrix.mean().item()
        mask_id = torch.where(ambi_matrix >= threshold, ambi_matrix, torch.zeros_like(ambi_matrix))
        frame_mask = torch.where(mask_id == 0, mask_id, torch.ones_like(mask_id))
        return frame_mask

    def visualize_and_save_mask(self, reference_frame, mask, save_path, file_name):
        overlay = reference_frame * mask
        overlay_pil = self.to_pil(overlay)
        gt_pil = self.to_pil(reference_frame)
        os.makedirs(save_path, exist_ok=True)
        overlay_pil.save(os.path.join(save_path, file_name))
        gt_pil.save(os.path.join(save_path, file_name.split('.png')[0]+'_gt.png'))

    def __len__(self):
        return len(self.gt_list)

    def __getitem__(self, idx):
        gt_img_path = self.gt_list[idx]

        if "LSDIR" in gt_img_path:
            gt_img = Image.open(self.gt_list[idx])
            gt_img = self.crop_preproc(gt_img)
            gt_img = self.transform(gt_img)
            # gt_img = self.crop_preproc(gt_img)
            # output images scaled to -1,1
            gt_img = F.normalize(gt_img, mean=[0.5], std=[0.5])

            ori_h, ori_w = gt_img.size()[1:3]

            img_t_dir = self.gt_list[idx].replace("gt", "lr")
            img_ts = []
            img_ts_gray = []

            if self.frame_number == 1:
                ids = [0]

            else:
                ids = random.sample(range(1, self.frame_number), k=(self.frame_number - 1))
                ids = [0, ] + ids

            for lr_index in ids:
                img_t = Image.open(
                    '{}/{}_{}.png'.format(img_t_dir.split('.png')[0], img_t_dir.split('.png')[0].split('/')[-1],
                                          lr_index))
                img_t_gray = img_t.convert("L")
                img_t = self.crop_preproc_lr(img_t)
                img_t = self.transform(img_t)
                # img_t = self.crop_preproc_lr(img_t)
                img_t_gray = self.crop_preproc_lr(img_t_gray)
                img_t_gray = self.transform(img_t_gray)
                # img_t_gray = self.crop_preproc_lr(img_t_gray)
                img_t = (torch.nn.functional.interpolate(img_t.unsqueeze(0), size=(ori_h, ori_w), mode='bicubic',
                                                         align_corners=False).squeeze(0))
                img_t_gray = (torch.nn.functional.interpolate(img_t_gray.unsqueeze(0), size=(ori_h//8, ori_w//8), mode='bicubic',
                                                              align_corners=False).squeeze(0))

                # input images scaled to -1,1
                img_t = F.normalize(img_t, mean=[0.5], std=[0.5])

                img_ts.append(img_t)
                img_ts_gray.append(img_t_gray)

            gt_img = gt_img.unsqueeze(0).repeat(self.args.frames, 1, 1, 1)

            apply_trans = transforms_aug[random.getrandbits(3)]
            gt_img = [getattr(augment, apply_trans)(gt_im) for gt_im in gt_img]
            img_ts = [getattr(augment, apply_trans)(im) for im in img_ts]
            img_ts_gray = [getattr(augment, apply_trans)(im_gray) for im_gray in img_ts_gray]

            img_t_final = torch.stack(img_ts)
            gt_img_final = torch.stack(gt_img)
            img_t_final_gray = torch.stack(img_ts_gray)

            frame_difference_mask = []
            for frame_counter in range(self.args.frames):
                if frame_counter != 0:
                    current_frame = img_t_final_gray[frame_counter]
                    prev_frame = img_t_final_gray[frame_counter - 1]
                    frames_compute = torch.stack([current_frame, prev_frame])
                    frame_difference_mask_each = self.compute_frame_difference_mask(frames_compute)
                    frame_difference_mask.append(frame_difference_mask_each)

            frame_difference_mask = torch.stack(frame_difference_mask)

            example = {}
            example["neg_prompt"] = self.args.neg_prompt_vsd
            example["null_prompt"] = ""
            example["output_pixel_values"] = gt_img_final
            example["conditioning_pixel_values"] = img_t_final
            example["seq_name"] = img_t_dir.split('.png')[0].split('/')[-1]
            example["difference_mask"] = frame_difference_mask
            example["is_lsdir"] = 1


        elif "REDS" in gt_img_path:
            gt_frames_dir = os.path.dirname(gt_img_path)  # .../GT_frames
            parent_dir = os.path.dirname(gt_frames_dir)  # .../560818491_1
            img_t_dir = os.path.join(parent_dir, "LR_frames")

            gt_frame_name = os.path.basename(gt_img_path)  # 例如，frame_1.png
            gt_frame_num_str = gt_frame_name.split('.')[0].split('_')[-1]  # '1'
            gt_frame_num = int(gt_frame_num_str)  # 1

            lr_frame_nums = list(range(gt_frame_num, gt_frame_num + self.args.frames))

            img_ts = []
            img_ts_gt = []
            img_ts_gray = []

            for lr_num in lr_frame_nums:
                lr_frame_name = f"{lr_num:08d}.png"  # 例如，00000010.png
                lr_frame_path = os.path.join(img_t_dir, lr_frame_name)
                gt_frame_path = os.path.join(gt_frames_dir, lr_frame_name)

                # 读取图像
                img_t = Image.open(lr_frame_path)
                img_t_gt = Image.open(gt_frame_path)
                img_t_gray = img_t.convert("L")

                # 裁剪
                img_t = self.crop_preproc_lr(img_t)
                img_t_gt = self.crop_preproc(img_t_gt)
                img_t_gray = self.crop_preproc_lr(img_t_gray)
                
                # 转换为tensor
                img_t = self.transform(img_t)
                img_t_gt = self.transform(img_t_gt)
                img_t_gray = self.transform(img_t_gray)
                
                # # 裁剪
                # img_t = self.crop_preproc_lr(img_t)
                # img_t_gt = self.crop_preproc(img_t_gt)
                # img_t_gray = self.crop_preproc_lr(img_t_gray)
                
                # 上采样（如果需要）
                ori_h, ori_w = img_t_gt.size()[1:3]
                if self.need_upscale:
                    img_t = (torch.nn.functional.interpolate(img_t.unsqueeze(0), size=(ori_h, ori_w), mode='bicubic', align_corners=False).squeeze(0))
                img_t_gray = (torch.nn.functional.interpolate(img_t_gray.unsqueeze(0), size=(ori_h//8, ori_w//8), mode='bicubic', align_corners=False).squeeze(0))

                # 标准化到 [-1, 1]
                img_t = F.normalize(img_t, mean=[0.5], std=[0.5])
                img_t_gt = F.normalize(img_t_gt, mean=[0.5], std=[0.5])
                
                # 添加到列表
                img_ts.append(img_t)
                img_ts_gt.append(img_t_gt)
                img_ts_gray.append(img_t_gray)

            # 应用数据增强
            apply_trans = transforms_aug[random.getrandbits(3)]
            img_ts_gt = [getattr(augment, apply_trans)(im_gt) for im_gt in img_ts_gt]
            img_ts = [getattr(augment, apply_trans)(im) for im in img_ts]
            img_ts_gray = [getattr(augment, apply_trans)(im_gray) for im_gray in img_ts_gray]

            # 堆叠帧
            img_t_final = torch.stack(img_ts)
            img_t_final_gt = torch.stack(img_ts_gt)
            img_t_final_gray = torch.stack(img_ts_gray)

            frame_difference_mask = []
            for frame_counter in range(self.args.frames):
                if frame_counter != 0:
                    current_frame = img_t_final_gray[frame_counter]
                    prev_frame = img_t_final_gray[frame_counter - 1]
                    frames_compute = torch.stack([current_frame, prev_frame])
                    frame_difference_mask_each = self.compute_frame_difference_mask(frames_compute)
                    frame_difference_mask.append(frame_difference_mask_each)

            frame_difference_mask = torch.stack(frame_difference_mask)

            example = {}
            example["neg_prompt"] = self.args.neg_prompt_vsd
            example["null_prompt"] = ""
            example["output_pixel_values"] = img_t_final_gt
            example["conditioning_pixel_values"] = img_t_final
            example["seq_name"] = parent_dir.replace('/', '-')
            example["difference_mask"] = frame_difference_mask
            example["is_lsdir"] = 0

        elif "Pexel" in gt_img_path:
            gt_frames_dir = os.path.dirname(gt_img_path)  # .../GT_frames
            parent_dir = os.path.dirname(gt_frames_dir)  # .../560818491_1
            img_t_dir = os.path.join(parent_dir, "LR_frames")

            gt_frame_name = os.path.basename(gt_img_path)  # 例如，frame_1.png
            gt_frame_num_str = gt_frame_name.split('.')[0].split('_')[-1]  # '1'
            gt_frame_num = int(gt_frame_num_str)  # 1

            frame_interval = 10
            lr_frame_nums = list(range(gt_frame_num, gt_frame_num + frame_interval * self.args.frames, frame_interval))
            # lr_frame_nums = list(range(gt_frame_num, gt_frame_num + self.args.frames))

            img_ts = []
            img_ts_gt = []
            img_ts_gray = []

            for lr_num in lr_frame_nums:
                lr_frame_name = f"frame_{lr_num}.png"  # 例如，frame_1.png
                lr_frame_path = os.path.join(img_t_dir, lr_frame_name)
                gt_frame_path = os.path.join(gt_frames_dir, lr_frame_name)

                # 读取图像
                img_t = Image.open(lr_frame_path)
                img_t_gt = Image.open(gt_frame_path)
                img_t_gray = img_t.convert("L")

                # 裁剪
                img_t = self.crop_preproc_lr(img_t)
                img_t_gt = self.crop_preproc(img_t_gt)
                img_t_gray = self.crop_preproc_lr(img_t_gray)
                
                # 转换为tensor
                img_t = self.transform(img_t)
                img_t_gt = self.transform(img_t_gt)
                img_t_gray = self.transform(img_t_gray)
                
                # # 裁剪
                # img_t = self.crop_preproc_lr(img_t)
                # img_t_gt = self.crop_preproc(img_t_gt)
                # img_t_gray = self.crop_preproc_lr(img_t_gray)
                
                # 上采样（如果需要）
                ori_h, ori_w = img_t_gt.size()[1:3]
                if self.need_upscale:
                    img_t = (torch.nn.functional.interpolate(img_t.unsqueeze(0), size=(ori_h, ori_w), mode='bicubic', align_corners=False).squeeze(0))
                img_t_gray = (torch.nn.functional.interpolate(img_t_gray.unsqueeze(0), size=(ori_h//8, ori_w//8), mode='bicubic', align_corners=False).squeeze(0))

                # 标准化到 [-1, 1]
                img_t = F.normalize(img_t, mean=[0.5], std=[0.5])
                img_t_gt = F.normalize(img_t_gt, mean=[0.5], std=[0.5])
                
                # 添加到列表
                img_ts.append(img_t)
                img_ts_gt.append(img_t_gt)
                img_ts_gray.append(img_t_gray)

            # 应用数据增强
            apply_trans = transforms_aug[random.getrandbits(3)]
            img_ts_gt = [getattr(augment, apply_trans)(im_gt) for im_gt in img_ts_gt]
            img_ts = [getattr(augment, apply_trans)(im) for im in img_ts]
            img_ts_gray = [getattr(augment, apply_trans)(im_gray) for im_gray in img_ts_gray]

            # 堆叠帧
            img_t_final = torch.stack(img_ts)
            img_t_final_gt = torch.stack(img_ts_gt)
            img_t_final_gray = torch.stack(img_ts_gray)

            frame_difference_mask = []
            for frame_counter in range(self.args.frames):
                if frame_counter != 0:
                    current_frame = img_t_final_gray[frame_counter]
                    prev_frame = img_t_final_gray[frame_counter - 1]
                    frames_compute = torch.stack([current_frame, prev_frame])
                    frame_difference_mask_each = self.compute_frame_difference_mask(frames_compute)
                    frame_difference_mask.append(frame_difference_mask_each)

            frame_difference_mask = torch.stack(frame_difference_mask)

            example = {}
            example["neg_prompt"] = self.args.neg_prompt_vsd
            example["null_prompt"] = ""
            example["output_pixel_values"] = img_t_final_gt
            example["conditioning_pixel_values"] = img_t_final
            example["seq_name"] = parent_dir.replace('/', '-')
            example["difference_mask"] = frame_difference_mask
            example["is_lsdir"] = 0

        return example


class PairedSROnlineTxtDataset_Pexel_and_REDS_and_LSDIRshift(torch.utils.data.Dataset):
    def __init__(self, args=None):
        super().__init__()

        self.args = args
        self.degradation = RealESRGAN_degradation(args.deg_file_path, device='cpu')
        # self.crop_preproc = transforms.Compose([
        #     transforms.RandomCrop((args.resolution, args.resolution)),
        #     transforms.RandomHorizontalFlip(),
        # ])
        # self.crop_preproc = transforms.Compose([
        #     transforms.CenterCrop((args.resolution, args.resolution))
        # ])
        # self.crop_preproc_lr = transforms.Compose([
        #     transforms.CenterCrop((args.resolution, args.resolution))
        # ])
        self.crop_preproc_centercrop = transforms.Compose([
            transforms.CenterCrop((args.resolution, args.resolution))
        ])
        self.crop_preproc_lr_centercrop = transforms.Compose([
            transforms.CenterCrop((args.resolution, args.resolution))
        ])
        self.crop_preproc = transforms.Compose([
            transforms.Lambda(lambda img: resize_and_crop(img, (args.resolution, args.resolution)))
        ])
        self.crop_preproc_lr = transforms.Compose([
            transforms.Lambda(lambda img: resize_and_crop(img, (args.resolution, args.resolution)))
        ])
        self.transform = transforms.Compose([transforms.ToTensor()])
        self.to_pil = transforms.Compose([transforms.ToPILImage()])
        self.frame_number = args.frames
        self.need_upscale = args.need_upscale

        with open(args.dataset_txt_paths, 'r') as f:
            self.gt_list = [line.strip() for line in f.readlines()]

    def compute_frame_difference_mask(self, frames):
        ambi_matrix = frames.var(dim=0)
        threshold = ambi_matrix.mean().item()
        mask_id = torch.where(ambi_matrix >= threshold, ambi_matrix, torch.zeros_like(ambi_matrix))
        frame_mask = torch.where(mask_id == 0, mask_id, torch.ones_like(mask_id))
        return frame_mask

    def visualize_and_save_mask(self, reference_frame, mask, save_path, file_name):
        overlay = reference_frame * mask
        overlay_pil = self.to_pil(overlay)
        gt_pil = self.to_pil(reference_frame)
        os.makedirs(save_path, exist_ok=True)
        overlay_pil.save(os.path.join(save_path, file_name))
        gt_pil.save(os.path.join(save_path, file_name.split('.png')[0]+'_gt.png'))

    def __len__(self):
        return len(self.gt_list)

    def __getitem__(self, idx):
        gt_img_path = self.gt_list[idx]

        # if "LSDIR" in gt_img_path:
        #     gt_img = Image.open(self.gt_list[idx])
        #     gt_img = self.crop_preproc(gt_img)
        #     gt_img = self.transform(gt_img)
        #     # gt_img = self.crop_preproc(gt_img)
        #     # output images scaled to -1,1
        #     gt_img = F.normalize(gt_img, mean=[0.5], std=[0.5])

        #     ori_h, ori_w = gt_img.size()[1:3]

        #     img_t_dir = self.gt_list[idx].replace("gt", "lr")
        #     img_ts = []
        #     img_ts_gray = []

        #     if self.frame_number == 1:
        #         ids = [0]

        #     else:
        #         ids = random.sample(range(1, self.frame_number), k=(self.frame_number - 1))
        #         ids = [0, ] + ids

        #     for lr_index in ids:
        #         img_t = Image.open(
        #             '{}/{}_{}.png'.format(img_t_dir.split('.png')[0], img_t_dir.split('.png')[0].split('/')[-1],
        #                                   lr_index))
        #         img_t_gray = img_t.convert("L")
        #         img_t = self.crop_preproc_lr(img_t)
        #         img_t = self.transform(img_t)
        #         # img_t = self.crop_preproc_lr(img_t)
        #         img_t_gray = self.crop_preproc_lr(img_t_gray)
        #         img_t_gray = self.transform(img_t_gray)
        #         # img_t_gray = self.crop_preproc_lr(img_t_gray)
        #         img_t = (torch.nn.functional.interpolate(img_t.unsqueeze(0), size=(ori_h, ori_w), mode='bicubic',
        #                                                  align_corners=False).squeeze(0))
        #         img_t_gray = (torch.nn.functional.interpolate(img_t_gray.unsqueeze(0), size=(ori_h//8, ori_w//8), mode='bicubic',
        #                                                       align_corners=False).squeeze(0))

        #         # input images scaled to -1,1
        #         img_t = F.normalize(img_t, mean=[0.5], std=[0.5])

        #         img_ts.append(img_t)
        #         img_ts_gray.append(img_t_gray)

        #     gt_img = gt_img.unsqueeze(0).repeat(self.args.frames, 1, 1, 1)

        #     apply_trans = transforms_aug[random.getrandbits(3)]
        #     gt_img = [getattr(augment, apply_trans)(gt_im) for gt_im in gt_img]
        #     img_ts = [getattr(augment, apply_trans)(im) for im in img_ts]
        #     img_ts_gray = [getattr(augment, apply_trans)(im_gray) for im_gray in img_ts_gray]

        #     img_t_final = torch.stack(img_ts)
        #     gt_img_final = torch.stack(gt_img)
        #     img_t_final_gray = torch.stack(img_ts_gray)

        #     frame_difference_mask = []
        #     for frame_counter in range(self.args.frames):
        #         if frame_counter != 0:
        #             current_frame = img_t_final_gray[frame_counter]
        #             prev_frame = img_t_final_gray[frame_counter - 1]
        #             frames_compute = torch.stack([current_frame, prev_frame])
        #             frame_difference_mask_each = self.compute_frame_difference_mask(frames_compute)
        #             frame_difference_mask.append(frame_difference_mask_each)

        #     frame_difference_mask = torch.stack(frame_difference_mask)

        #     example = {}
        #     example["neg_prompt"] = self.args.neg_prompt_vsd
        #     example["null_prompt"] = ""
        #     example["output_pixel_values"] = gt_img_final
        #     example["conditioning_pixel_values"] = img_t_final
        #     example["seq_name"] = img_t_dir.split('.png')[0].split('/')[-1]
        #     example["difference_mask"] = frame_difference_mask
        #     example["is_lsdir"] = 1

        if "LSDIR_new" in gt_img_path:
            gt_frames_dir = os.path.dirname(gt_img_path)  # .../GT_frames
            parent_dir = os.path.dirname(gt_frames_dir)  # .../560818491_1
            img_t_dir = os.path.join(parent_dir, "LR_frames")

            gt_frame_name = os.path.basename(gt_img_path)  # 例如，frame_1.png
            gt_frame_num_str = gt_frame_name.split('.')[0].split('_')[-1]  # '1'
            gt_frame_num = int(gt_frame_num_str)  # 1

            # frame_interval = 10
            # lr_frame_nums = list(range(gt_frame_num, gt_frame_num + frame_interval * self.args.frames, frame_interval))
            lr_frame_nums = list(range(gt_frame_num, gt_frame_num + self.args.frames))

            img_ts = []
            img_ts_gt = []
            img_ts_gray = []

            for lr_num in lr_frame_nums:
                lr_frame_name = f"frame_{lr_num}.png"  # 例如，frame_1.png
                lr_frame_path = os.path.join(img_t_dir, lr_frame_name)
                gt_frame_path = os.path.join(gt_frames_dir, lr_frame_name)

                # 读取图像
                img_t = Image.open(lr_frame_path)
                img_t_gt = Image.open(gt_frame_path)
                img_t_gray = img_t.convert("L")

                # # 裁剪
                # img_t = self.crop_preproc_lr(img_t)
                # img_t_gt = self.crop_preproc(img_t_gt)
                # img_t_gray = self.crop_preproc_lr(img_t_gray)
                
                # 转换为tensor
                img_t = self.transform(img_t)
                img_t_gt = self.transform(img_t_gt)
                img_t_gray = self.transform(img_t_gray)
                
                # 裁剪
                img_t = self.crop_preproc_lr_centercrop(img_t)
                img_t_gt = self.crop_preproc_centercrop(img_t_gt)
                img_t_gray = self.crop_preproc_lr_centercrop(img_t_gray)
                
                # 上采样（如果需要）
                ori_h, ori_w = img_t_gt.size()[1:3]
                if self.need_upscale:
                    img_t = (torch.nn.functional.interpolate(img_t.unsqueeze(0), size=(ori_h, ori_w), mode='bicubic', align_corners=False).squeeze(0))
                img_t_gray = (torch.nn.functional.interpolate(img_t_gray.unsqueeze(0), size=(ori_h//8, ori_w//8), mode='bicubic', align_corners=False).squeeze(0))

                # 标准化到 [-1, 1]
                img_t = F.normalize(img_t, mean=[0.5], std=[0.5])
                img_t_gt = F.normalize(img_t_gt, mean=[0.5], std=[0.5])
                
                # 添加到列表
                img_ts.append(img_t)
                img_ts_gt.append(img_t_gt)
                img_ts_gray.append(img_t_gray)

            # 应用数据增强
            apply_trans = transforms_aug[random.getrandbits(3)]
            img_ts_gt = [getattr(augment, apply_trans)(im_gt) for im_gt in img_ts_gt]
            img_ts = [getattr(augment, apply_trans)(im) for im in img_ts]
            img_ts_gray = [getattr(augment, apply_trans)(im_gray) for im_gray in img_ts_gray]

            # 堆叠帧
            img_t_final = torch.stack(img_ts)
            img_t_final_gt = torch.stack(img_ts_gt)
            img_t_final_gray = torch.stack(img_ts_gray)

            frame_difference_mask = []
            for frame_counter in range(self.args.frames):
                if frame_counter != 0:
                    current_frame = img_t_final_gray[frame_counter]
                    prev_frame = img_t_final_gray[frame_counter - 1]
                    frames_compute = torch.stack([current_frame, prev_frame])
                    frame_difference_mask_each = self.compute_frame_difference_mask(frames_compute)
                    frame_difference_mask.append(frame_difference_mask_each)

            frame_difference_mask = torch.stack(frame_difference_mask)

            example = {}
            example["neg_prompt"] = self.args.neg_prompt_vsd
            example["null_prompt"] = ""
            example["output_pixel_values"] = img_t_final_gt
            example["conditioning_pixel_values"] = img_t_final
            example["seq_name"] = parent_dir.replace('/', '-')
            example["difference_mask"] = frame_difference_mask
            example["is_lsdir"] = 1


        elif "REDS" in gt_img_path:
            gt_frames_dir = os.path.dirname(gt_img_path)  # .../GT_frames
            parent_dir = os.path.dirname(gt_frames_dir)  # .../560818491_1
            img_t_dir = os.path.join(parent_dir, "LR_frames")

            gt_frame_name = os.path.basename(gt_img_path)  # 例如，frame_1.png
            gt_frame_num_str = gt_frame_name.split('.')[0].split('_')[-1]  # '1'
            gt_frame_num = int(gt_frame_num_str)  # 1

            lr_frame_nums = list(range(gt_frame_num, gt_frame_num + self.args.frames))

            img_ts = []
            img_ts_gt = []
            img_ts_gray = []

            for lr_num in lr_frame_nums:
                lr_frame_name = f"{lr_num:08d}.png"  # 例如，00000010.png
                lr_frame_path = os.path.join(img_t_dir, lr_frame_name)
                gt_frame_path = os.path.join(gt_frames_dir, lr_frame_name)

                # 读取图像
                img_t = Image.open(lr_frame_path)
                img_t_gt = Image.open(gt_frame_path)
                img_t_gray = img_t.convert("L")

                # 裁剪
                img_t = self.crop_preproc_lr(img_t)
                img_t_gt = self.crop_preproc(img_t_gt)
                img_t_gray = self.crop_preproc_lr(img_t_gray)
                
                # 转换为tensor
                img_t = self.transform(img_t)
                img_t_gt = self.transform(img_t_gt)
                img_t_gray = self.transform(img_t_gray)
                
                # # 裁剪
                # img_t = self.crop_preproc_lr(img_t)
                # img_t_gt = self.crop_preproc(img_t_gt)
                # img_t_gray = self.crop_preproc_lr(img_t_gray)
                
                # 上采样（如果需要）
                ori_h, ori_w = img_t_gt.size()[1:3]
                if self.need_upscale:
                    img_t = (torch.nn.functional.interpolate(img_t.unsqueeze(0), size=(ori_h, ori_w), mode='bicubic', align_corners=False).squeeze(0))
                img_t_gray = (torch.nn.functional.interpolate(img_t_gray.unsqueeze(0), size=(ori_h//8, ori_w//8), mode='bicubic', align_corners=False).squeeze(0))

                # 标准化到 [-1, 1]
                img_t = F.normalize(img_t, mean=[0.5], std=[0.5])
                img_t_gt = F.normalize(img_t_gt, mean=[0.5], std=[0.5])
                
                # 添加到列表
                img_ts.append(img_t)
                img_ts_gt.append(img_t_gt)
                img_ts_gray.append(img_t_gray)

            # 应用数据增强
            apply_trans = transforms_aug[random.getrandbits(3)]
            img_ts_gt = [getattr(augment, apply_trans)(im_gt) for im_gt in img_ts_gt]
            img_ts = [getattr(augment, apply_trans)(im) for im in img_ts]
            img_ts_gray = [getattr(augment, apply_trans)(im_gray) for im_gray in img_ts_gray]

            # 堆叠帧
            img_t_final = torch.stack(img_ts)
            img_t_final_gt = torch.stack(img_ts_gt)
            img_t_final_gray = torch.stack(img_ts_gray)

            frame_difference_mask = []
            for frame_counter in range(self.args.frames):
                if frame_counter != 0:
                    current_frame = img_t_final_gray[frame_counter]
                    prev_frame = img_t_final_gray[frame_counter - 1]
                    frames_compute = torch.stack([current_frame, prev_frame])
                    frame_difference_mask_each = self.compute_frame_difference_mask(frames_compute)
                    frame_difference_mask.append(frame_difference_mask_each)

            frame_difference_mask = torch.stack(frame_difference_mask)

            example = {}
            example["neg_prompt"] = self.args.neg_prompt_vsd
            example["null_prompt"] = ""
            example["output_pixel_values"] = img_t_final_gt
            example["conditioning_pixel_values"] = img_t_final
            example["seq_name"] = parent_dir.replace('/', '-')
            example["difference_mask"] = frame_difference_mask
            example["is_lsdir"] = 0

        elif "Pexel" in gt_img_path:
            gt_frames_dir = os.path.dirname(gt_img_path)  # .../GT_frames
            parent_dir = os.path.dirname(gt_frames_dir)  # .../560818491_1
            img_t_dir = os.path.join(parent_dir, "LR_frames")

            gt_frame_name = os.path.basename(gt_img_path)  # 例如，frame_1.png
            gt_frame_num_str = gt_frame_name.split('.')[0].split('_')[-1]  # '1'
            gt_frame_num = int(gt_frame_num_str)  # 1

            frame_interval = 10
            lr_frame_nums = list(range(gt_frame_num, gt_frame_num + frame_interval * self.args.frames, frame_interval))
            # lr_frame_nums = list(range(gt_frame_num, gt_frame_num + self.args.frames))

            img_ts = []
            img_ts_gt = []
            img_ts_gray = []

            for lr_num in lr_frame_nums:
                lr_frame_name = f"frame_{lr_num}.png"  # 例如，frame_1.png
                lr_frame_path = os.path.join(img_t_dir, lr_frame_name)
                gt_frame_path = os.path.join(gt_frames_dir, lr_frame_name)

                # 读取图像
                img_t = Image.open(lr_frame_path)
                img_t_gt = Image.open(gt_frame_path)
                img_t_gray = img_t.convert("L")

                # 裁剪
                img_t = self.crop_preproc_lr(img_t)
                img_t_gt = self.crop_preproc(img_t_gt)
                img_t_gray = self.crop_preproc_lr(img_t_gray)
                
                # 转换为tensor
                img_t = self.transform(img_t)
                img_t_gt = self.transform(img_t_gt)
                img_t_gray = self.transform(img_t_gray)
                
                # # 裁剪
                # img_t = self.crop_preproc_lr(img_t)
                # img_t_gt = self.crop_preproc(img_t_gt)
                # img_t_gray = self.crop_preproc_lr(img_t_gray)
                
                # 上采样（如果需要）
                ori_h, ori_w = img_t_gt.size()[1:3]
                if self.need_upscale:
                    img_t = (torch.nn.functional.interpolate(img_t.unsqueeze(0), size=(ori_h, ori_w), mode='bicubic', align_corners=False).squeeze(0))
                img_t_gray = (torch.nn.functional.interpolate(img_t_gray.unsqueeze(0), size=(ori_h//8, ori_w//8), mode='bicubic', align_corners=False).squeeze(0))

                # 标准化到 [-1, 1]
                img_t = F.normalize(img_t, mean=[0.5], std=[0.5])
                img_t_gt = F.normalize(img_t_gt, mean=[0.5], std=[0.5])
                
                # 添加到列表
                img_ts.append(img_t)
                img_ts_gt.append(img_t_gt)
                img_ts_gray.append(img_t_gray)

            # 应用数据增强
            apply_trans = transforms_aug[random.getrandbits(3)]
            img_ts_gt = [getattr(augment, apply_trans)(im_gt) for im_gt in img_ts_gt]
            img_ts = [getattr(augment, apply_trans)(im) for im in img_ts]
            img_ts_gray = [getattr(augment, apply_trans)(im_gray) for im_gray in img_ts_gray]

            # 堆叠帧
            img_t_final = torch.stack(img_ts)
            img_t_final_gt = torch.stack(img_ts_gt)
            img_t_final_gray = torch.stack(img_ts_gray)

            frame_difference_mask = []
            for frame_counter in range(self.args.frames):
                if frame_counter != 0:
                    current_frame = img_t_final_gray[frame_counter]
                    prev_frame = img_t_final_gray[frame_counter - 1]
                    frames_compute = torch.stack([current_frame, prev_frame])
                    frame_difference_mask_each = self.compute_frame_difference_mask(frames_compute)
                    frame_difference_mask.append(frame_difference_mask_each)

            frame_difference_mask = torch.stack(frame_difference_mask)

            example = {}
            example["neg_prompt"] = self.args.neg_prompt_vsd
            example["null_prompt"] = ""
            example["output_pixel_values"] = img_t_final_gt
            example["conditioning_pixel_values"] = img_t_final
            example["seq_name"] = parent_dir.replace('/', '-')
            example["difference_mask"] = frame_difference_mask
            example["is_lsdir"] = 0

        return example





class PairedSROnlineTxtDataset_multiframe_openvid(torch.utils.data.Dataset):
    def __init__(self, args=None):
        super().__init__()

        self.args = args
        self.degradation = RealESRGAN_degradation(args.deg_file_path, device='cpu')
        # self.crop_preproc = transforms.Compose([
        #     transforms.RandomCrop((args.resolution, args.resolution)),
        #     transforms.RandomHorizontalFlip(),
        # ])
        self.crop_preproc = transforms.Compose([
            transforms.CenterCrop((args.resolution, args.resolution))
        ])
        self.crop_preproc_lr = transforms.Compose([
            transforms.CenterCrop((args.resolution, args.resolution))
        ])
        self.transform = transforms.Compose([transforms.ToTensor()])
        self.frame_number = args.frames
        self.need_upscale = args.need_upscale

        with open(args.dataset_txt_paths, 'r') as f:
            self.gt_list = [line.strip() for line in f.readlines()]

    def __len__(self):
        return len(self.gt_list)

    def __getitem__(self, idx):
        gt_img_path = self.gt_list[idx]
        gt_img = Image.open(gt_img_path)
        gt_img = self.transform(gt_img)
        gt_img = self.crop_preproc(gt_img)
        # output images scaled to -1,1
        gt_img = F.normalize(gt_img, mean=[0.5], std=[0.5])

        ori_h, ori_w = gt_img.size()[1:3]

        gt_frames_dir = os.path.dirname(gt_img_path)  # .../GT_frames
        parent_dir = os.path.dirname(gt_frames_dir)  # .../xxx
        img_t_dir = os.path.join(parent_dir, "LR_frames_serious_sisr")

        gt_frame_name = os.path.basename(gt_img_path)  # e.g., frame_0002.png
        gt_frame_num_str = gt_frame_name.split('.')[0].split('_')[-1]  # '0002'
        gt_frame_num = int(gt_frame_num_str)  # 2

        lr_frame_nums = list(range(gt_frame_num, gt_frame_num + self.args.frames))
        random.shuffle(lr_frame_nums)

        img_ts = []

        for lr_num in lr_frame_nums:
            lr_frame_name = f"frame_{lr_num:04d}.png"  # e.g., frame_0002.png
            lr_frame_path = os.path.join(img_t_dir, lr_frame_name)
            img_t = Image.open(lr_frame_path)
            img_t = self.transform(img_t)
            img_t = self.crop_preproc_lr(img_t)
            if self.need_upscale:
                img_t = (torch.nn.functional.interpolate(img_t.unsqueeze(0), size=(ori_h, ori_w), mode='bicubic', align_corners=False).squeeze(0))
            # input images scaled to -1,1
            img_t = F.normalize(img_t, mean=[0.5], std=[0.5])

            img_ts.append(img_t)

        apply_trans = transforms_aug[random.getrandbits(3)]
        gt_img = getattr(augment, apply_trans)(gt_img)
        img_ts = [getattr(augment, apply_trans)(im) for im in img_ts]

        img_t_final = torch.stack(img_ts)

        example = {}
        example["neg_prompt"] = self.args.neg_prompt_vsd
        example["null_prompt"] = ""
        example["output_pixel_values"] = gt_img
        example["conditioning_pixel_values"] = img_t_final
        example["seq_name"] = parent_dir.split('/')[-1]

        return example



def get_crop(img, r1, r2, c1, c2):
    im_raw = img[:, r1:r2, c1:c2]
    return im_raw

center_crop_lr = transforms.CenterCrop(60)
center_crop_gt = transforms.CenterCrop(160)

class ManualDatasets_validation(torch.utils.data.Dataset):
    """ Real-world burst super-resolution dataset. """

    def __init__(self, root, burst_size=14, center_crop=False, random_flip=False, sift_lr=False,
                 split='train'):
        """
        args:
            root : path of the root directory
            burst_size : Burst size. Maximum allowed burst size is 14.
            crop_sz: Size of the extracted crop. Maximum allowed crop size is 80
            center_crop: Whether to extract a random crop, or a centered crop.
            random_flip: Whether to apply random horizontal and vertical flip
            split: Can be 'train' or 'val'
        """
        assert burst_size <= 14, 'burst_sz must be less than or equal to 14'
        # assert crop_sz <= 80, 'crop_sz must be less than or equal to 80'
        assert split in ['train', 'val']
        super().__init__()

        self.transform = transforms.Compose([transforms.ToTensor()])

        self.burst_size = burst_size
        self.split = split
        self.center_crop = center_crop
        self.random_flip = random_flip

        self.sift_lr = sift_lr

        self.root = root
        # split trainset and testset in one dir
        if self.split == 'val':
            root = root + '/test'
        else:
            root = root + '/train'

        self.hrdir = root + '/' + 'HR'
        self.lrdir = root + '/' + 'LR_aligned'
        print(self.lrdir)

        self.substract_black_level = True
        self.white_balance = False

        self.burst_list = self._get_burst_list()
        self.data_length = len(self.burst_list)
        # self.data_length = 20

    def _get_burst_list(self):
        burst_list = sorted(os.listdir(self.lrdir))
        # print(burst_list)
        return burst_list

    def _get_raw_image(self, burst_id, im_id):
        # Manual_dataset/train/LR/109_28/109_MFSR_Sony_0028_x4_00.png
        burst_number = self.burst_list[burst_id].split('_')[0]
        burst_number2 = int(self.burst_list[burst_id].split('_')[-1])

        path = '{}/{}/{}_MFSR_Sony_{:04d}_x1_{:02d}.png'.format(self.lrdir, self.burst_list[burst_id], burst_number,
                                                                burst_number2, im_id)

        #         path = '{}/{}/{}_MFSR_Sony_{:04d}_x4_{:02d}.png'.format(self.lrdir, self.burst_list[burst_id], burst_number,
        #                                                                 burst_number2, im_id)

        image = Image.open(path)  # RGB,W, H, C
        image = center_crop_lr(image)
        image = self.transform(image)
        # print(image.shape)
        # image = cv2.imread(path, cv2.COLOR_BGR2RGB)

        # image = torch.nn.functional.interpolate(image.unsqueeze(0), size=(640, 640), mode='bicubic', align_corners=False).squeeze(0)
        # # input images scaled to -1,1
        # image = F.normalize(image, mean=[0.5], std=[0.5])
        return image

    def _get_gt_image(self, burst_id):
        # 000_MFSR_Sony_0001_x4.png
        burst_number = self.burst_list[burst_id].split('_')[0]
        burst_nmber2 = int(self.burst_list[burst_id].split('_')[-1])
        #         path = '{}/{}_MFSR_Sony_{:04d}_x4.png'.format(self.hrdir, burst_number, burst_nmber2)

        path = '{}/{}/{}_MFSR_Sony_{:04d}_x4.png'.format(self.hrdir, self.burst_list[burst_id], burst_number,
                                                         burst_nmber2)

        image = Image.open(path)  # RGB,W, H, C
        image = center_crop_gt(image)
        image = self.transform(image)
        # image = F.normalize(image, mean=[0.5], std=[0.5])
        return image

    def get_burst(self, burst_id, im_ids):
        frames = [self._get_raw_image(burst_id, i) for i in im_ids]
        # pic = self._get_raw_image(burst_id, 0)
        gt = self._get_gt_image(burst_id)

        return frames, gt

    def _sample_images(self):
        burst_size = self.burst_size
        ids = random.sample(range(1, burst_size), k=self.burst_size - 1)
        ids = [0, ] + ids
        return ids

    def get_burst_info(self, burst_id):
        burst_info = {'burst_size': 14, 'burst_name': self.burst_list[burst_id]}
        return burst_info

    def __len__(self):
        return self.data_length

    def __getitem__(self, index):
        # Sample the images in the burst, in case a burst_size < 14 is used.
        im_ids = self._sample_images()

        frames, gt = self.get_burst(index, im_ids)
        info = self.get_burst_info(index)

        # # Extract crop if needed
        # print("Before cropping: frames.shape: {}, gt.shape: {}".format(frames[0].shape, gt.shape))
        # if frames[0].shape[-1] != self.crop_sz:
        #     r1 = (frames[0].shape[-2] - self.crop_sz) // 2
        #     c1 = (frames[0].shape[-1] - self.crop_sz) // 2
        #     r2 = r1 + self.crop_sz
        #     c2 = c1 + self.crop_sz
        #
        #     print("Cropping: start: {}, end: {}".format(r1, r2))
        #
        #     frames = [get_crop(im, r1, r2, c1, c2) for im in frames]
        #     gt = get_crop(gt, r1, r2, c1, c2)

        burst = torch.stack(frames, dim=0)
        burst = burst.float()
        frame_gt = gt.float()

        data = {}
        data['LR'] = burst
        data['HR'] = frame_gt
        data['burst_name'] = info['burst_name']

        return data


class PairedSROnlineTxtDataset_multiframe_puzzle(torch.utils.data.Dataset):
    def __init__(self, args=None):
        super().__init__()

        self.args = args
        self.degradation = RealESRGAN_degradation(args.deg_file_path, device='cpu')
        # self.crop_preproc = transforms.Compose([
        #     transforms.RandomCrop((args.resolution, args.resolution)),
        #     transforms.RandomHorizontalFlip(),
        # ])
        self.crop_preproc = transforms.Compose([
            transforms.CenterCrop((args.resolution, args.resolution))
        ])
        self.crop_preproc_lr = transforms.Compose([
            transforms.CenterCrop((args.resolution // 4, args.resolution // 4))
        ])
        self.transform = transforms.Compose([transforms.ToTensor()])
        self.frame_number = args.frames
        self.need_upscale = args.need_upscale

        with open(args.dataset_txt_paths, 'r') as f:
            self.gt_list = [line.strip() for line in f.readlines()]

    def __len__(self):
        return len(self.gt_list)

    def __getitem__(self, idx):
        gt_img = Image.open(self.gt_list[idx])
        gt_img = self.transform(gt_img)
        gt_img = self.crop_preproc(gt_img)
        # output images scaled to -1,1
        gt_img = F.normalize(gt_img, mean=[0.5], std=[0.5])

        ori_h, ori_w = gt_img.size()[1:3]

        img_t_dir = self.gt_list[idx].replace("gt", "lr")
        img_ts = []

        frame_img_collect = []

        ids = random.sample(range(1, self.frame_number), k=(self.frame_number - 1))
        ids = [0, ] + ids

        for lr_index in ids:
            img_t = Image.open(
                '{}/{}_{}.png'.format(img_t_dir.split('.png')[0], img_t_dir.split('.png')[0].split('/')[-1], lr_index))
            frame_img = self.transform(img_t.convert('L')).unsqueeze(0)
            img_t = self.transform(img_t)

            img_t = self.crop_preproc_lr(img_t)
            frame_img = self.crop_preproc_lr(frame_img)

            if self.need_upscale:
                img_t = (torch.nn.functional.interpolate(img_t.unsqueeze(0), size=(ori_h, ori_w), mode='bicubic',
                                                         align_corners=False).squeeze(0))
                frame_img = (torch.nn.functional.interpolate(frame_img, size=(ori_h, ori_w), mode='bicubic',
                                                         align_corners=False))
            # input images scaled to -1,1
            img_t = F.normalize(img_t, mean=[0.5], std=[0.5])

            img_ts.append(img_t)
            frame_img_collect.append(frame_img.squeeze(0))

        frame_collect = torch.stack(frame_img_collect, dim=0)
        ambi_matrix = frame_collect.var(0)

        threshold = ambi_matrix.median().item()
        mask_id = torch.where(ambi_matrix >= threshold, ambi_matrix, torch.zeros_like(ambi_matrix))
        frame_mask = torch.where(mask_id == 0, mask_id, torch.ones_like(mask_id))

        # transform_topil = transforms.ToPILImage()
        # try_image_save = frame_mask * img_ts[0]
        # try_image_save = transform_topil(try_image_save)
        # try_image_save.save('{}_try.png'.format(img_t_dir.split('.png')[0].split('/')[-1]))
        #
        # try_gt_image = transform_topil(gt_img)
        # try_gt_image.save('{}_gt.png'.format(img_t_dir.split('.png')[0].split('/')[-1]))
        #
        # try_ori_image = img_ts[0]
        # try_ori_image = transform_topil(try_ori_image)
        # try_ori_image.save('{}.png'.format(img_t_dir.split('.png')[0].split('/')[-1]))

        apply_trans = transforms_aug[random.getrandbits(3)]
        gt_img = getattr(augment, apply_trans)(gt_img)
        img_ts = [getattr(augment, apply_trans)(im) for im in img_ts]
        frame_mask = getattr(augment, apply_trans)(frame_mask)

        img_t_final = torch.stack(img_ts)

        # gt_img = Image.open(self.gt_list[idx]).convert('RGB')
        # gt_img = self.crop_preproc(gt_img)
        #
        # output_t, img_t = self.degradation.degrade_process(np.asarray(gt_img)/255., resize_bak=True)
        # output_t, img_t = output_t.squeeze(0), img_t.squeeze(0)

        example = {}
        example["neg_prompt"] = self.args.neg_prompt_vsd
        example["null_prompt"] = ""
        example["output_pixel_values"] = gt_img
        example["conditioning_pixel_values"] = img_t_final
        example["frame_mask"] = frame_mask

        return example

import torch.nn as nn

class GWLoss(nn.Module):
    def __init__(self, rgb_range=1.):
        super(GWLoss, self).__init__()
        self.rgb_range=rgb_range
    def forward(self, x1, x2):
        x1=torch.clamp(x1,min=0.0,max=1.0)
        x2=torch.clamp(x2,min=0.0,max=1.0)

        sobel_x = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
        sobel_y = [[-1, -2, -1], [0, 0, 0], [1, 2, 1]]
        b, c, w, h = x1.shape
        sobel_x = torch.FloatTensor(sobel_x).expand(c, 1, 3, 3)
        sobel_y = torch.FloatTensor(sobel_y).expand(c, 1, 3, 3)
        sobel_x = sobel_x.type_as(x1)
        sobel_y = sobel_y.type_as(x1)
        weight_x = nn.Parameter(data=sobel_x, requires_grad=False)
        weight_y = nn.Parameter(data=sobel_y, requires_grad=False)
        Ix1 = nn.functional.conv2d(x1, weight_x, stride=1, padding=1, groups=c)
        Ix2 = nn.functional.conv2d(x2, weight_x, stride=1, padding=1, groups=c)
        Iy1 = nn.functional.conv2d(x1, weight_y, stride=1, padding=1, groups=c)
        Iy2 = nn.functional.conv2d(x2, weight_y, stride=1, padding=1, groups=c)
        dx = torch.abs(Ix1 - Ix2)
        dy = torch.abs(Iy1 - Iy2)
        #     loss = torch.exp(2*(dx + dy)) * torch.abs(x1 - x2)
        loss = (1 + 4 * dx) * (1 + 4 * dy) * torch.abs(x1 - x2)

        return torch.mean(loss)
