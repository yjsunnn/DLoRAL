import os
os.environ['TRANSFORMERS_OFFLINE']='0'
import requests
import sys
import copy
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, CLIPTextModel
from diffusers import DDPMScheduler
# from models.autoencoder_kl import AutoencoderKL
# from models.unet_2d_condition import UNet2DConditionModel
from diffusers import AutoencoderKL
from diffusers import UNet2DConditionModel
from diffusers.utils.peft_utils import set_weights_and_activate_adapters
from peft import LoraConfig
p = "src/"
sys.path.append(p)

import torch.nn as nn
from diffusers.utils.import_utils import is_xformers_available
import torch.nn.functional as F

from types import SimpleNamespace
sys.path.append(os.getcwd())
from src.my_utils.vaehook import VAEHook, perfcount
import random
from src.cross_frame_retrieval.cfr_main import CFR_model
import torchvision.transforms as vision_transforms

def make_1step_sched(args):
    noise_scheduler_1step = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    noise_scheduler_1step.set_timesteps(1, device="cuda")
    noise_scheduler_1step.alphas_cumprod = noise_scheduler_1step.alphas_cumprod.cuda()
    return noise_scheduler_1step

import glob
def find_filepath(directory, filename):
    matches = glob.glob(f"{directory}/**/{filename}", recursive=True)
    return matches[0] if matches else None

import yaml
def read_yaml(file_path):
    with open(file_path, 'r') as file:
        data = yaml.safe_load(file)
    return data

def initialize_vae(rank, return_lora_module_names=False, pretrained_model_name_or_path=None):
    vae = AutoencoderKL.from_pretrained(pretrained_model_name_or_path, subfolder="vae")
    vae.requires_grad_(False)
    vae.train()

    l_target_modules_encoder, l_target_modules_decoder, l_modules_others = [], [], []
    l_grep = ["conv1","conv2","conv_in", "conv_shortcut",
        "conv", "conv_out", "to_k", "to_q", "to_v", "to_out.0",
    ]
    for n, p in vae.named_parameters():
        check_flag = 0
        if "bias" in n or "norm" in n:
            check_flag=1
            continue
        for pattern in l_grep:
            if pattern in n and ("encoder" in n):
                l_target_modules_encoder.append(n.replace(".weight",""))
                break
            elif ('quant_conv' in n) and ('post_quant_conv' not in n):
                l_target_modules_encoder.append(n.replace(".weight",""))
                break

    lora_conf_encoder = LoraConfig(r=rank, init_lora_weights="gaussian",target_modules=l_target_modules_encoder)
    vae.add_adapter(lora_conf_encoder, adapter_name="default_encoder")
    if return_lora_module_names:
        return vae, l_target_modules_encoder
    else:
        return vae


def initialize_unet(rank_quality, rank_consistency, return_lora_module_names=False, pretrained_model_name_or_path=None):
    unet = UNet2DConditionModel.from_pretrained(pretrained_model_name_or_path, subfolder="unet")
    # reformulate the conv_in in unet
    conv_in = nn.Conv2d(8, 320, kernel_size=3, padding=1)
    conv_in.weight.data[:, 0:4, ...] = unet.conv_in.weight.data
    conv_in.weight.data[:, 4:8, ...] = unet.conv_in.weight.data

    unet.conv_in = conv_in
    unet.requires_grad_(False)
    unet.train()

    l_target_modules_encoder_quality, l_target_modules_decoder_quality, l_modules_others_quality = [], [], []
    l_target_modules_encoder_consistency, l_target_modules_decoder_consistency, l_modules_others_consistency = [], [], []
    l_grep = ["to_k", "to_q", "to_v", "to_out.0", "conv", "conv1", "conv2", "conv_in", "conv_shortcut", "conv_out", "proj_out", "proj_in", "ff.net.2", "ff.net.0.proj"]

    for n, p in unet.named_parameters():
        check_flag = 0
        if "bias" in n or "norm" in n:
            continue
        for pattern in l_grep:
            if pattern in n and ("down_blocks" in n or "conv_in" in n):
                l_target_modules_encoder_quality.append(n.replace(".weight",""))
                l_target_modules_encoder_consistency.append(n.replace(".weight",""))
                break
            elif pattern in n and ("up_blocks" in n or "conv_out" in n):
                l_target_modules_decoder_quality.append(n.replace(".weight",""))
                l_target_modules_decoder_consistency.append(n.replace(".weight",""))
                break
            elif pattern in n:
                l_modules_others_quality.append(n.replace(".weight",""))
                l_modules_others_consistency.append(n.replace(".weight",""))
                break

    lora_conf_encoder_quality = LoraConfig(r=rank_quality, init_lora_weights="gaussian",target_modules=l_target_modules_encoder_quality)
    lora_conf_decoder_quality = LoraConfig(r=rank_quality, init_lora_weights="gaussian",target_modules=l_target_modules_decoder_quality)
    lora_conf_others_quality = LoraConfig(r=rank_quality, init_lora_weights="gaussian",target_modules=l_modules_others_quality)
    lora_conf_encoder_consistency = LoraConfig(r=rank_consistency, init_lora_weights="gaussian",target_modules=l_target_modules_encoder_consistency)
    lora_conf_decoder_consistency = LoraConfig(r=rank_consistency, init_lora_weights="gaussian",target_modules=l_target_modules_decoder_consistency)
    lora_conf_others_consistency = LoraConfig(r=rank_consistency, init_lora_weights="gaussian",target_modules=l_modules_others_consistency)

    unet.add_adapter(lora_conf_encoder_quality, adapter_name="default_encoder_quality")
    unet.add_adapter(lora_conf_decoder_quality, adapter_name="default_decoder_quality")
    unet.add_adapter(lora_conf_others_quality, adapter_name="default_others_quality")
    unet.add_adapter(lora_conf_encoder_consistency, adapter_name="default_encoder_consistency")
    unet.add_adapter(lora_conf_decoder_consistency, adapter_name="default_decoder_consistency")
    unet.add_adapter(lora_conf_others_consistency, adapter_name="default_others_consistency")

    if return_lora_module_names:
        return unet, \
               l_target_modules_encoder_quality, l_target_modules_decoder_quality, l_modules_others_quality, \
               l_target_modules_encoder_consistency, l_target_modules_decoder_consistency, l_modules_others_consistency
    else:
        return unet



def initialize_unet_regularizer(rank, return_lora_module_names=False, pretrained_model_name_or_path=None):
    unet = UNet2DConditionModel.from_pretrained(pretrained_model_name_or_path, subfolder="unet")
    unet.requires_grad_(False)
    unet.train()

    l_target_modules_encoder, l_target_modules_decoder, l_modules_others = [], [], []
    l_grep = ["to_k", "to_q", "to_v", "to_out.0", "conv", "conv1", "conv2", "conv_in", "conv_shortcut", "conv_out", "proj_out", "proj_in", "ff.net.2", "ff.net.0.proj"]
    for n, p in unet.named_parameters():
        check_flag = 0
        if "bias" in n or "norm" in n:
            continue
        for pattern in l_grep:
            if pattern in n and ("down_blocks" in n or "conv_in" in n):
                l_target_modules_encoder.append(n.replace(".weight",""))
                break
            elif pattern in n and ("up_blocks" in n or "conv_out" in n):
                l_target_modules_decoder.append(n.replace(".weight",""))
                break
            elif pattern in n:
                l_modules_others.append(n.replace(".weight",""))
                break

    lora_conf_encoder = LoraConfig(r=rank, init_lora_weights="gaussian",target_modules=l_target_modules_encoder)
    lora_conf_decoder = LoraConfig(r=rank, init_lora_weights="gaussian",target_modules=l_target_modules_decoder)
    lora_conf_others = LoraConfig(r=rank, init_lora_weights="gaussian",target_modules=l_modules_others)
    unet.add_adapter(lora_conf_encoder, adapter_name="default_encoder")
    unet.add_adapter(lora_conf_decoder, adapter_name="default_decoder")
    unet.add_adapter(lora_conf_others, adapter_name="default_others")

    if return_lora_module_names:
        return unet, l_target_modules_encoder, l_target_modules_decoder, l_modules_others
    else:
        return unet


class CSDLoss(torch.nn.Module):
    def __init__(self, args, accelerator):
        super().__init__()
        args.pretrained_model_path_csd = '/home/notebook/data/group/syj/OSEDiff/OSEDiff/preset_models/stable-diffusion-2-1-base'
        self.tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model_path_csd, subfolder="tokenizer")
        self.sched = DDPMScheduler.from_pretrained(args.pretrained_model_path_csd, subfolder="scheduler")
        self.args = args

        weight_dtype = torch.float32
        if accelerator.mixed_precision == "fp16":
            weight_dtype = torch.float16
        elif accelerator.mixed_precision == "bf16":
            weight_dtype = torch.bfloat16

        self.unet_fix = UNet2DConditionModel.from_pretrained(args.pretrained_model_path_csd, subfolder="unet")

        if args.enable_xformers_memory_efficient_attention:
            if is_xformers_available():
                self.unet_fix.enable_xformers_memory_efficient_attention()
            else:
                raise ValueError("xformers is not available, please install it by running `pip install xformers`")

        self.unet_fix.to(accelerator.device, dtype=weight_dtype)

        self.unet_fix.requires_grad_(False)
        self.unet_fix.eval()

    def forward_latent(self, model, latents, timestep, prompt_embeds):

        noise_pred = model(
            latents,
            timestep=timestep,
            encoder_hidden_states=prompt_embeds,
        ).sample

        return noise_pred

    def eps_to_mu(self, scheduler, model_output, sample, timesteps):
        alphas_cumprod = scheduler.alphas_cumprod.to(device=sample.device, dtype=sample.dtype)
        alpha_prod_t = alphas_cumprod[timesteps]
        while len(alpha_prod_t.shape) < len(sample.shape):
            alpha_prod_t = alpha_prod_t.unsqueeze(-1)
        beta_prod_t = 1 - alpha_prod_t
        pred_original_sample = (sample - beta_prod_t ** (0.5) * model_output) / alpha_prod_t ** (0.5)
        return pred_original_sample

    def cal_csd(
            self,
            latents,
            prompt_embeds,
            negative_prompt_embeds,
            args,
            noise_csd,
    ):
        bsz = latents.shape[0]
        min_dm_step = int(self.sched.config.num_train_timesteps * args.min_dm_step_ratio)
        max_dm_step = int(self.sched.config.num_train_timesteps * args.max_dm_step_ratio)

        timestep = torch.randint(min_dm_step, max_dm_step, (bsz,), device=latents.device).long()
        # noise = torch.randn_like(latents)
        noise = noise_csd
        noisy_latents = self.sched.add_noise(latents, noise, timestep)

        with torch.no_grad():
            noisy_latents_input = torch.cat([noisy_latents] * 2)
            timestep_input = torch.cat([timestep] * 2)
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
            noise_pred = self.forward_latent(
                self.unet_fix,
                latents=noisy_latents_input.to(dtype=torch.float16),
                timestep=timestep_input,
                prompt_embeds=prompt_embeds.to(dtype=torch.float16),
            )
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + args.cfg_csd * (noise_pred_text - noise_pred_uncond)
            noise_pred.to(dtype=torch.float32)
            noise_pred_uncond.to(dtype=torch.float32)

            pred_real_latents = self.eps_to_mu(self.sched, noise_pred, noisy_latents, timestep)
            pred_fake_latents = self.eps_to_mu(self.sched, noise_pred_uncond, noisy_latents, timestep)

        weighting_factor = torch.abs(latents - pred_real_latents).mean(dim=[1, 2, 3], keepdim=True)

        grad = (pred_fake_latents - pred_real_latents) / weighting_factor
        loss = F.mse_loss(latents, self.stopgrad(latents - grad))

        return loss

    def stopgrad(self, x):
        return x.detach()


class Generator(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        pretrained_name = setattr(args, 'pretrained_name', getattr(args, 'pretrained_name', None))
        pretrained_path = setattr(args, 'pretrained_path', getattr(args, 'pretrained_path', None))

        args.pretrained_model_name_or_path = '/home/notebook/data/group/syj/OSEDiff/OSEDiff/preset_models/stable-diffusion-2-1-base'

        self.tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer")
        self.text_encoder = CLIPTextModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="text_encoder").cuda()
        self.sched = make_1step_sched(args)
        self.args = args

        self.cfr_main_net = CFR_model(mid_channels=64, num_blocks=7, is_low_res_input=False,
                                                          spynet_pretrained='https://download.openmmlab.com/mmediting/restorers/''basicvsr/spynet_20210409-c6c1bd09.pth')

        ######################################################
        if args.resume_ckpt is None:
            self.unet, lora_unet_modules_encoder_quality, lora_unet_modules_decoder_quality, lora_unet_others_quality, \
            lora_unet_modules_encoder_consistency, lora_unet_modules_decoder_consistency, lora_unet_others_consistency, = \
                initialize_unet(rank_quality=args.lora_rank_unet_quality, rank_consistency=args.lora_rank_unet_consistency,
                                pretrained_model_name_or_path=args.pretrained_model_name_or_path,
                                return_lora_module_names=True)

            self.lora_rank_unet_quality = args.lora_rank_unet_quality
            self.lora_rank_unet_consistency = args.lora_rank_unet_consistency
            self.lora_unet_modules_encoder_quality, self.lora_unet_modules_decoder_quality, self.lora_unet_others_quality, \
            self.lora_unet_modules_encoder_consistency, self.lora_unet_modules_decoder_consistency, self.lora_unet_others_consistency = \
                lora_unet_modules_encoder_quality, lora_unet_modules_decoder_quality, lora_unet_others_quality, \
                lora_unet_modules_encoder_consistency, lora_unet_modules_decoder_consistency, lora_unet_others_consistency
        else:
            print(f'====> resume from {args.resume_ckpt}')
            stage1_yaml = find_filepath(args.resume_ckpt.split('/checkpoints')[0], 'hparams.yml')
            stage1_args = read_yaml(stage1_yaml)
            stage1_args = SimpleNamespace(**stage1_args)
            self.unet = UNet2DConditionModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="unet")
            # reformulate the conv_in in unet
            conv_in = nn.Conv2d(8, 320, kernel_size=3, padding=1)
            conv_in.weight.data[:, 0:4, ...] = self.unet.conv_in.weight.data
            conv_in.weight.data[:, 4:8, ...] = self.unet.conv_in.weight.data

            self.unet.conv_in = conv_in
            self.lora_rank_unet_quality = stage1_args.lora_rank_unet_quality
            self.lora_rank_unet_consistency = stage1_args.lora_rank_unet_consistency
            osediff = torch.load(args.resume_ckpt)
            self.load_ckpt_from_state_dict(osediff)
        ######################################################

        self.vae_fix = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae")
        self.vae_fix.to('cuda')

        # unet.enable_xformers_memory_efficient_attention()
        self.unet.to("cuda")
        self.vae_fix.to("cuda")
        self.cfr_main_net.to("cuda")
        self.timesteps = torch.tensor([1], device="cuda").long()
        self.text_encoder.requires_grad_(False)

    def set_eval(self):
        self.unet.eval()
        self.vae_fix.eval()
        self.cfr_main_net.eval()
        self.unet.requires_grad_(False)
        self.vae_fix.requires_grad_(False)
        self.cfr_main_net.requires_grad_(False)


    def set_train_quality(self):
        self.unet.train()
        self.cfr_main_net.train()
        for n, _p in self.unet.named_parameters():
            if "quality" in n:
                _p.requires_grad = True
            if "consistency" in n:
                _p.requires_grad = False
        self.unet.conv_in.requires_grad_(True)
        for n, _p in self.cfr_main_net.named_parameters():
            _p.requires_grad = False
            
    def set_train_consistency(self):
        self.unet.train()
        self.cfr_main_net.train()
        for n, _p in self.unet.named_parameters():
            if "consistency" in n:
                _p.requires_grad = True
            if "quality" in n:
                _p.requires_grad = False
        self.unet.conv_in.requires_grad_(True)
        for n, _p in self.cfr_main_net.named_parameters():
            _p.requires_grad = True

    def set_train_together(self):
        self.unet.train()
        self.cfr_main_net.train()
        for n, _p in self.unet.named_parameters():
            if "consistency" in n:
                _p.requires_grad = True
            if "quality" in n:
                _p.requires_grad = True
        self.unet.conv_in.requires_grad_(True)
        for n, _p in self.cfr_main_net.named_parameters():
            _p.requires_grad = True
    
    
    def load_ckpt_from_state_dict(self, sd):
        # load unet lora
        self.lora_conf_encoder_quality = LoraConfig(r=sd["lora_rank_unet_quality"], init_lora_weights="gaussian", target_modules=sd["unet_lora_encoder_modules_quality"])
        self.lora_conf_decoder_quality = LoraConfig(r=sd["lora_rank_unet_quality"], init_lora_weights="gaussian", target_modules=sd["unet_lora_decoder_modules_quality"])
        self.lora_conf_others_quality = LoraConfig(r=sd["lora_rank_unet_quality"], init_lora_weights="gaussian", target_modules=sd["unet_lora_others_modules_quality"])

        self.lora_conf_encoder_consistency = LoraConfig(r=sd["lora_rank_unet_consistency"], init_lora_weights="gaussian", target_modules=sd["unet_lora_encoder_modules_consistency"])
        self.lora_conf_decoder_consistency = LoraConfig(r=sd["lora_rank_unet_consistency"], init_lora_weights="gaussian", target_modules=sd["unet_lora_decoder_modules_consistency"])
        self.lora_conf_others_consistency = LoraConfig(r=sd["lora_rank_unet_consistency"], init_lora_weights="gaussian", target_modules=sd["unet_lora_others_modules_consistency"])

        self.unet.add_adapter(self.lora_conf_encoder_quality, adapter_name="default_encoder_quality")
        self.unet.add_adapter(self.lora_conf_decoder_quality, adapter_name="default_decoder_quality")
        self.unet.add_adapter(self.lora_conf_others_quality, adapter_name="default_others_quality")

        self.unet.add_adapter(self.lora_conf_encoder_consistency, adapter_name="default_encoder_consistency")
        self.unet.add_adapter(self.lora_conf_decoder_consistency, adapter_name="default_decoder_consistency")
        self.unet.add_adapter(self.lora_conf_others_consistency, adapter_name="default_others_consistency")

        self.lora_unet_modules_encoder_quality, self.lora_unet_modules_decoder_quality, self.lora_unet_others_quality, \
        self.lora_unet_modules_encoder_consistency, self.lora_unet_modules_decoder_consistency, self.lora_unet_others_consistency= \
        sd["unet_lora_encoder_modules_quality"], sd["unet_lora_decoder_modules_quality"], sd["unet_lora_others_modules_quality"], \
            sd["unet_lora_encoder_modules_consistency"], sd["unet_lora_decoder_modules_consistency"], sd["unet_lora_others_modules_consistency"]

        for n, p in self.unet.named_parameters():
            if "lora" or "conv_in" in n:
                p.data.copy_(sd["state_dict_unet"][n])

        if self.args.load_cfr:
            for n, p in self.cfr_main_net.named_parameters():
                p.data.copy_(sd["state_dict_bpp"][n])
    
    
    
    # Adapted from pipelines.StableDiffusionXLPipeline.encode_prompt
    def encode_prompt(self, prompt_batch):
        prompt_embeds_list = []
        with torch.no_grad():
            for caption in prompt_batch:
                text_input_ids = self.tokenizer(
                    caption, max_length=self.tokenizer.model_max_length,
                    padding="max_length", truncation=True, return_tensors="pt"
                ).input_ids
                prompt_embeds = self.text_encoder(
                    text_input_ids.to(self.text_encoder.device),
                )[0]
                prompt_embeds_list.append(prompt_embeds)
        prompt_embeds = torch.concat(prompt_embeds_list, dim=0)
        return prompt_embeds

    def forward(self, c_t, batch=None, args=None):
        b, t, c, h, w = c_t.shape

        c_t = c_t.view(b * t, c, h, w)
        encoded_control = self.vae_fix.encode(c_t).latent_dist.sample() * self.vae_fix.config.scaling_factor

        uncertainty_map = batch["difference_mask"]

        c_t = c_t.view(b, t, c, h, w)
        encoded_control = encoded_control.view(b, t, -1, h // 8, w // 8)
        encoded_control, weight_map, aligned_return = self.cfr_main_net(c_t, encoded_control, uncertainty_map)
        encoded_control[:, 0, :, :, :] = aligned_return[:, 1, :, :, :]
        encoded_control = encoded_control.view(b * t, -1, h // 8, w // 8)
        c_t = c_t.view(b * t, c, h, w)

        # calculate prompt_embeddings and neg_prompt_embeddings
        prompt_embeds = self.encode_prompt(batch["prompt"])
        neg_prompt_embeds = self.encode_prompt(batch["neg_prompt"])
        null_prompt_embeds = self.encode_prompt(batch["null_prompt"])
        pos_caption_enc = prompt_embeds
        # pos_caption_enc = pos_caption_enc.repeat(t, 1, 1)

        encoded_control = encoded_control.view(b, -1, h//8, w//8)
        model_pred = self.unet(encoded_control, self.timesteps,
                               encoder_hidden_states=pos_caption_enc.to(torch.float32), ).sample

        x_denoised = encoded_control[:, 4:8, :, :] - model_pred

        output_image = (self.vae_fix.decode(x_denoised / self.vae_fix.config.scaling_factor).sample).clamp(-1, 1)

        return output_image, x_denoised, prompt_embeds, neg_prompt_embeds, weight_map, aligned_return

    def save_model(self, outf):
        sd = {}
        sd["unet_lora_encoder_modules_quality"], \
        sd["unet_lora_decoder_modules_quality"], \
        sd["unet_lora_others_modules_quality"] = \
            self.lora_unet_modules_encoder_quality, self.lora_unet_modules_decoder_quality, self.lora_unet_others_quality
        sd["unet_lora_encoder_modules_consistency"], \
        sd["unet_lora_decoder_modules_consistency"], \
        sd["unet_lora_others_modules_consistency"] = \
            self.lora_unet_modules_encoder_consistency, self.lora_unet_modules_decoder_consistency, self.lora_unet_others_consistency
        sd["lora_rank_unet_quality"] = self.lora_rank_unet_quality
        sd["lora_rank_unet_consistency"] = self.lora_rank_unet_consistency
        sd["state_dict_unet"] = {k: v for k, v in self.unet.state_dict().items() if "lora" or "conv_in" in k}
        sd["state_dict_bpp"] = {k: v for k, v in self.cfr_main_net.state_dict().items()}
        torch.save(sd, outf)


class Generator_eval(torch.nn.Module):
    def __init__(self, args):
        super().__init__()

        self.device = 'cuda'

        assert args.pretrained_path is not None
        # stage1_yaml = find_filepath(args.pretrained_path.split('/checkpoints')[0], 'hparams.yml')
        # stage1_args = read_yaml(stage1_yaml)
        # stage1_args = SimpleNamespace(**stage1_args)

        self.tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model_path, subfolder="tokenizer")
        self.text_encoder = CLIPTextModel.from_pretrained(args.pretrained_model_path, subfolder="text_encoder").cuda()
        self.sched = make_1step_sched(args)
        # self.stage1_args = stage1_args
        self.args = args

        self.vae = AutoencoderKL.from_pretrained(args.pretrained_model_path, subfolder="vae")
        self.unet = UNet2DConditionModel.from_pretrained(args.pretrained_model_path, subfolder="unet")
        # reformulate the conv_in in unet
        conv_in = nn.Conv2d(8, 320, kernel_size=3, padding=1)
        conv_in.weight.data[:, 0:4, ...] = self.unet.conv_in.weight.data
        conv_in.weight.data[:, 4:8, ...] = self.unet.conv_in.weight.data
        self.unet.conv_in = conv_in

        self.cfr_main_net = CFR_model(mid_channels=64, num_blocks=7, is_low_res_input=False,
                                      spynet_pretrained='https://download.openmmlab.com/mmediting/restorers/''basicvsr/spynet_20210409-c6c1bd09.pth')

        sd = torch.load(args.pretrained_path)

        self.load_ckpt_from_state_dict(sd)
        # vae tile
        self._init_tiled_vae(encoder_tile_size=args.vae_encoder_tiled_size, decoder_tile_size=args.vae_decoder_tiled_size)
        # # merge lora
        # if self.args.merge_and_unload_lora:
        #     print(f'MERGE LORA')
        #     print("self.args.mergeandlora: {}".format(self.args.merge_and_unload_lora))
        #     self.vae = self.vae.merge_and_unload()
        #     self.unet = self.unet.merge_and_unload()

        self.unet.enable_xformers_memory_efficient_attention()
        self.unet.to("cuda")
        self.vae.to("cuda")
        self.cfr_main_net.to("cuda")
        self.timesteps = torch.tensor([1], device="cuda").long() # self.stage1_args.timesteps
        self.text_encoder.requires_grad_(False)

    def load_ckpt_from_state_dict(self, sd):
        # load unet lora
        self.lora_conf_encoder_quality = LoraConfig(r=sd["lora_rank_unet_quality"], init_lora_weights="gaussian",
                                                   target_modules=sd["unet_lora_encoder_modules_quality"])
        self.lora_conf_decoder_quality = LoraConfig(r=sd["lora_rank_unet_quality"], init_lora_weights="gaussian",
                                                   target_modules=sd["unet_lora_decoder_modules_quality"])
        self.lora_conf_others_quality = LoraConfig(r=sd["lora_rank_unet_quality"], init_lora_weights="gaussian",
                                                  target_modules=sd["unet_lora_others_modules_quality"])

        self.lora_conf_encoder_consistency = LoraConfig(r=sd["lora_rank_unet_consistency"], init_lora_weights="gaussian",
                                                    target_modules=sd["unet_lora_encoder_modules_consistency"])
        self.lora_conf_decoder_consistency = LoraConfig(r=sd["lora_rank_unet_consistency"], init_lora_weights="gaussian",
                                                    target_modules=sd["unet_lora_decoder_modules_consistency"])
        self.lora_conf_others_consistency = LoraConfig(r=sd["lora_rank_unet_consistency"], init_lora_weights="gaussian",
                                                   target_modules=sd["unet_lora_others_modules_consistency"])

        self.unet.add_adapter(self.lora_conf_encoder_quality, adapter_name="default_encoder_quality")
        self.unet.add_adapter(self.lora_conf_decoder_quality, adapter_name="default_decoder_quality")
        self.unet.add_adapter(self.lora_conf_others_quality, adapter_name="default_others_quality")

        self.unet.add_adapter(self.lora_conf_encoder_consistency, adapter_name="default_encoder_consistency")
        self.unet.add_adapter(self.lora_conf_decoder_consistency, adapter_name="default_decoder_consistency")
        self.unet.add_adapter(self.lora_conf_others_consistency, adapter_name="default_others_consistency")

        for n, p in self.unet.named_parameters():
            if "lora" or "conv_in" in n:
                p.data.copy_(sd["state_dict_unet"][n])

        if self.args.load_cfr:
            for n, p in self.cfr_main_net.named_parameters():
                    p.data.copy_(sd["state_dict_bpp"][n])

    def set_eval(self):
        self.unet.eval()
        self.vae.eval()
        self.cfr_main_net.eval()
        self.unet.requires_grad_(False)
        self.vae.requires_grad_(False)
        self.cfr_main_net.requires_grad_(False)

    def set_train_quality(self):
        self.unet.train()
        for n, _p in self.unet.named_parameters():
            if "quality" in n:
                _p.requires_grad = True
            if "consistency" in n:
                _p.requires_grad = False

        self.cfr_main_net.train()
        for n, _p in self.cfr_main_net.named_parameters():
            _p.requires_grad = True

    def set_train_consistency(self):
        self.unet.train()
        for n, _p in self.unet.named_parameters():
            if "content" in n:
                _p.requires_grad = True
            if "consistency" in n:
                _p.requires_grad = False
                
        self.cfr_main_net.train()
        for n, _p in self.cfr_main_net.named_parameters():
            _p.requires_grad = True

    # Adapted from pipelines.StableDiffusionXLPipeline.encode_prompt
    def encode_prompt(self, prompt_batch):
        prompt_embeds_list = []
        with torch.no_grad():
            for caption in prompt_batch:
                text_input_ids = self.tokenizer(
                    caption, max_length=self.tokenizer.model_max_length,
                    padding="max_length", truncation=True, return_tensors="pt"
                ).input_ids
                prompt_embeds = self.text_encoder(
                    text_input_ids.to(self.text_encoder.device),
                )[0]
                prompt_embeds_list.append(prompt_embeds)
        prompt_embeds = torch.concat(prompt_embeds_list, dim=0)
        return prompt_embeds

    @perfcount
    @torch.no_grad()
    def forward(self, stages, c_t, uncertainty_map, prompt=None, weight_dtype=None):
        prompt_embeds = self.encode_prompt([prompt])
        b_ct, t_ct, c_ct, h_ct, w_ct = c_t.shape

        c_t = c_t.view(b_ct * t_ct, c_ct, h_ct, w_ct)
        # prompt_embeds = prompt_embeds.expand(t_ct, -1, -1)
        encoded_control = self.vae.encode(
            c_t.to(dtype=weight_dtype).cuda()).latent_dist.sample() * self.vae.config.scaling_factor

        encoded_control = encoded_control.view(b_ct, -1, h_ct//8, w_ct//8)
        weight_map = None
        aligned_input = None
        ## add tile function
        _, _, h, w = encoded_control.size()
        tile_size, tile_overlap = (self.args.latent_tiled_size, self.args.latent_tiled_overlap)
        if h*w<=tile_size*tile_size:
            ####################################################################################
            c_t = c_t.view(b_ct, t_ct, c_ct, h_ct, w_ct)
            encoded_control = encoded_control.view(b_ct, t_ct, -1, h_ct // 8, w_ct // 8)
            encoded_control, weight_map, aligned_return = self.cfr_main_net(c_t, encoded_control,
                                                                                uncertainty_map)
            encoded_control[:, 0, :, :, :] = aligned_return[:, 1, :, :, :]
            encoded_control = encoded_control.view(b_ct * t_ct, -1, h_ct // 8, w_ct // 8)
            c_t = c_t.view(b_ct * t_ct, c_ct, h_ct, w_ct)
            encoded_control = encoded_control.view(b_ct, -1, h_ct // 8, w_ct // 8)
            model_pred = self.unet(encoded_control, self.timesteps,
                                    encoder_hidden_states=prompt_embeds.to(dtype=weight_dtype), ).sample

            final_enc_ctrl = encoded_control[:, 4:8, :, :]
            ####################################################################################

        else:
            c_t = c_t.view(b_ct, t_ct * c_ct, h_ct, w_ct)
            print(f"[Tiled Latent]: the input size is {c_t.shape[-2]}x{c_t.shape[-1]}, need to tiled")
            tile_weights = self._gaussian_weights(tile_size, tile_size, 1)
            tile_size = min(tile_size, min(h, w))
            tile_weights = self._gaussian_weights(tile_size, tile_size, 1)

            grid_rows = 0
            cur_x = 0
            while cur_x < encoded_control.size(-1):
                cur_x = max(grid_rows * tile_size - tile_overlap * grid_rows, 0) + tile_size
                grid_rows += 1

            grid_cols = 0
            cur_y = 0
            while cur_y < encoded_control.size(-2):
                cur_y = max(grid_cols * tile_size - tile_overlap * grid_cols, 0) + tile_size
                grid_cols += 1

            input_list = []
            c_t_list = []
            flows_forward_fullsize_list = []
            flows_backward_fullsize_list = []
            noise_preds = []
            enc_ctrls = []

            for row in range(grid_rows):
                noise_preds_row = []
                for col in range(grid_cols):
                    if col < grid_cols - 1 or row < grid_rows - 1:
                        # extract tile from input image
                        ofs_x = max(row * tile_size - tile_overlap * row, 0)
                        ofs_y = max(col * tile_size - tile_overlap * col, 0)
                        # input tile area on total image
                    if row == grid_rows - 1:
                        ofs_x = w - tile_size
                    if col == grid_cols - 1:
                        ofs_y = h - tile_size

                    input_start_x = ofs_x
                    input_end_x = ofs_x + tile_size
                    input_start_y = ofs_y
                    input_end_y = ofs_y + tile_size

                    c_tile_start_x = ofs_x * 8
                    c_tile_end_x = (ofs_x + tile_size) * 8
                    c_tile_start_y = ofs_y * 8
                    c_tile_end_y = (ofs_y + tile_size) * 8

                    # input tile dimensions
                    input_tile = encoded_control[:, :, input_start_y:input_end_y, input_start_x:input_end_x]
                    input_list.append(input_tile)

                    c_tile = c_t[:, :, c_tile_start_y:c_tile_end_y, c_tile_start_x:c_tile_end_x]
                    uncertainty_map_tile = uncertainty_map[:, :, :, input_start_y:input_end_y, input_start_x:input_end_x]

                    c_t_list.append(c_tile)

                    if len(input_list) == 1 or col == grid_cols - 1:
                        input_list_t = torch.cat(input_list, dim=0)
                        c_t_list_t = torch.cat(c_t_list, dim=0)

                        if stages == 0:
                            set_weights_and_activate_adapters(self.unet,
                                                              ["default_encoder_consistency", "default_decoder_consistency",
                                                               "default_others_consistency", "default_encoder_quality",
                                                               "default_decoder_quality", "default_others_quality"],
                                                              [1.0, 1.0, 1.0, 0.0, 0.0, 0.0])

                            c_t_list_t = c_t_list_t.view(b_ct, t_ct, c_ct, c_t_list_t.shape[-2], c_t_list_t.shape[-1])
                            input_list_t = input_list_t.view(b_ct, t_ct, -1, input_list_t.shape[-2], input_list_t.shape[-1])

                            assert input_list_t.shape[-3] == 4, 'should be latent output'

                            input_list_t, weight_map_input_list_t, aligned_return_input_list_t = self.cfr_main_net(
                                c_t_list_t.to(dtype=weight_dtype),
                                input_list_t.to(dtype=weight_dtype),
                                uncertainty_map_tile.to(dtype=weight_dtype),
                                external_flows=None)
                            input_list_t[:, 0, :, :, :] = aligned_return_input_list_t[:, 1, :, :, :]
                            input_list_t = input_list_t.view(b_ct, -1, input_list_t.shape[-2], input_list_t.shape[-1])
                            model_pred_output_2lora = self.unet(input_list_t, self.timesteps,
                                                          cross_attention_kwargs={"scale": 1.0},
                                                          encoder_hidden_states=prompt_embeds.to(
                                                          dtype=weight_dtype), ).sample

                        elif stages == 1:

                            set_weights_and_activate_adapters(self.unet,
                                                              ["default_encoder_consistency",
                                                               "default_decoder_consistency",
                                                               "default_others_consistency", "default_encoder_quality",
                                                               "default_decoder_quality", "default_others_quality"],
                                                              [1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
                            c_tile = c_tile.view(b_ct, t_ct, c_ct, c_tile.shape[-2], c_tile.shape[-1])
                            input_list_t = input_list_t.view(b_ct, t_ct, -1, input_list_t.shape[-2],
                                                             input_list_t.shape[-1])
                            input_list_t, weight_map_input_list_t, aligned_return_input_list_t = self.cfr_main_net(
                                c_tile.to(dtype=weight_dtype),
                                input_list_t.to(dtype=weight_dtype),
                                uncertainty_map_tile.to(dtype=weight_dtype),
                                external_flows=None)
                            input_list_t[:, 0, :, :, :] = aligned_return_input_list_t[:, 1, :, :, :]
                            input_list_t = input_list_t.view(b_ct, -1, input_list_t.shape[-2], input_list_t.shape[-1])
                            model_pred_output_2lora = self.unet(input_list_t, self.timesteps,
                                                                cross_attention_kwargs={"scale": 1.0},
                                                                encoder_hidden_states=prompt_embeds.to(
                                                                dtype=weight_dtype), ).sample

                        model_out = model_pred_output_2lora
                        model_enc = input_list_t.view(b_ct, -1, input_list_t.shape[-2], input_list_t.shape[-1])

                        input_list = []
                        c_t_list = []
                        flows_forward_fullsize_list = []
                        flows_backward_fullsize_list = []
                    noise_preds.append(model_out)
                    enc_ctrls.append(model_enc[:, 4:8, :, :])

            noise_pred = torch.zeros([b_ct, 4, h_ct//8, w_ct//8], device=encoded_control.device)
            contributors = torch.zeros([b_ct, 4, h_ct//8, w_ct//8], device=encoded_control.device)
            enc_ctrl = torch.zeros([b_ct, 4, h_ct//8, w_ct//8], device=encoded_control.device)

            # Add each tile contribution to overall latents
            for row in range(grid_rows):
                for col in range(grid_cols):
                    if col < grid_cols - 1 or row < grid_rows - 1:
                        # extract tile from input image
                        ofs_x = max(row * tile_size - tile_overlap * row, 0)
                        ofs_y = max(col * tile_size - tile_overlap * col, 0)
                        # input tile area on total image
                    if row == grid_rows - 1:
                        ofs_x = w - tile_size
                    if col == grid_cols - 1:
                        ofs_y = h - tile_size

                    input_start_x = ofs_x
                    input_end_x = ofs_x + tile_size
                    input_start_y = ofs_y
                    input_end_y = ofs_y + tile_size

                    noise_pred[:, :, input_start_y:input_end_y, input_start_x:input_end_x] += noise_preds[
                                                                                                  row * grid_cols + col] * tile_weights
                    enc_ctrl[:, :, input_start_y:input_end_y, input_start_x:input_end_x] += enc_ctrls[
                                                                                                  row * grid_cols + col] * tile_weights
                    contributors[:, :, input_start_y:input_end_y, input_start_x:input_end_x] += tile_weights
            
            # Average overlapping areas with more than 1 contributor
            noise_pred /= contributors
            enc_ctrl /= contributors
            model_pred = noise_pred
            final_enc_ctrl = enc_ctrl

        x_denoised = final_enc_ctrl - model_pred

        output_image = (self.vae.decode(x_denoised.to(dtype=weight_dtype) / self.vae.config.scaling_factor).sample).clamp(-1, 1)

        if stages == 0 or weight_map is None or aligned_input is None:
            weight_map = torch.zeros([b_ct, t_ct, 1, h_ct//8, w_ct//8], device=encoded_control.device, dtype=encoded_control.dtype)
            for weight_map_index in range(1, t_ct):
                weight_map[:, weight_map_index] = uncertainty_map[:, weight_map_index - 1]

            aligned_input = output_image


        return output_image, x_denoised, prompt_embeds, weight_map, aligned_input

    def _init_tiled_vae(self,
            encoder_tile_size = 256,
            decoder_tile_size = 256,
            fast_decoder = False,
            fast_encoder = False,
            color_fix = False,
            vae_to_gpu = True):
        # save original forward (only once)
        if not hasattr(self.vae.encoder, 'original_forward'):
            setattr(self.vae.encoder, 'original_forward', self.vae.encoder.forward)
        if not hasattr(self.vae.decoder, 'original_forward'):
            setattr(self.vae.decoder, 'original_forward', self.vae.decoder.forward)

        encoder = self.vae.encoder
        decoder = self.vae.decoder

        self.vae.encoder.forward = VAEHook(
            encoder, encoder_tile_size, is_decoder=False, fast_decoder=fast_decoder, fast_encoder=fast_encoder, color_fix=color_fix, to_gpu=vae_to_gpu)
        self.vae.decoder.forward = VAEHook(
            decoder, decoder_tile_size, is_decoder=True, fast_decoder=fast_decoder, fast_encoder=fast_encoder, color_fix=color_fix, to_gpu=vae_to_gpu)

    def _gaussian_weights(self, tile_width, tile_height, nbatches):
        """Generates a gaussian mask of weights for tile contributions"""
        from numpy import pi, exp, sqrt
        import numpy as np

        latent_width = tile_width
        latent_height = tile_height

        var = 0.01
        midpoint = (latent_width - 1) / 2  # -1 because index goes from 0 to latent_width - 1
        x_probs = [exp(-(x-midpoint)*(x-midpoint)/(latent_width*latent_width)/(2*var)) / sqrt(2*pi*var) for x in range(latent_width)]
        midpoint = latent_height / 2
        y_probs = [exp(-(y-midpoint)*(y-midpoint)/(latent_height*latent_height)/(2*var)) / sqrt(2*pi*var) for y in range(latent_height)]

        weights = np.outer(y_probs, x_probs)
        return torch.tile(torch.tensor(weights, device=self.device), (nbatches, self.unet.config.in_channels, 1, 1))




