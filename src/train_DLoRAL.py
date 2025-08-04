import os
import gc
import lpips
import clip
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.utils import set_seed
from PIL import Image
from torchvision import transforms
from tqdm.auto import tqdm
import random

import diffusers
from diffusers.utils.import_utils import is_xformers_available
from diffusers.optimization import get_scheduler

from DLoRAL_model import CSDLoss, Generator
from my_utils.training_utils import parse_args, PairedSROnlineTxtDataset_Pexel_and_REDS_and_LSDIRshift

from pathlib import Path
from accelerate.utils import set_seed, ProjectConfiguration
from accelerate import DistributedDataParallelKwargs

from src.my_utils.wavelet_color_fix import adain_color_fix, wavelet_color_fix
from my_utils.training_utils import batch_calculate_optical_flow

from datetime import datetime


class DynamicDatasetManager:
    def __init__(self, args):
        self.args = args
        self.accelerator = None
        self.current_stage = 0  # 0: Pexel, 1: LSDIR
        
        self.original_txt_paths = args.dataset_txt_paths
        
        self.lsdir_txt_path = args.lsdir_txt_path
        self.pexel_txt_path = args.pexel_txt_path
        
        # Set stage switching steps
        self.stage_switch_step = args.quality_iter + 1
        
        # Initialize Stage 1 (LSDIR)
        self._setup_stage(0)
    
    def _setup_stage(self, stage):
        """Set dataset for current stage"""
        self.current_stage = stage
        
        if stage == 0:  # Pexel
            self.args.dataset_txt_paths = self.pexel_txt_path
            self.current_dataset = PairedSROnlineTxtDataset_Pexel_and_REDS_and_LSDIRshift(args=self.args)
        else:  # LSDIR
            self.args.dataset_txt_paths = self.lsdir_txt_path
            self.current_dataset = PairedSROnlineTxtDataset_Pexel_and_REDS_and_LSDIRshift(args=self.args)
        
        # Create new dataloader
        self.dataloader = torch.utils.data.DataLoader(
            self.current_dataset, 
            batch_size=self.args.train_batch_size, 
            shuffle=True, 
            num_workers=self.args.dataloader_num_workers
        )
        
        # Prepare dataset if accelerator is already set
        if self.accelerator is not None:
            self.dataloader = self.accelerator.prepare(self.dataloader)
        
        self.dataloader_iter = iter(self.dataloader)
    
    def set_accelerator(self, accelerator):
        """Set accelerator"""
        self.accelerator = accelerator
        # Re-prepare current dataset
        self.dataloader = self.accelerator.prepare(self.dataloader)
        self.dataloader_iter = iter(self.dataloader)
    
    def get_batch(self, global_step):
        """Get a batch of data"""

        target_stage = 0
        if (global_step > self.args.quality_iter and global_step <= self.args.quality_iter_1_final) or \
        (global_step > self.args.quality_iter_1_final + self.args.quality_iter_2):
            target_stage = 1

        # Only perform switch operation when needed
        if target_stage != self.current_stage:
            self.current_stage = target_stage
            self._setup_stage(self.current_stage)
            print(f"Step {global_step}: switch to Stage {self.current_stage} - {'LSDIR' if self.current_stage == 1 else 'Pexel'}")
        
        # Get next batch
        try:
            batch = next(self.dataloader_iter)
        except StopIteration:
            # Recreate iterator if exhausted
            self.dataloader_iter = iter(self.dataloader)
            batch = next(self.dataloader_iter)
        
        return batch


def set_requires_grad(params, requires_grad):
    for param in params:
        param.requires_grad = requires_grad


def main(args):
    logging_dir = Path(args.output_dir, args.logging_dir)
    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
        kwargs_handlers=[ddp_kwargs],
    )

    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    if args.seed is not None:
        set_seed(args.seed)

    if accelerator.is_main_process:
        os.makedirs(os.path.join(args.output_dir, "checkpoints"), exist_ok=True)
        # os.makedirs(os.path.join(args.output_dir, "eval"), exist_ok=True)
        os.makedirs(os.path.join(args.output_dir, "train_output"), exist_ok=True)

        import shutil
        import sys
        import inspect
        
        train_and_model_dir = os.path.join(args.output_dir, "train_and_model")
        os.makedirs(train_and_model_dir, exist_ok=True)

        # Create directory for training experience information
        exp_info_dir = os.path.join(args.output_dir, "train_command")
        os.makedirs(exp_info_dir, exist_ok=True)
        
        # 1. Save the complete command used to run the script
        with open(os.path.join(exp_info_dir, "command.txt"), "w") as f:
            f.write(" ".join(sys.argv) + "\n")
            f.write(f"Run at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        # 2. Try to save the shell script that launched this (if run via script)
        try:
            # Get parent process info (likely the shell script)
            import psutil
            parent = psutil.Process(os.getpid()).parent()
            if parent and parent.cmdline():
                script_path = parent.cmdline()[0]
                if script_path.endswith('.sh'):
                    shutil.copy(script_path, os.path.join(exp_info_dir, "launch_script.sh"))
        except Exception as e:
            print(f"Could not save launch script: {e}")

        # Copy current running train script
        current_train_script = sys.argv[0]
        shutil.copy(current_train_script, train_and_model_dir)

        try:
            generator_file = inspect.getfile(Generator)
            shutil.copy(generator_file, train_and_model_dir)
        except Exception as e:
            print("Copy Generator file failed:", e)
            exit(0)

    net_osediff = Generator(args)

    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            net_osediff.unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available, please install it by running `pip install xformers`")

    if args.gradient_checkpointing:
        net_osediff.unet.enable_gradient_checkpointing()

    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    net_csd = CSDLoss(args=args, accelerator=accelerator)
    net_csd.requires_grad_(False)

    net_lpips = lpips.LPIPS(net='vgg').cuda()
    net_lpips.requires_grad_(False)

    # # set gen adapter
    net_osediff.unet.set_adapter(['default_encoder_quality', 'default_decoder_quality','default_others_quality','default_encoder_consistency', 'default_decoder_consistency','default_others_consistency'])

    param_dict_conv_in = {n: p for n, p in net_osediff.unet.conv_in.named_parameters()}

    base_param_names = [n for n in param_dict_conv_in.keys() if "quality" not in n and "consistency" not in n]
                                         
    param_dict = {n: p for n, p in net_osediff.unet.named_parameters()}

    quality_param_names = [n for n in param_dict.keys() if "lora" in n and "quality" in n]
    consistency_param_names = [n for n in param_dict.keys() if "lora" in n and "consistency" in n]

    param_dict_bpp = {n: p for n, p in net_osediff.cfr_main_net.named_parameters()}
    bpp_param_names = [n for n in param_dict_bpp.keys() if "spynet" not in n]

    quality_params = [param_dict[n] for n in quality_param_names]
    consistency_params = [param_dict[n] for n in consistency_param_names]
    bpp_params = [param_dict_bpp[n] for n in bpp_param_names]
    base_params = [param_dict_conv_in[n] for n in base_param_names]

    print("=== quality_params ===")
    for n, param in zip(quality_param_names, quality_params):
        print(f"Parameter Name: {n}")

    # 打印 consistency_params
    print("=== consistency_params ===")
    for n, param in zip(consistency_param_names, consistency_params):
        print(f"Parameter Name: {n}")

    # 打印 bpp_params
    print("=== bpp_params ===")
    for n, param in zip(bpp_param_names, bpp_params):
        print(f"Parameter Name: {n}")

    # 打印 base_params
    print("=== base_params ===")
    for n, param in zip(base_param_names, base_params):
        print(f"Parameter Name: {n}")

    param_groups = [
        {'params': quality_params, 'name': 'quality'},                 # lora_quality params
        {'params': consistency_params, 'name': 'consistency'},         # lora_consistency params
        {'params': bpp_params, 'name': 'cfr_main_net'},             # cfr_main_net params
        {'params': base_params, 'name': 'base'}                        # conv_in params
    ]

    optimizer = torch.optim.AdamW(
        param_groups,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon
    )

    optimizer = torch.optim.AdamW(
        param_groups,
        lr=args.learning_rate, 
        betas=(args.adam_beta1, args.adam_beta2), 
        weight_decay=args.adam_weight_decay, 
        eps=args.adam_epsilon
    )
    #################################


    optimizer = torch.optim.AdamW(param_groups, lr=args.learning_rate,
                                  betas=(args.adam_beta1, args.adam_beta2), 
                                  weight_decay=args.adam_weight_decay, eps=args.adam_epsilon)


    lr_scheduler = get_scheduler(args.lr_scheduler, optimizer=optimizer,
                                 num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
                                 num_training_steps=args.max_train_steps * accelerator.num_processes,
                                 num_cycles=args.lr_num_cycles, power=args.lr_power, )


    dataset_manager = DynamicDatasetManager(args)
    dl_train = dataset_manager.dataloader

    # init RAM
    from ram.models.ram_lora import ram
    from ram import inference_ram as inference
    ram_transforms = transforms.Compose([
        transforms.Resize((384, 384)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    RAM = ram(pretrained=args.ram_path,
            pretrained_condition=None,
            image_size=384,
            vit='swin_l')
    RAM.eval()
    RAM.to("cuda", dtype=torch.float16)

    # Prepare everything with our `accelerator`.
    net_osediff, optimizer, dl_train, lr_scheduler = accelerator.prepare(
        net_osediff, optimizer, dl_train, lr_scheduler
    )
    dataset_manager.set_accelerator(accelerator)
    net_lpips = accelerator.prepare(net_lpips)
    # renorm with image net statistics
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16



    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        tracker_config = dict(vars(args))
        accelerator.init_trackers(args.tracker_project_name, config=tracker_config)

    progress_bar = tqdm(range(0, args.max_train_steps), initial=0, desc="Steps",
                        disable=not accelerator.is_local_main_process, )

    # start the training loop
    global_step = 0
    stages = 0

    if args.resume_ckpt is not None:
        args.quality_iter = -1
    for epoch in range(0, args.num_training_epochs):
        # Update dataset length each epoch (may have switched)
        epoch_steps = len(dataset_manager.dataloader)
        for _ in range(epoch_steps):
            batch = dataset_manager.get_batch(global_step)
            l_acc = [net_osediff]
            with accelerator.accumulate(*l_acc):
                x_src = batch["conditioning_pixel_values"]
                x_tgt = batch["output_pixel_values"]

                B, T, C, H, W = x_src.shape

                # get text prompts from GT
                x_tgt_ram = ram_transforms(x_tgt[:, 1, :, :, :] * 0.5 + 0.5)
                caption = inference(x_tgt_ram.to(dtype=torch.float16), RAM)
                batch["prompt"] = [f'{each_caption}, {args.pos_prompt_vsd}' for each_caption in caption]



                ###############################################################
                # Multi-stage training schedule with smooth transitions
                # Stage 0 (Consistency): Enhanced consistency loss, reduced LPIPS
                # Stage 1 (Quality): Full LPIPS/CSD, reduced consistency loss
                # Timeline: Consistency -> Quality -> Consistency -> Quality (final)
                # Each stage switch includes 1000-step smooth transition
                ###############################################################

                
                # [0, quality_iter], consistency stage
                if global_step <= args.quality_iter:
                    lambda_l2 = args.lambda_l2
                    lambda_lpips = args.lambda_lpips * 0.02
                    lambda_vsd = 0.0
                    lambda_consistency = args.lambda_consistency * 2
                    stages = 0
                
                # (quality_iter, quality_iter_1_final]
                elif global_step > args.quality_iter and global_step <= args.quality_iter_1_final:
                    stages = 1  # set to quality stage
                    
                    transition_steps = 1000

                    # if in transition
                    if global_step <= args.quality_iter + transition_steps:
                        # from cons to quality stage
                        progress = (global_step - args.quality_iter) / transition_steps
                        lambda_l2 = args.lambda_l2
                        lambda_lpips = args.lambda_lpips * (0.02 + progress * 0.98)  # LPIPS scales: small -> full
                        lambda_vsd = args.lambda_vsd * progress  # VSD scales: 0 -> full
                        lambda_consistency = args.lambda_consistency * (2.0 - progress)  # Consistency scales: 2x -> 1x
                    
                    
                    # if not in transition
                    else:
                        lambda_l2 = args.lambda_l2
                        lambda_lpips = args.lambda_lpips
                        lambda_vsd = args.lambda_vsd
                        lambda_consistency = args.lambda_consistency

                # (quality_iter_1_final, quality_iter_1_final + quality_iter_2]    
                elif global_step > args.quality_iter_1_final and global_step <= args.quality_iter_1_final + args.quality_iter_2:
                    stages = 0  # set to consistency stage
                    
                    transition_steps = 1000

                    # if in transition
                    if global_step <= args.quality_iter_1_final + transition_steps:
                        # from quality to cons
                        progress = (global_step - args.quality_iter_1_final) / transition_steps
                        lambda_l2 = args.lambda_l2
                        lambda_lpips = args.lambda_lpips
                        lambda_vsd = args.lambda_vsd
                        lambda_consistency = args.lambda_consistency * (1.0 + progress)
                    else:
                        lambda_l2 = args.lambda_l2
                        lambda_lpips = args.lambda_lpips 
                        lambda_vsd = args.lambda_vsd
                        lambda_consistency = args.lambda_consistency * 2

                # (quality_iter_1_final + quality_iter_2, ∞)  
                else:
                    stages = 1  # quality stage
                    
                     # if in transition
                    transition_steps = 1000

                    if global_step <= args.quality_iter_1_final + args.quality_iter_2 + transition_steps:
                        # from cons to quality

                        progress = (global_step - (args.quality_iter_1_final + args.quality_iter_2)) / transition_steps
                        lambda_l2 = args.lambda_l2
                        lambda_lpips = args.lambda_lpips
                        lambda_vsd = args.lambda_vsd
                        lambda_consistency = args.lambda_consistency * (2.0 - progress)
                        
                        if global_step % 100 == 0:
                            print(f"Quality 2 transition: step {global_step}, progress={progress:.3f}, vsd={lambda_vsd:.4f}")
                    else:
                        lambda_l2 = args.lambda_l2
                        lambda_lpips = args.lambda_lpips
                        lambda_vsd = args.lambda_vsd
                        lambda_consistency = args.lambda_consistency


                batch["stages"] = stages


                net_osediff.module.unet.set_adapter(
                        ['default_encoder_quality', 'default_decoder_quality', 'default_others_quality',
                         'default_encoder_consistency', 'default_decoder_consistency', 'default_others_consistency'])

                if stages == 1:
                    set_requires_grad(quality_params, True)
                    set_requires_grad(consistency_params, False)
                    set_requires_grad(bpp_params, False)
                    set_requires_grad(base_params, True)
                    net_osediff.module.set_train_quality()  # begin to optimize consistency

                elif stages == 0:
                    set_requires_grad(quality_params, False)
                    set_requires_grad(consistency_params, True)
                    set_requires_grad(bpp_params, True)
                    set_requires_grad(base_params, True)
                    net_osediff.module.set_train_consistency()  # begin to optimize quality

                # forward pass
                x_tgt_pred1, latents_pred1, prompt_embeds, neg_prompt_embeds, weight_map, aligned_input = net_osediff(x_src[:, 0:2, :, :, :], batch=batch, args=args)
                x_tgt_pred2, latents_pred2, prompt_embeds, neg_prompt_embeds, weight_map, aligned_input = net_osediff(x_src[:, 1:3, :, :, :], batch=batch, args=args)


                # Save training samples every 1000 steps
                if global_step % 1000 == 0:
                    c_t_save_dir = os.path.join(os.path.join(args.output_dir, "train_output"), str(global_step))
                    if not os.path.exists(c_t_save_dir):
                        os.makedirs(c_t_save_dir, exist_ok=True)

                    output_pil_eval1 = transforms.ToPILImage()(x_tgt_pred1[0, :, :, :].cpu() * 0.5 + 0.5)
                    output_pil_eval2 = transforms.ToPILImage()(x_tgt_pred2[0, :, :, :].cpu() * 0.5 + 0.5)
                    output_pil_eval_ref1 = transforms.ToPILImage()(x_src[0, 1, :, :, :].cpu() * 0.5 + 0.5)
                    output_pil_eval_ref2 = transforms.ToPILImage()(x_src[0, 2, :, :, :].cpu() * 0.5 + 0.5)
                    output_pil_eval_gt1 = transforms.ToPILImage()(x_tgt[0, 1, :, :, :].cpu() * 0.5 + 0.5)
                    output_pil_eval_gt2 = transforms.ToPILImage()(x_tgt[0, 2, :, :, :].cpu() * 0.5 + 0.5)
                    
                    output_pil_eval1.save('{}/{}_1.png'.format(c_t_save_dir, batch['seq_name'][0]))
                    output_pil_eval2.save('{}/{}_2.png'.format(c_t_save_dir, batch['seq_name'][0]))
                    output_pil_eval_ref1.save('{}/{}_1_lr.png'.format(c_t_save_dir, batch['seq_name'][0]))
                    output_pil_eval_ref2.save('{}/{}_2_lr.png'.format(c_t_save_dir, batch['seq_name'][0]))
                    output_pil_eval_gt1.save('{}/{}_1_gt.png'.format(c_t_save_dir, batch['seq_name'][0]))
                    output_pil_eval_gt2.save('{}/{}_2_gt.png'.format(c_t_save_dir, batch['seq_name'][0]))
                    


                # Reconstruction loss
                loss_l2 = F.mse_loss(x_tgt_pred1.float(), x_tgt[:, 1, :, :, :].float(), reduction="mean") * lambda_l2
                loss_l2_2 = F.mse_loss(x_tgt_pred2.float(), x_tgt[:, 2, :, :, :].float(), reduction="mean") * lambda_l2
                loss_lpips = net_lpips(x_tgt_pred1.float(), x_tgt[:, 1, :, :, :].float()).mean() * lambda_lpips
                loss_lpips_2 = net_lpips(x_tgt_pred2.float(), x_tgt[:, 2, :, :, :].float()).mean() * lambda_lpips

                consistency_source = batch_calculate_optical_flow(x_tgt_pred1, x_tgt_pred2)
                consistency_target = batch_calculate_optical_flow(x_tgt[:, 1, :, :, :], x_tgt[:, 2, :, :, :])


                loss_consistency = F.l1_loss(consistency_source, consistency_target, reduction="mean") * lambda_consistency

                loss = loss_l2 + loss_lpips + loss_l2_2 + loss_lpips_2 + loss_consistency

                # CSD Loss
                try:
                    noise_csd = torch.randn_like(latents_pred1)
                    if stages == 1 and lambda_vsd > 0:
                        if torch.isnan(latents_pred1).any():
                            print(f"Step {global_step}: NaN detected in latents_pred1 before CSD calculation")
                            latents_pred1_safe = torch.where(torch.isnan(latents_pred1), torch.zeros_like(latents_pred1), latents_pred1)
                            latents_pred2_safe = torch.where(torch.isnan(latents_pred2), torch.zeros_like(latents_pred2), latents_pred2)
                            
                            loss_csd = net_csd.cal_csd(latents_pred1_safe, prompt_embeds, neg_prompt_embeds, args, noise_csd) * lambda_vsd
                            loss_csd_2 = net_csd.cal_csd(latents_pred2_safe, prompt_embeds, neg_prompt_embeds, args, noise_csd) * lambda_vsd
                        else:
                            loss_csd = net_csd.cal_csd(latents_pred1, prompt_embeds, neg_prompt_embeds, args, noise_csd) * lambda_vsd
                            loss_csd_2 = net_csd.cal_csd(latents_pred2, prompt_embeds, neg_prompt_embeds, args, noise_csd) * lambda_vsd
                            
                        if torch.isnan(loss_csd).any():
                            print(f"Step {global_step}: NaN detected in loss_csd, using zero instead")
                            loss_csd = torch.tensor(0.0, device=loss.device)
                        if torch.isnan(loss_csd_2).any():
                            print(f"Step {global_step}: NaN detected in loss_csd_2, using zero instead")
                            loss_csd_2 = torch.tensor(0.0, device=loss.device)
                    else:
                        loss_csd = torch.tensor(0.0, device=loss.device)
                        loss_csd_2 = torch.tensor(0.0, device=loss.device)
                        
                    loss = loss + loss_csd + loss_csd_2
                    
                except Exception as e:
                    print(f"Error in CSD loss calculation at step {global_step}: {e}")
                    loss_csd = torch.tensor(0.0, device=loss.device)
                    loss_csd_2 = torch.tensor(0.0, device=loss.device)

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(optimizer.param_groups[0]['params'], args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=args.set_grads_to_none)

            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                if accelerator.is_main_process:
                    logs = {}
                    # log all the losses
                    logs["loss_csd"] = loss_csd.detach().item()
                    logs["loss_l2"] = loss_l2.detach().item()
                    logs["stages"] = stages
                    logs["loss_lpips"] = loss_lpips.detach().item()
                    logs["loss_consistency"] = loss_consistency.detach().item()
                    progress_bar.set_postfix(**logs)

                    # checkpoint the model
                    if global_step % args.checkpointing_steps == 1:
                        outf = os.path.join(args.output_dir, "checkpoints", f"model_{global_step}.pkl")
                        accelerator.unwrap_model(net_osediff).save_model(outf)


                    accelerator.log(logs, step=global_step)

        del x_src, x_tgt, x_tgt_pred1, x_tgt_pred2, latents_pred1, latents_pred2, prompt_embeds, neg_prompt_embeds
        del batch
        del weight_map, aligned_input
        gc.collect()
        torch.cuda.empty_cache()

if __name__ == "__main__":
    args = parse_args()
    main(args)
