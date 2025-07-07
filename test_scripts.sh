#!/bin/bash

python src/test_DLoRAL.py     \
--pretrained_model_path /home/notebook/data/group/syj/OSEDiff/OSEDiff/preset_models/stable-diffusion-2-1-base     \
--ram_ft_path /home/notebook/data/group/syj/DLoRAL/preset/models/DAPE.pth     \
--ram_path '/home/notebook/data/group/syj/DLoRAL/preset/models/ram_swin_large_14m.pth'     \
--merge_and_unload_lora False     \
--process_size 512     \
--pretrained_model_name_or_path '/home/notebook/data/group/syj/OSEDiff/OSEDiff/preset_models/stable-diffusion-2-1-base'     \
--vae_encoder_tiled_size 4096     \
--load_cfr     \
--pretrained_path /home/notebook/data/group/syj/OSEDiff_video/OSEDiff/experience/OSEDiff_lsdirffhq_bs016_supirNeg_Pexel_2lora_topk_3in2out_opticalflowloss_updatedset_fixnoise_dynamic_optimizersetTrue_0509/checkpoints/model_11001.pkl     \
--stages 1     \
-i /home/notebook/data/group/syj/VideoLQ_videos/     \
-o /home/notebook/data/group/syj/DLoRAL/results_videolq_11001_wholeimage