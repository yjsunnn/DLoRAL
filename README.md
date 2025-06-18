# DLoRAL
Official Implementation of “One-Step Diffusion for Detail-Rich and Temporally Consistent Video Super-Resolution”

## ⏰ Update
- **2025.06.17**: The repo are released.

:star: If DLoRAL is helpful to your videos or projects, please help star this repo. Thanks! :hugs:

## 🌟 Overview Framework

![DLoRAL](figs/framework.png)


(a) Training procedure of PiSA-SR. During the training process, two LoRA modules are respectively optimized for pixel-level and semantic-level enhancement.

(b) Inference procedure of PiSA-SR. During the inference stage, users can use the default setting to reconstruct the high-quality image in one-step diffusion or adjust λ<sub>pix</sub> and λ<sub>sem</sub> to control the strengths of pixel-level and semantic-level enhancement.
## 😍 Visual Results
