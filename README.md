# DLoRAL
Official Implementation of “One-Step Diffusion for Detail-Rich and Temporally Consistent Video Super-Resolution”

## ⏰ Update
- **2025.06.17**: The repo are released.

:star: If DLoRAL is helpful to your videos or projects, please help star this repo. Thanks! :hugs:

## 🌟 Overview Framework

<p align="center">

<img src="assets/pipeline.svg" alt="DLoRAL Framework">

</p>

🛠️**Training**: A dynamic dual-stage training scheme alternates between optimizing temporal coherence (consistency stage) and refining high-frequency spatial details (enhancement stage) with smooth loss interpolation to ensure stability.

🖼️**Inference**: During inference, both C-LoRA and D-LoRA are merged into the frozen diffusion UNet, enabling one-step enhancement of low-quality inputs into high-quality outputs.

## 😍 Visual Results
