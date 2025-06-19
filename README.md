<div align="center">
<h2>One-Step Diffusion for Detail-Rich and Temporally Consistent Video Super-Resolution</h2>

[Yujing Sun](https://yjsunnn.github.io/)<sup>1,2, *</sup> | 
[Lingchen Sun](https://scholar.google.com/citations?hl=zh-CN&tzom=-480&user=ZCDjTn8AAAAJ)<sup>1,2, *</sup> | 
[Shuaizheng Liu](https://scholar.google.com/citations?user=wzdCc-QAAAAJ&hl=en)<sup>1,2</sup> | 
[Rongyuan Wu](https://scholar.google.com/citations?user=A-U8zE8AAAAJ&hl=zh-CN)<sup>1,2</sup> | 
[Zhengqiang Zhang](https://scholar.google.com/citations?user=F15mLDYAAAAJ&hl=en)<sup>1</sup> | 
[Qiaosi Yi](https://dblp.org/pid/249/8335.html)<sup>1,2</sup> |
[Lei Zhang](https://www4.comp.polyu.edu.hk/~cslzhang)<sup>1,2</sup>

<sup>1</sup>The Hong Kong Polytechnic University, <sup>2</sup>OPPO Research Institute
</div>


## ⏰ Update
- **2025.06.17**: The repo are released.

:star: If DLoRAL is helpful to your videos or projects, please help star this repo. Thanks! :hugs:

## 🌟 Overview Framework

<p align="center">

<img src="assets/pipeline.svg" alt="DLoRAL Framework">

</p>

🛠️**Training**: A dynamic dual-stage training scheme alternates between optimizing temporal coherence (consistency stage) and refining high-frequency spatial details (enhancement stage) with smooth loss interpolation to ensure stability.

🖼️**Inference**: During inference, both C-LoRA and D-LoRA are merged into the frozen diffusion UNet, enabling one-step enhancement of low-quality inputs into high-quality outputs.
