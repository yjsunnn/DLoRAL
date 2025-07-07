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
        img_t_dir = os.path.join(parent_dir, "LR_lowparam_nonoise_frames")

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