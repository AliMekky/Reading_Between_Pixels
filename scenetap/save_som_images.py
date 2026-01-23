import argparse
import json
import os

import numpy as np
import torch
from PIL import Image
from pytorch_lightning import seed_everything
from utils.som import SoM
from tqdm import tqdm

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="typo_base_complex",
                        help="Dataset name: typo_base_complex, typo_base_color, vqav2_val2014")
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--question-file", type=str, default="tables/question.jsonl")
    parser.add_argument("--log_dir", type=str, default="./som_images")
    parser.add_argument("--filter", type=float, default=10.0)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=3153)

    # som
    parser.add_argument("--slider", type=float, default=3)

    # seed
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    # seed everything
    seed_everything(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # logger
    args.log_dir = os.path.join(args.log_dir, args.dataset, f"slider_{args.slider}")
    args.log_dir = os.path.join(args.log_dir, f"seed_{args.seed}", f"filter_{args.filter}")
    os.makedirs(args.log_dir, exist_ok=True)

    # load the question file
    with open(args.question_file, 'r') as f:
        questions = json.load(f)

    # SoM
    semsam_cfg = "../SoM/configs/semantic_sam_only_sa-1b_swinL.yaml"
    seem_cfg = "../SoM/configs/seem_focall_unicl_lang_v1.yaml"
    semsam_ckpt = "../SoM/swinl_only_sam_many2many.pth"
    sam_ckpt = "../SoM/sam_vit_h_4b8939.pth"
    seem_ckpt = "../SoM/seem_focall_v1.pt"

    som = SoM(semsam_cfg, seem_cfg, semsam_ckpt, sam_ckpt, seem_ckpt)
    image_set = set()

    for i, entry in tqdm(enumerate(questions), total=len(questions)):
        if i < args.start or i >= args.end:
            continue
        
        image_name = entry["image"]
        if image_name in image_set:
            continue
        image_set.add(image_name)
        
        # Check if image already exists in output folder
        output_image_path = os.path.join(args.log_dir, image_name)
        if os.path.exists(output_image_path):
            continue
        
        image_path = os.path.join(args.image_folder, image_name)
        # Load the image
        image = Image.open(image_path).convert("RGB")

        # get segmentation image and map
        seg_image, mask = som.inference(image=image, slider=args.slider, mode="Automatic", alpha=0.2,
                                             label_mode="Number",
                                             anno_mode=['Mask', 'Mark'], filter=args.filter)
        seg_image = Image.fromarray(seg_image)
        seg_image = seg_image.resize(image.size)
        seg_image.save(output_image_path)
        # print(f"image saved into {output_image_path}")
        
        # save the mask to .npy file, mask it now a list of numpy arrays
        mask_save_path = os.path.join(args.log_dir, image_name.split('.')[0] + '.npy')
        np.save(mask_save_path, mask)
        # print(f"mask saved into {mask_save_path}")




