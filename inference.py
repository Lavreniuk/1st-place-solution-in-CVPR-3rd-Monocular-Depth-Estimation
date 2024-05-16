from zoedepth.utils.config import get_config
from zoedepth.models.builder import build_model
import torch
import numpy as np
from PIL import Image
import os
import tqdm


# Define function to convert linear depth into disparity
def to_inv(depth):
    depth = depth.clamp(min=1.19e-7)
    disp = (depth > 0) / depth.clamp(min=1.19e-7)
    return disp

def load_config(dataset):
    conf = get_config("zoedepth", "eval", dataset)
    if dataset == "kitti":
        conf.pretrained_resource = 'local::depth/shortcuts/monodepth3_checkpoints/ZoeDepthv1_23-Mar_15-32-90f7975043dc_3.00.pt'
    elif dataset == "nyu":
        conf.pretrained_resource = 'local::./checkpoints/depth_anything_metric_depth_indoor.pt'
    #conf.img_size = [518, 1708]
    return conf

def load_model(conf):
    return build_model(conf).to('cuda' if torch.cuda.is_available() else 'cpu')

def infer_images(model, file_list):
    preds = []
    for file_path in tqdm.tqdm(file_list):
        image = Image.open(file_path).convert("RGB")
        pred = model.infer_pil(image, pad_input=False, with_flip_aug=False)
        preds.append(pred)
    return preds

def main():
    # Load configurations for datasets
    outdoor_conf = load_config("kitti")  # For Outdoor dataset
    indoor_conf = load_config("nyu")     # For Indoor dataset

    # Build models
    outdoor_model = load_model(outdoor_conf)
    indoor_model = load_model(indoor_conf)

    # Base directory for inference
    base_dir = "depth/monodepth3/data/syns_patches"

    with open(os.path.join(base_dir, "splits/test_files.txt"), "r") as file:
        preds = []
        for line in tqdm.tqdm(file):
            folder_name, image_name = line.strip().split()
            file_path = os.path.join(base_dir, folder_name, "images", image_name)
            image = Image.open(file_path).convert("RGB")
            if int(folder_name) < 81:
                pred = outdoor_model.infer_pil(image, pad_input=False, with_flip_aug=False)
            else:
                pred = indoor_model.infer_pil(image, pad_input=False, with_flip_aug=False)
            preds.append(pred)
        
        preds_np = np.array(preds)
        preds_tensor = torch.tensor(preds_np)
        preds_tensor = preds_tensor.unsqueeze(0)

    out = preds_tensor
    out = to_inv(out)[0]
    np.savez_compressed('pred', pred=out)


if __name__ == "__main__":
    main()
