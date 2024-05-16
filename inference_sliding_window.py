import os
import torch
import cv2
import numpy as np
from PIL import Image
from zoedepth.models.builder import build_model
from zoedepth.utils.config import get_config
import torch.nn.functional as F

def predict_on_windows(image, window_size, step_size, model):
    predictions = []
    for x, y, window in sliding_window(image, window_size, step_size):
        prediction = model.infer_pil(Image.fromarray(window), pad_input=False, with_flip_aug=False)
        predictions.append((x, y, prediction))
    return predictions

def merge_predictions_v1(predictions, image_size, pred_full):
    result = np.zeros(image_size[:2], np.float32)
    counts = np.zeros(image_size[:2], np.float32)
    cut_pixels = 4
    for x, y, prediction in predictions:
        h, w = prediction.shape[:2]
        part_d = pred_full[y+cut_pixels:y-cut_pixels+h, x+cut_pixels:x+w-cut_pixels]
        prediction = prediction[cut_pixels:-cut_pixels, cut_pixels:-cut_pixels]
        min_pred1, max_pred1 = part_d.min(), part_d.max()
        min_pred2, max_pred2 = prediction.min(), prediction.max()
        prediction = ((prediction - min_pred2) / (max_pred2 - min_pred2)) * (max_pred1 - min_pred1) + min_pred1
        result[y+cut_pixels:y+h-cut_pixels, x+cut_pixels:x-cut_pixels+w] += prediction
        counts[y+cut_pixels:y+h-cut_pixels, x+cut_pixels:x-cut_pixels+w] += 1
    counts[counts == 0] = 1
    result /= counts
    result[result == 0] = pred_full[result == 0]
    return result

def merge_predictions_v2(predictions, image_size):
    result = np.zeros(image_size[:2], np.float32)
    counts = np.zeros(image_size[:2], np.float32)
    for x, y, prediction in predictions:
        h, w = prediction.shape[:2]
        result[y:y+h, x:x+w] += prediction
        counts[y:y+h, x:x+w] += 1
    counts[counts == 0] = 1
    result /= counts
    return result

def sliding_window(image, window_size, step_size):
    h, w = image.shape[:2]
    for y in range(0, h - window_size[0] + 1, step_size):
        for x in range(0, w - window_size[1] + 1, step_size):
            yield x, y, image[y:y + window_size[0], x:x + window_size[1]]
    for x in range(0, w - window_size[1] + 1, step_size):
        yield x, h - window_size[0], image[h - window_size[0]:h, x:x + window_size[1]]
    for y in range(0, h - window_size[0] + 1, step_size):
        yield w - window_size[1], y, image[y:y + window_size[0], w - window_size[1]:w]

def main():
    conf_kitti = get_config("zoedepth", "eval", "kitti")
    conf_kitti.pretrained_resource = 'local::./checkpoints/depth_anything_metric_depth_outdoor.pt'
    model_kitti = build_model(conf_kitti).to('cuda' if torch.cuda.is_available() else 'cpu')

    conf_nyu = get_config("zoedepth", "eval", "nyu")
    conf_nyu.pretrained_resource = 'local::./checkpoints/depth_anything_metric_depth_indoor.pt'
    model_nyu = build_model(conf_nyu).to('cuda' if torch.cuda.is_available() else 'cpu')

    window_size = (376, 752)  # Define window size
    step_size = 16  # Define step size

    base_dir = "depth/monodepth3/data/syns_patches"
    with open(os.path.join(base_dir, "splits/test_files.txt"), "r") as file:
        preds = []
        for line in file:
            folder_name, image_name = line.strip().split()
            file_path = os.path.join(base_dir, folder_name, "images", image_name)
            image = np.array(Image.open(file_path).convert("RGB"))
            model = model_kitti if int(folder_name) < 81 else model_nyu
            pred = predict_on_windows(image, window_size, step_size, model)
            pred_full = model.infer_pil(image, pad_input=False, with_flip_aug=False)
            pred = merge_predictions_v1(pred, image.shape, pred_full) if int(folder_name) < 81 else merge_predictions_v2(pred, image.shape)
            preds.append(pred)
        preds_np = np.array(preds)
        preds_tensor = torch.tensor(preds_np)
        preds_tensor = preds_tensor.unsqueeze(0)
        out = F.interpolate(preds_tensor, size=(376, 621), mode='bilinear', align_corners=False)
        out = to_inv(out)[0]
        np.savez_compressed('pred', pred=out)


if __name__ == "__main__":
    main()
