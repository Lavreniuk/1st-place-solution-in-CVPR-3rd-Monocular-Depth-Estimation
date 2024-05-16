import os
import numpy as np
from PIL import Image


def process_vkitti_data(rgb_folder, depth_folder, output_file):
    def list_image_files(root_dir):
        image_files = []
        for dirpath, _, filenames in os.walk(root_dir):
            for filename in filenames:
                if filename.endswith(('.png', '.jpg', '.jpeg')) and \
                        'rain' not in dirpath and 'fog' not in dirpath and \
                        '15-deg' not in dirpath and 'sunset' not in dirpath and \
                        'Camera_1' not in dirpath:
                    image_files.append(os.path.join(dirpath, filename))
        return image_files

    # List image files in both folders
    folder1_files = sorted(list_image_files(rgb_folder))
    folder2_files = sorted(list_image_files(depth_folder))

    # Zip the lists together to iterate over them
    zipped_files = zip(folder1_files, folder2_files)

    with open(output_file, "w") as f:
        for file1, file2 in zipped_files:
            relative_path1 = os.path.relpath(file1, depth_folder)
            relative_path2 = os.path.relpath(file2, depth_folder)
            distance = 721.5377  # Example distance, you need to replace this with actual distance
            f.write(f"{relative_path1} {relative_path2} {distance}\n")


def process_diode_data(data_folder, output_file):
    def process_npy_files(file_path):
        # Load depth data from .npy file
        depth_data = np.load(file_path)
        # Load corresponding mask file
        mask_file = file_path.replace("_depth.npy", "_depth_mask.npy")
        mask_data = np.load(mask_file)
        # Set depth values to 0 where mask value is 0
        depth_data[mask_data == 0] = 0
        # Multiply depth values by 256
        depth_data *= 256
        # Get filename without extension
        filename = file_path.replace('.npy', '.png')
        # Save depth data as .png file
        depth_image = Image.fromarray(depth_data.astype(np.uint16).squeeze())
        print(filename)
        depth_image.save(filename)

    for root, dirs, files in os.walk(data_folder):
        for file in files:
            # Check if the file is a depth .npy file
            if file.endswith("_depth.npy"):
                npy_file_path = os.path.join(root, file)
                process_npy_files(npy_file_path)

    with open(output_file, 'w') as f:
        for root, dirs, files in os.walk(data_folder):
            for file in files:
                if file.endswith('_depth.png'):
                    depth_file_path = os.path.join(root, file)
                    jpg_file_path = depth_file_path.replace('_depth.png', '.png')
                    # Read the depth image and convert it to a NumPy array
                    d = Image.open(depth_file_path)
                    d = np.array(d)
                    if np.sum(d/256 > 10) > 10000:
                        f.write(f"{jpg_file_path} {depth_file_path} 721.5377\n")


def main():
    # Paths for Virtual KITTI data processing
    vkitti_rgb_folder = 'depth/vkitti_rgb'
    vkitti_depth_folder = 'depth/vkitti_depth'
    vkitti_output_file = 'vkitti_image_pairs.txt'

    # Paths for DIODE data processing
    diode_data_folder = "depth/DIODE/train/outdoor"
    diode_output_file = "diode_image_pairs.txt"

    # Process Virtual KITTI data
    process_vkitti_data(vkitti_rgb_folder, vkitti_depth_folder, vkitti_output_file)

    # Process DIODE data
    process_diode_data(diode_data_folder, diode_output_file)


if __name__ == "__main__":
    main()
