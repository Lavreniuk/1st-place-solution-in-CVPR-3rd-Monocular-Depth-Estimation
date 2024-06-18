# [CVPR2024] 3rd Monocular Depth Estimation Challenge
It is [1st place solution of the EVP++ team](https://jspenmar.github.io/MDEC/) in **CVPR2024** 3rd Monocular Depth Estimation Challenge.  
The objective was to evaluate monocular depth estimation on the challenging SYNS-Patches dataset. This dataset contains complex natural, urban and indoor scenes.

Different SOTA models for the Monocular Depth Estimation have been tested and the best results were obtained by our [EVP model](https://lavreniuk.github.io/EVP/) and [Depth-Anything model](https://github.com/LiheYoung/Depth-Anything).
The best submission was received using Depth-Anything model probably due to very large dataset that it used for pretraining.

## :rocket: News
* **(June 18, 2024):** EVP++ solution has been presented on [CVPR 2024](https://jspenmar.github.io/MDEC), the [presentation file](https://github.com/Lavreniuk/1st-place-solution-in-CVPR-3rd-Monocular-Depth-Estimation/blob/main/CVPR2024-EVP%2B%2Bsolution-in-3nd-Monocular-Depth-Estimation-Challenge.pdf). :fire::fire:
* **(April 27, 2024):** EVP++ solution has been published at CVPR 2024, here is [paper link](https://openaccess.thecvf.com/content/CVPR2024W/MDEC/papers/Spencer_The_Third_Monocular_Depth_Estimation_Challenge_CVPRW_2024_paper.pdf). :fire:

## Dataset
[Kitti](https://github.com/jspenmar/monodepth_benchmark/blob/main/api/data/README.md#download), [Virtual KITTI 2](https://europe.naverlabs.com/research-old2/computer-vision/proxy-virtual-worlds/) and [DIODE](https://diode-dataset.org/) datasets have been downloaded and preprocessed for training the models.
Set the correct paths in the vkitti_and_diode_preprocessing.py file, and for preprocessing vkitti and diode run:
```
python vkitti_and_diode_preprocessing.py
```

## Training
Set up the configs and run:
```
CUDA_VISIBLE_DEVICES=0 python train_mono.py -m zoedepth -d kitti --pretrained_resource=""
```

## Inference
Set the correct paths in the inference.py and inference_sliding_window.py files.
For whole image inference run:
```
python inference.py
```
For image inference using sliding window (for better edges) run:
```
python inference_sliding_window.py
```

## Results
Final leaderboard results on the test data
| # | User            | Entries | Date of Last Entry | Team Name | Total Rank↓ | F-Score↑ | F-Score (Edges)↑  | MAE↓     | RMSE↓   | AbsRel↓ | Edge Accuracy↓ | Edge Completion↓ |
|---|-----------------|---------|--------------------|-----------|------------|---------|------------------|---------|---------|---------|---------------|-----------------|
| 1 | lavreniuk       | 32      | 03/25/24           | EVP++     |18  | 20.8658 (4)  | 10.9224 (3)  | 3.7086 (1)  | 6.5308 (1)  | 19.0214 (1) | 2.8835 (3) | 6.7700 (5)     |
| 2 | zhouguangyuan   | 10      | 03/25/24           | PICO-MR   |31  | 23.7210 (1)  | 11.0130 (2)  | 3.7787 (2)  | 6.6115 (2)  | 21.2386 (4) | 3.8974 (17)| 4.4517 (3)     |
| 3 | moyushin        | 12      | 03/25/24           |           |38  | 23.2499 (2)  | 10.7783 (4)  | 3.8725 (3)  | 6.7038 (3)  | 21.6987 (5) | 3.5911 (14)| 9.8569 (7)     |
| 4 | inso-13         | 3       | 03/20/24           |           |41  | 18.5956 (8)  | 9.4273 (9)   | 3.9224 (4)  | 7.1649 (4)  | 20.1185 (2) | 2.8921 (4) | 15.6460 (10)   |
| 5 | pihai           | 15      | 03/24/24           |           |51  | 17.8328 (9)  | 9.1387 (12)  | 4.1147 (5)  | 7.7332 (6)  | 21.2310 (3) | 2.9465 (5) | 17.8145 (11)   |
| 6 | Depth_3DV       | 1       | 03/24/24           |           |57  | 20.4244 (6)  | 10.1868 (5)  | 4.4079 (9)  | 7.8909 (8)  | 23.9416 (9) | 3.6105 (16)| 5.7953 (4)     |
| 7 | surajiitd       | 9       | 03/24/24           | visioniitd|57  | 19.0651 (7)  | 9.9200 (7)   | 4.5318 (10) | 7.9626 (9)  | 23.2745 (7) | 3.2596 (11)| 7.9953 (6)     |
| 8 | erdosv001       | 3       | 03/24/24           |           |58  | 20.7673 (5)  | 9.9617 (6)   | 4.3302 (8)  | 7.8348 (7)  | 27.7973 (11)| 3.4458 (13)| 13.2527 (8)    |
| 9 | jsk24           | 11      | 03/25/24           | RGA Inc.  |65  | 22.7924 (3)  | 11.5192 (1)  | 5.2061 (13) | 9.2339 (13) | 28.8613 (12)| 4.1541 (21)| 0.8980 (2)     |
| 10| weijianing      | 2       | 03/25/24           |           |67  | 17.8121 (10) | 9.7525 (8)   | 5.0386 (12) | 8.9196 (11) | 24.0101 (10)| 3.1615 (7) | 14.1550 (9)    |
| 11| luo0207         | 6       | 03/03/24           |           |67  | 16.9120 (12) | 9.0666 (14)  | 4.1357 (6)  | 7.3481 (5)  | 22.0509 (6) | 3.2377 (10)| 18.5220 (14)   |
| 12| qing            | 10      | 03/14/24           |           |73  | 17.5704 (11) | 9.1273 (13)  | 4.2759 (7)  | 8.3635 (10) | 23.3455 (8) | 3.1765 (8) | 20.6621 (16)   |
| 13| hitcslj         | 2       | 03/10/24           | HIT-AIIA  |89  | 16.7148 (13) | 9.2525 (10)  | 5.4767 (15) | 11.0510 (19)| 34.2035 (19)| 2.5703 (1) | 18.0436 (12)   |
| 14| dagouqin        | 1       | 03/01/24           |           |95  | 16.4478 (14) | 8.8896 (15)  | 5.2907 (14) | 10.5310 (17)| 33.6741 (18)| 2.5965 (2) | 18.7283 (15)   |
| 15| al              | 7       | 03/24/24           | ReadingLS |98  | 14.8093 (16) | 8.1357 (16)  | 5.0099 (11) | 8.9448 (12) | 29.3938 (13)| 3.2837 (12)| 30.2778 (18)   |
| 16| hyc123          | 3       | 03/25/24           |           |107 | 15.9223 (15) | 9.1679 (11)  | 8.2542 (20) | 13.8783 (20)| 43.8823 (20)| 4.1054 (20)| 0.7403 (1)     |
| 17| yogurts         | 2       | 03/16/24           |           |110 | 13.7089 (18) | 7.5505 (19)  | 5.4867 (16) | 9.4419 (14) | 30.7377 (15)| 3.6072 (15)| 18.3600 (13)   |
| 18| SmartHust       | 1       | 03/03/24           |           |112 | 11.8998 (19) | 8.0770 (17)  | 6.3256 (19) | 10.8861 (18)| 30.4607 (14)| 2.9862 (6) | 33.6279 (19)   |
| 19| mdec            | 1       | 01/16/24           |           |118 | 13.7211 (17) | 7.7630 (18)  | 5.5645 (17) | 9.7169 (15) | 32.0420 (16)| 3.9712 (18)| 21.6256 (17)   |
| 20| journey2japan   | 5       | 03/24/24           |           |132 | 11.3561 (20) | 6.6000 (21)  | 5.9075 (18) | 9.9886 (16) | 33.4098 (17)| 3.9832 (19)| 54.6467 (21)   |
| 21| smhh            | 5       | 03/25/24           |           |133 | 11.0444 (21) | 7.0866 (20)  | 8.7645 (21) | 15.8637 (21)| 63.3160 (21)| 3.2209 (9) | 40.6098 (20)   |
