# Single Domain Generalization for Crowd Counting

This is an official repository for our CVPR2024 work, "Single Domain Generalization for Crowd Counting". You can read our paper [here]().

## Requirements
* Python 3.10.12
* PyTorch 2.0.1
* Torchvision 0.15.2
* Others specified in [requirements.txt](requirements.txt)

## Data Preparation
1. Download ShanghaiTech and UCF-QNRF datasets from official sites and unzip them.
2. Run the following commands to preprocess the datasets:
    ```
    python utils/preprocess_data.py --origin-dir [path_of_ShanghaiTech]/part_A --data-dir data/sta
    python utils/preprocess_data.py --origin-dir [path_of_ShanghaiTech]/part_B --data-dir data/stb
    python utils/preprocess_data.py --origin-dir [path_of_UCF-QNRF] --data-dir data/qnrf
    ```
3. Run the following commands to generate GT density maps:
    ```
    python dmap_gen.py --path data/sta
    python dmap_gen.py --path data/stb
    python dmap_gen.py --path data/qnrf
    ```

## Training
Run the following command:
```
python main.py --task train --config configs/sta_train.yml
```
You may edit the `.yml` config file as you like.

## Testing
Run the following commands after you specify the path to the model weight in the config file:
```
python main.py --task test --config configs/sta_test_stb.yml
python main.py --task test --config configs/sta_test_qnrf.yml
```

## Pretrained Weights
We provide pretrained weights in the table below:
| Source | Performance                                   | Weights                                                                                                                                          |
| ------ | --------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------ |
| A      | B: 11.4MAE, 19.7MSE<br>Q: 115.7MAE, 199.8MSE  | [OneDrive](https://hkustconnect-my.sharepoint.com/:u:/g/personal/zpengac_connect_ust_hk/EaWnUPugulxIiP4gK2F_bqcBJwJhru0aWa8JH6l8Dbk5DQ?e=2B0kJP) |
| B      | A: 99.6MAE, 182.9MSE<br>Q: 165.6MAE, 290.4MSE | [OneDrive](https://hkustconnect-my.sharepoint.com/:u:/g/personal/zpengac_connect_ust_hk/EZp54KXswPVFnXHP2dhIGRABUZYrH4ZXaxBr5y9M7io2Bg?e=DnGP6v) |
| Q      | A: 65.5MAE, 110.1MSE<br>B: 12.3MAE, 24.1MSE   | [OneDrive](https://hkustconnect-my.sharepoint.com/:u:/g/personal/zpengac_connect_ust_hk/EQYuWSlkgMtNqWUbL5lmv1YBOkmeXF3sa0hNFQ5QGcSpvQ?e=Hy1Pf6) |

## Citation
If you find this work helpful in your research, please cite the following:
```
@inproceedings{pengMPCount2024,
  title = {Single Domain Generalization for Crowd Counting},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR 2024)},
  author = {Peng, Zhuoxuan and Chan, S.-H. Gary},
  year = {2024}
}
```