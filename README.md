# 🌈 BRC-CDM

**Bi-Residual Compression Network with Conditional Diffusion Model for Hyperspectral Image Compression**

---

## 🔧 1. Training the BRC Compression Network

Before training the full BRC-CDM framework, you must first train the BRC compression network.

### 🛠️ Modify Command-Line Arguments

Update the following arguments in your training command to match your local setup:
```
--out_dir_train path/to/save/training/logs \
--out_dir_test path/to/save/testing/results \
--checkpoint_dir path/to/save/checkpoints \
--train_path path/to/your/training/data \
--test_path path/to/your/testing/data
```
Example:
```
python train.py \
--out_dir_train ./logs/train_logs \
--out_dir_test ./logs/test_logs \
--checkpoint_dir ./checkpoints \
--train_path /data/HSI/train \
--test_path /data/HSI/test
```
📁 Update Dataset Path
In the dataset folder, open the corresponding dataset loading script (e.g., hsi_dataset.py) and update the root_dir variable to point to your dataset path:
```
self.root_dir = "/your/custom/path/to/hyperspectral/data"
```
We provide the datasets :
https://pan.baidu.com/s/1VdYe_JfshNE-fMEZKbk3rg?pwd=tn7w Extraction code: tn7w 


4. Pretrained Models and Training Logs
We provide the training logs and pretrained weights for reproducibility and further evaluation.
You can download them from the following link:
https://pan.baidu.com/s/1Ui1iKO4TkyssQMEIeln9ZA?pwd=8piq Extraction code: 8piq

🧠 2. Training the Full BRC-CDM Model
In our paper, we freeze the pretrained BRC compression network and train the full BRC-CDM model on top of it.

The training code for BRC-CDM is located in the epsilonparam directory.

✅ Steps to Train BRC-CDM:
Open the configuration file in the epsilonparam/config folder.
Modify the following fields to match your dataset and desired output locations:

result_root: Path to save final results

tensorboard_root: Path to save TensorBoard logs

train_path: Path to your training data

test_path: Path to your test data

Run the training script:
```
python train.py
```
Make sure the pretrained BRC compression network is loaded correctly before starting the training.

📦 BRC-CDM Pretrained Weights and TensorBoard Logs
We provide the pretrained weights and TensorBoard logs of the BRC-CDM model for reproducibility and analysis.

🔗 BRC-CDM Results and Weights
📁 File name: BRC-CDM-result
🔗 Download link: https://pan.baidu.com/s/18IVlnDRq6vo-CacmuKHVag?pwd=ravq
🔒 Extraction code: ravq


📊 TensorBoard Logs
📁 File name: BRC-CDM-tensorboard
🔗 Download link: https://pan.baidu.com/s/1WaMcQM-67lNkh-tylsNH_A?pwd=na4u
🔒 Extraction code: na4u


