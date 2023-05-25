# adversial_git
## model
* deepMal-model
* lstm-model
* MGREL-model
* M1CNN-model
* M2CNN-model
* RTETC-model
* TLSVEC-model
* MaIDIST-model
* DISSTILLER-model
## attack
* fgsm
* deepfool
## 运行
* python ds.py（fgsm）
* python ds_deepfool.py(deepfool)
* --d_1 选择你的第一个数据集
* --d_2 选择你的第二个数据集
* --f 选择你的特征
* --m 选择你的模型
* --e 选择epoch
* --b 选择batch_size的大小
## 备注
其中fgsm攻击，已设置多次循环运行，尝试多种攻击粒的模型鲁棒性
