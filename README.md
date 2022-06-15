# 音视频场景识别
## 依赖
```bash
pip install -r requirements.txt
```
## 主要代码结构
- config： 模型配置文件
- data：包含处理好的原始数据以及数据处理脚本
- experiment：保存了实验结果
- ModelWeight：PANNs的模型参数文件
- Net：定义了PANNs的不同模型结构
- psla_models：定义了PSLA的模型结构
- train.py：模型训练脚本
- evaluate.py：生成预测文件
- eval_prediction.py：预测结果性能的评价脚本
- best_run.sh：用于复现最好配置下的结果

## 代码复现
1. 数据：

   为了加快数据的读取速度，我提前将音频与视频数据进行了预处理并存入 ./data/raw_data中。请下载以下6个数据文件并放入上述文件夹。
   - [train_audio链接](https://jbox.sjtu.edu.cn/l/31EmzH), [train_video链接](https://jbox.sjtu.edu.cn/l/H1zPpM)
   - [val_audio链接](https://jbox.sjtu.edu.cn/l/F1GEsI), [val_video链接](https://jbox.sjtu.edu.cn/l/F1Gzrv)
   - [test_audio链接](https://jbox.sjtu.edu.cn/l/91SM43), [test_video链接](https://jbox.sjtu.edu.cn/l/P1yOR5)
2. 预训练模型：
   要复现代码，需要下载以下三个模型
   - PANNs：[jbox链接](https://jbox.sjtu.edu.cn/l/X1y2xi)，下载后放入ModelWeight内。
   - Video模态finetune后的模型：[jbox链接](https://jbox.sjtu.edu.cn/l/K1zyK9)，下载后放入./ModelWeight/video_best中。
   - Audio模态finetune 后的模型：[jbox链接](https://jbox.sjtu.edu.cn/l/g1cYbF)，下载后放入./ModelWeight/audio_best中。
3. 复现最优配置的结果：
    
    在当前文件夹下，运行
```bash
bash best_run.sh
```

3. ./experiments/best_run下保存了最优配置下得到的各种结果。


