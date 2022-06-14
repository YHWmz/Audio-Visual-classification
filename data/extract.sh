#!/bin/bash
MODEL_TYPE="Transfer_ResNet38"
#CHECKPOINT_PATH="./ModelWeight/Wavegram_Logmel_Cnn14_mAP=0.439.pth"
CHECKPOINT_PATH="./ModelWeight/ResNet38_mAP=0.434.pth"
#CUDA_VISIBLE_DEVICES=1
EXPER_NAME="embedding_mix"
DATETIME=''
#`date -u -Is`


python extract_openl3_emb.py --input_path /dssg/home/acct-stu/stu464/data/audio_visual_scenes/evaluation_setup/fold1_test.csv  \
                                    --dataset_path /dssg/home/acct-stu/stu464/data/audio_visual_scenes \
                                    --output_path /dssg/home/acct-stu/stu519/av_scene_classify_finetune/data/new_feature \
                                    --split test \
