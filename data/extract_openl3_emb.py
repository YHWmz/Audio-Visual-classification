import multiprocessing
import os
import pandas
import numpy as np
import h5py
from torch.utils.data import DataLoader
from IPython import embed
from tqdm import tqdm, trange
import argparse
import sys
from tqdm import tqdm
import torch
import random
sys.path.append("..")
import models
os.environ["CUDA_VISIBLE_DEVICES"]="3"

seed = 0
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

parser = argparse.ArgumentParser(description='Generating audio and video features from L3 network for training data')
parser.add_argument('--input_path', type=str,
                    help='give the file path of train.csv file you generated from split_data.py')
parser.add_argument('--dataset_path', type=str,
                    help='give the path of TAU-urban-audio-visual-scenes-2021-development data')
parser.add_argument('--output_path', type=str,
                    help='give the folder path of where you want to save audio and video features')
parser.add_argument('--split', type=str,
                    help='give the dataset split, train / val / test')
parser.add_argument('--batch_size', type=int, default=32,
                    help='inference batch size')
parser.add_argument('--num_process', type=int, default=4,
                    help='the number of loading data processes')

args, _ = parser.parse_known_args()

#### digitalize the target according to the alphabet order#####
classes_dic = {
    'airport': 0,
    'bus': 1,
    'metro': 2,
    'metro_station': 3,
    'park': 4,
    'public_square': 5,
    'shopping_mall': 6,
    'street_pedestrian': 7,
    'street_traffic': 8,
    'tram': 9
}

split = args.split
batch_size = args.batch_size
num_process = args.batch_size

#### set the data path#######
path_csv = args.input_path

#### load the training data using pandas######
df = pandas.read_csv(path_csv, sep="\t")

input_dir_audio = df['filename_audio'].values
input_dir_audio = list(input_dir_audio)
input_dir_audio.sort()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#embed()
####### load the model to extract the audio and video embeddings#####
model = models.earlyfusion_dense(len(classes_dic), device).to(device)
print('Finish loading')

save_data_audio = os.path.join(args.output_path, 'audio_features_data')
if not os.path.exists(save_data_audio):
    os.makedirs(save_data_audio)
    print("Directory " , save_data_audio ,  " Created ")
else:
    print("Directory " , save_data_audio ,  " already exists")

save_data_video = os.path.join(args.output_path, 'video_features_data')
if not os.path.exists(save_data_video):
    os.makedirs(save_data_video)
    print("Directory ", save_data_video,  " Created ")
else:
    print("Directory ", save_data_video,  " already exists")

audio_data = "/dssg/home/acct-stu/stu519/av_scene_classify_finetune/data/raw_data/test_rawaudio.hdf5"
video_data = "/dssg/home/acct-stu/stu519/av_scene_classify_finetune/data/raw_data/test_video-data_new.hdf5"
class anvDataset(object):
    def __init__(self, path_csv, audio_data, video_data, classes_dic):
        df = pandas.read_csv(path_csv, sep="\t")

        input_dir_audios = df['filename_audio'].values
        input_dir_audios = list(input_dir_audios)
        input_dir_audios.sort()
        self.input_dir_audios = input_dir_audios
        self.audio_data = audio_data
        self.video_data = video_data
        self.classes_dic = classes_dic
        self.audio_hf = None
        self.video_hf = None

    def __getitem__(self, index):
        if self.audio_hf is None:
            self.audio_hf = h5py.File(self.audio_data, 'r')
        if self.video_hf is None:
            self.video_hf = h5py.File(self.video_data, 'r')

        input_dir_audio = self.input_dir_audios[index]
        audio_name = input_dir_audio
        label = audio_name.split('/')[-1].split('-')[0]
        label = self.classes_dic[label]

        fname = audio_name.split('/')[-1].split('.')[0]
        audio = np.array(self.audio_hf[fname])
        video = np.array(self.video_hf[fname])

        return {
            "audio": audio,
            "video": video,
            "label": label,
            "fname": fname
        }

    def __len__(self):
        return len(self.input_dir_audios)

def fusion_collate(data_list):
    audios = []
    videos = []
    labels = []
    fnames = []
    for data in data_list:
        if len(data["audio"].shape) == 1:
            if data["audio"].shape[0] >= 220500:
                signal = data["audio"][0:220500]
            elif data["audio"].shape[0] < 220500:
                signal = np.zeros(220500)
                signal[0:data["audio"].shape[0]] = data["audio"]
        else:
            signal = data["audio"]
        audios.append(signal.astype(np.float32))

        videos.append(data["video"])
        labels.append(data["label"])
        fnames.append(data["fname"])

    audios = np.array(audios)
    audios = torch.tensor(audios).float()
    videos = np.array(videos)
    videos = torch.tensor(videos)
    return audios, videos, labels, fnames

################create training features #########################

print('generating features data ...')

dataloader = DataLoader(
    anvDataset(path_csv, audio_data, video_data, classes_dic),
    batch_size=8,
    num_workers=args.num_process,
    collate_fn=fusion_collate,
    shuffle=True
)
#
# for batch in tqdm((dataloader)):
#     audio, video, labels, fnames = batch
#     print(fnames)
#     print('*'*100)
#     audio = audio.to(device)
#     video = video.to(device)
#     # labels = labels.numpy()
#     audio_embs, video_embs = model.extract_feature(audio, video)
#     print(audio_embs[0][0:20])
#     print(video_embs[0][0:20])
#     exit(0)
#     # logit = model(audio, video)



hf_audio = h5py.File(os.path.join(save_data_audio, f'{split}.hdf5'), 'w')
hf_video = h5py.File(os.path.join(save_data_video, f'{split}.hdf5'), 'w')

for batch in tqdm(dataloader):
    audio, video, labels, fnames = batch
    audio = audio.to(device)
    video = video.to(device)
    # labels = labels.numpy()
    audio_embs, video_embs = model.extract_feature(audio, video)
    video_embs = video_embs.detach().cpu().numpy()
    audio_embs = audio_embs.detach().cpu().numpy()
    for idx in range(len(fnames)):
        # labels[idx] = labels[idx].numpy()
        hf_video.create_dataset(
            str(labels[idx]) + '/' + "video" + '/' + fnames[idx].replace('.wav','').replace('audio', 'video'),
            data=video_embs[idx])
        hf_audio.create_dataset(
            str(labels[idx]) + '/' + "audio" + '/' + fnames[idx].replace('.wav', ''),
            data=audio_embs[idx])
hf_video.close()
hf_audio.close()
# ################create training features#########################
 