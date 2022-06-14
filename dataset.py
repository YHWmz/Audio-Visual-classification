import numpy as np
import torch
from torch.utils import data
import h5py

import vggish
from efficientnet_pytorch import EfficientNet
import vggish_input
import os
import pandas
import soundfile as sf

class SceneDataset(data.Dataset):

    def __init__(self,
                 audio_feature,
                 video_feature,
                 audio_transform=None,
                 video_transform=None):
        super().__init__()
        self.audio_feature = audio_feature
        self.video_feature = video_feature
        self.audio_transform = audio_transform
        self.video_transform = video_transform
        self.audio_hf = None
        self.video_hf = None

        self.all_files = []

        def traverse(name, obj):
            if isinstance(obj, h5py.Dataset):
                self.all_files.append(name)

        hf = h5py.File(self.audio_feature, 'r')
        hf.visititems(traverse)
        hf.close()
        print("Finish loading indexes")

    def __len__(self):
        return len(self.all_files)
    
    def __getitem__(self, index):
        if self.audio_hf is None:
            self.audio_hf = h5py.File(self.audio_feature, 'r')
        if self.video_hf is None:
            self.video_hf = h5py.File(self.video_feature, 'r')

        audio_feat = []
        aid = self.all_files[index]
        audio_feat = self.audio_hf[aid][:96, :]
        # import pdb; pdb.set_trace()
        if self.audio_transform:
            audio_feat = self.audio_transform(audio_feat)
        
        vid = aid.replace("audio", "video")
        video_feat = self.video_hf[vid][:96, :]
        if self.video_transform:
            video_feat = self.video_transform(video_feat)
        
        target = int(aid.split('/')[0])

        audio_feat = torch.as_tensor(audio_feat).float()
        video_feat = torch.as_tensor(video_feat).float()
        target = torch.as_tensor(target).long()
        # print(audio_feat.shape, video_feat.shape)
        return {
            "aid": aid.split("/")[-1],
            "audio_feat": audio_feat,
            "video_feat": video_feat,
            "target": target
        }

def collate(data_list):
    if "audio" in data_list[0]:
        audios = []
    elif "video" in data_list[0]:
        videos = []
    labels = []
    fnames = []
    for data in data_list:
        if "audio" in data:
            if len(data["audio"].shape) == 1:
                if data["audio"].shape[0] >= 220500:
                    signal = data["audio"][0:220500]
                elif data["audio"].shape[0] < 220500:
                    signal = np.zeros(220500)
                    signal[0:data["audio"].shape[0]] = data["audio"]
            else:
                signal = data["audio"]
            audios.append(signal.astype(np.float32))
        elif "video" in data:
            videos.append(data["video"])
        labels.append(data["label"])
        fnames.append(data["fname"])
    if "audio" in data_list[0]:
        audios = np.array(audios)
        audios = torch.tensor(audios).float()
        labels = torch.tensor(labels).long()
        return audios, labels, fnames
    elif "video" in data_list[0]:
        videos = np.array(videos)
        videos = torch.tensor(videos)
        labels = torch.tensor(labels).long()
        return videos, labels, fnames

class InferenceAudioDataset(object):
    def __init__(self, path_csv, dataset_path, classes_dic):
        df = pandas.read_csv(path_csv, sep="\t")

        input_dir_audios = df['filename_audio'].values
        input_dir_audios = list(input_dir_audios)
        input_dir_audios.sort()
        self.input_dir_audios = input_dir_audios
        self.dataset_path = dataset_path
        self.classes_dic = classes_dic
        self.audio_hf = None

    def __getitem__(self, index):
        if self.audio_hf is None:
            self.audio_hf = h5py.File(self.dataset_path, 'r')

        input_dir_audio = self.input_dir_audios[index]
        audio_name = os.path.join(self.dataset_path, input_dir_audio)
        label = audio_name.split('/')[-1].split('-')[0]
        label = self.classes_dic[label]

        fname = audio_name.split('/')[-1].split('.')[0]
        audio = np.array(self.audio_hf[fname])

        return {
            "audio": audio,
            "label": label,
            "fname": fname
        }

    def __len__(self):
        return len(self.input_dir_audios)

# class InferenceAudioDataset(object):
#     def __init__(self, path_csv, dataset_path, classes_dic):
#         df = pandas.read_csv(path_csv, sep="\t")
#
#         input_dir_audios = df['filename_audio'].values
#         input_dir_audios = list(input_dir_audios)
#         input_dir_audios.sort()
#         self.input_dir_audios = input_dir_audios
#         self.dataset_path = dataset_path
#         self.classes_dic = classes_dic
#
#     def __getitem__(self, index):
#         input_dir_audio = self.input_dir_audios[index]
#         audio_name = os.path.join(self.dataset_path, input_dir_audio)
#         label = audio_name.split('/')[-1].split('-')[0]
#         label = self.classes_dic[label]
#
#         audio, sr = sf.read(audio_name, dtype='int16')
#         assert audio.dtype == np.int16, 'Bad sample type: %r' % audio.dtype
#         audio = audio / 32768.0  # Convert to [-1.0, +1.0]
#         audio = vggish_input.waveform_to_examples(audio, sr, False)
#
#         return {
#             "audio": audio,
#             "label": label,
#             "fname": input_dir_audio
#         }
#
#     def __len__(self):
#         return len(self.input_dir_audios)

class InferenceVideoDataset(object):

    def __init__(self, path_csv, dataset_path, classes_dic):
        df = pandas.read_csv(path_csv, sep="\t")

        input_dir_audios = df['filename_audio'].values
        input_dir_audios = list(input_dir_audios)
        input_dir_audios.sort()
        self.input_dir_audios = input_dir_audios
        self.dataset_path = dataset_path
        self.classes_dic = classes_dic
        self.video_hf = None

    def __getitem__(self, index):
        if self.video_hf is None:
            self.video_hf = h5py.File(self.dataset_path, 'r')

        input_dir_audio = self.input_dir_audios[index]
        audio_name = os.path.join(self.dataset_path, input_dir_audio)
        label = audio_name.split('/')[-1].split('-')[0]
        label = self.classes_dic[label]

        fname = audio_name.split('/')[-1].split('.')[0]
        video = np.array(self.video_hf[fname])

        return {
            "video": video,
            "label": label,
            "fname": fname
        }

    def __len__(self):
        return len(self.input_dir_audios)

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
    labels = torch.tensor(labels).long()
    return audios, videos, labels, fnames