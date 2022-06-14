import numpy as np
import h5py
import os
from moviepy.video.io.VideoFileClip import VideoFileClip
import torchvision
import vggish_input
import pandas
import soundfile as sf
from tqdm import tqdm
import librosa
from torchaudio import transforms
from efficientnet_pytorch import EfficientNet
import torchaudio

panns = False
VGGish = True
psla = False
save_data_audio = '/dssg/home/acct-stu/stu519/av_scene_classify_finetune/data/raw_data/val_pslaaudio.hdf5'
save_data_video = '/dssg/home/acct-stu/stu519/av_scene_classify_finetune/data/raw_data/test_video-data_new.hdf5'
path_csv = '/dssg/home/acct-stu/stu464/data/audio_visual_scenes/evaluation_setup/fold1_test.csv'
dataset_path = '/dssg/home/acct-stu/stu464/data/audio_visual_scenes'
df = pandas.read_csv(path_csv, sep="\t")

input_dir_audios = df['filename_audio'].values
input_dir_audios = list(input_dir_audios)
input_dir_audios.sort()

TS = 10

# transform0 = torchvision.transforms.ToTensor()
# transform1 = torchvision.transforms.CenterCrop(224)
# transform2 = torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

# hf_audio = h5py.File(save_data_audio, 'w')


SR = 16000
atf = transforms.Resample(SR, 16000)

def wav2fbank(first_sample, audio_transform):
    audio, SR = torchaudio.load(first_sample)
    # 重采样
    audio = audio_transform(audio)
    SR = 16000
    # 归一
    audio = audio - audio.mean()
    fbank = torchaudio.compliance.kaldi.fbank(audio, htk_compat=True, sample_frequency=SR, use_energy=False,
                                              window_type='hanning', num_mel_bins=128, dither=0.0, frame_shift=10)
    target_length = 1000
    n_frames = fbank.shape[0]
    p = target_length - n_frames

    # cut and pad
    if p > 0:
        m = torch.nn.ZeroPad2d((0, 0, 0, p))
        fbank = m(fbank)
    elif p < 0:
        fbank = fbank[0:target_length, :]
    return fbank.numpy()

if psla:
    for index in tqdm(range(len(input_dir_audios))):
        input_dir_audio = input_dir_audios[index]
        audio_name = os.path.join(dataset_path, input_dir_audio)

        fbank = wav2fbank(audio_name, atf)

        fname = audio_name.split('/')[-1].split('.')[0]
        hf_audio.create_dataset(fname, data=fbank)

if panns:
    for index in tqdm(range(len(input_dir_audios))):
        input_dir_audio = input_dir_audios[index]
        audio_name = os.path.join(dataset_path, input_dir_audio)

        audio, rate = librosa.load(audio_name)
        audio = (np.array(audio).astype(np.float32))

        fname = audio_name.split('/')[-1].split('.')[0]
        hf_audio.create_dataset(fname, data=audio)

if VGGish:
    hf_video = h5py.File(save_data_video, 'w')
    mix_transforms = torchvision.transforms.Compose([
        torchvision.transforms.ToPILImage(),
        torchvision.transforms.Resize(256, torchvision.transforms.InterpolationMode.BICUBIC),
        torchvision.transforms.CenterCrop(224),
        torchvision.transforms.ToTensor(),
    ])
    for index in tqdm(range(len(input_dir_audios))):
        input_dir_audio = input_dir_audios[index]
        audio_name = os.path.join(dataset_path, input_dir_audio)

        # audio, sr = sf.read(audio_name, dtype='int16')
        # assert audio.dtype == np.int16, 'Bad sample type: %r' % audio.dtype
        # audio = audio / 32768.0  # Convert to [-1.0, +1.0]
        # audio = vggish_input.waveform_to_examples(audio, sr, False)

        fname = audio_name.split('/')[-1].split('.')[0]
        # hf_audio.create_dataset(fname, data=audio)

        video_name = input_dir_audio.replace('audio', 'video')
        video_name = video_name.replace('.wav', '.mp4')
        video_name = os.path.join(dataset_path, video_name)
        with VideoFileClip(video_name, audio=False) as clip:
            images = []
            for t, frame in clip.iter_frames(with_times=True):
                # (720, 1280, 3) --> (3, 720, 1280)
                # frame = frame.astype(np.float16)
                # frame = torch.tensor(frame).permute(2,0,1)
                images.append(frame)
            index = np.linspace(0, len(images) - 1, TS)
            index = index.astype(int)
            images = [images[i] for i in list(index)]
            for i in range(len(images)):
                images[i] = mix_transforms(images[i])
                # images[i] = transform0(images[i])
                # images[i] = transform1(images[i])
                # images[i] = transform2(images[i])
                images[i] = images[i].numpy()
            video = np.array(images)
        hf_video.create_dataset(fname, data=video)

