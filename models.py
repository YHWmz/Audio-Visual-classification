import torch
import torch.nn as nn
import vggish
from efficientnet_pytorch import EfficientNet
import psla_models

class MeanConcatDense(nn.Module):

    def __init__(self, audio_emb_dim, video_emb_dim, num_classes) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.audio_embed = nn.Sequential(
            nn.Linear(audio_emb_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(512, 128)
        )
        self.video_embed = nn.Sequential(
            nn.Linear(video_emb_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(512, 128)
        )
        self.outputlayer = nn.Sequential(
            nn.Linear(256, 128),
            nn.Linear(128, self.num_classes),
        )
    
    def forward(self, audio_feat, video_feat):
        # audio_feat: [batch_size, time_steps, feat_dim]
        # video_feat: [batch_size, time_steps, feat_dim]
        audio_emb = audio_feat.mean(1)
        audio_emb = self.audio_embed(audio_emb)

        video_emb = video_feat.mean(1)
        video_emb = self.video_embed(video_emb)
        
        embed = torch.cat((audio_emb, video_emb), 1)
        output = self.outputlayer(embed)
        return output

class FintuneModel(nn.Module):
    def __init__(self):
        super(FintuneModel, self).__init__()
        urls = {
            'vggish': "https://github.com/harritaylor/torchvggish/releases/download/v0.1/vggish-10086976.pth"
        }
        self.pretrain = vggish.VGGish(urls, preprocess=False, postprocess=False)
        # self.classifier = classifier()

    def forward(self, x):
        """
        :param x: [bs, num_frames, 96, 64]
        :return:
        """
        bs, num_frames, _, _ = x.size()
        x = x.view(bs*num_frames, 1, x.size(2), x.size(3))
        x = self.pretrain(x) # [bs*num_frames, 128]
        # y = x.view(bs, x.size(1), num_frames)
        x = x.view(bs, num_frames, x.size(1))
        # x = self.classifier(x)
        return x

class VGGbase_audiodense(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        urls = {
            'vggish': "https://github.com/harritaylor/torchvggish/releases/download/v0.1/vggish-10086976.pth"
        }
        self.pretrain = vggish.VGGish(urls, pretrained=True, preprocess=False, postprocess=False)
        self.num_classes = num_classes
        # for p in self.pretrain.parameters():  # 将需要冻结的参数的 requires_grad 设置为 False
        #     p.requires_grad = False
        self.audio_embed = nn.Sequential(
            nn.Linear(1280, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(512, 256)
        )
        self.outputlayer = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, self.num_classes),
        )
        # self.audio_cls = nn.Sequential(
        #     nn.Linear(1280, 128),
        #     nn.ReLU(),
        #     nn.Linear(128, self.num_classes),
        # )

    def forward(self, x):
        ''':
        input: [N, 10, 96, 64]
        output: []
        '''
        bs, num_frames, _, _ = x.size()
        x = x.view(bs * num_frames, 1, x.size(2), x.size(3))
        x = self.pretrain(x)  # [bs*num_frames, 128]
        x = x.view(bs, num_frames, x.size(1))  # [bs, 10, 128]
        x = x.view(bs, -1)  # [bs, 1280]

        # x = self.audio_cls(x) # [bs, 256]
        x = self.audio_embed(x)
        x = self.outputlayer(x) # [bs, 10]
        return x

from Net.models import *
from Net.transfer_model import *

def BuiltModel(device):
    # Arugments & parameters
    sample_rate = 22050
    window_size = 1024
    hop_size = 441
    mel_bins = 64
    fmin = 50
    fmax = 14000
    model_type = "Transfer_ResNet38"
    pretrained_checkpoint_path = "/dssg/home/acct-stu/stu519/av_scene_classify_finetune/ModelWeight/ResNet38_mAP=0.434.pth"

    pretrain = True if pretrained_checkpoint_path else False
    classes_num = 10

    # Model
    Model = eval(model_type)
    model = Model(sample_rate, window_size, hop_size, mel_bins, fmin, fmax)

    # Load pretrained model
    if pretrain:
        model.load_from_pretrain(pretrained_checkpoint_path)

    torch.backends.cudnn.benchmark = True
    model.to(device)

    print('Load pretrained model successfully!')
    return model


class Pannsbase_audiodense(nn.Module):
    def __init__(self, num_classes, device, deeper=True):
        super().__init__()

        self.pretrain = BuiltModel(device)
        print('Finish PANNs loading')
        self.num_classes = num_classes
        self.deeper = deeper

        self.audio_cls = nn.Sequential(
            nn.Linear(2048, 1024),
                nn.ReLU(),
            nn.Linear(1024, self.num_classes),
        )
        self.audio_embed = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(1024, 512)
        )
        self.outputlayer = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, self.num_classes),
        )

    def forward(self, x):
        ''':
        input: [N, 480000]
        output: []
        '''
        x = self.pretrain.forward(x) # [N, 2048]
        if self.deeper:
            x = self.audio_embed(x)
            x = self.outputlayer(x)  # [bs, 10]
        else:
            x = self.audio_cls(x) # [bs, 256]

        return x

    def extract_feature(self, x):
        x = self.pretrain.forward(x)  # [N, 2048]

        return x

import models

class Pslabase_audiodense(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.deeper = True
        self.num_classes = num_classes
        model_path = '/dssg/home/acct-stu/stu519/psla/pretrained_models/audioset/as_mdl_0.pth'
        num_class = 527 if 'audioset' in model_path else 200

        self.pretrain = psla_models.EffNetAttention(label_dim=num_class, b=2, pretrain=False, head_num=4)
        self.pretrain = torch.nn.DataParallel(self.pretrain)
        self.pretrain.load_state_dict(torch.load(model_path), strict=False)
        print('Finish psla loading')
        self.audio_cls = nn.Sequential(
            nn.Linear(1408, 512),
                nn.ReLU(),
            nn.Linear(512, self.num_classes),
        )
        self.audio_embed = nn.Sequential(
            nn.Linear(1408, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(1024, 512)
        )
        self.outputlayer = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, self.num_classes),
        )


    def forward(self, x):
        x, _ = self.pretrain(x)# [N, 32, 1408]
        x = x.mean(1)

        if self.deeper:
            x = self.audio_embed(x)
            x = self.outputlayer(x)  # [bs, 10]
        else:
            x = self.audio_cls(x) # [bs, 256]
        return x

import torchvision

class Effi_vediodense(nn.Module):
    def __init__(self, num_classes, pre_name="efficientnet"):
        super().__init__()
        self.deeper = True
        self.num_classes = num_classes
        self.pretrain_name = pre_name
        if "efficient" in self.pretrain_name:
            self.pretrain = EfficientNet.from_pretrained('efficientnet-b0')
        else:
            self.pretrain = ResNetBackbone()
        self._avg_pooling = nn.AdaptiveAvgPool2d(1)

        self.vedio_cls = nn.Sequential(
            nn.Linear(1280, 512),
                nn.ReLU(),
            nn.Linear(512, self.num_classes),
        )
        self.vedio_embed = nn.Sequential(
            nn.Linear(1280, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(1024, 512)
        )
        self.outputlayer = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, self.num_classes),
        )

    def forward(self, x):
        # [N, 10, 3, 224, 224]
        # print(x.shape)
        N, T, C, W, H = x.shape
        x = x.view(N*T, C, W, H)    # [N*10, 3, 224, 224]
        # print(x.shape)
        x = self.pretrain.extract_features(x)# [N*10, 1280, 7, 7]
        if "efficient" in self.pretrain_name:
            x = self._avg_pooling(x).squeeze()    # [N*10, 1280]
            # print(x.shape)
        x = x.view(N, T, x.shape[-1])       # [N, 10, 1280]
        x = x.mean(1)


        if self.deeper:
            x = self.vedio_embed(x)
            x = self.outputlayer(x)  # [bs, 10]
        else:
            x = self.vedio_cls(x) # [bs, 256]
        return x

    def extract_feature(self, x):
        # [N, 10, 3, 224, 224]
        # print(x.shape)
        N, T, C, W, H = x.shape
        x = x.view(N * T, C, W, H)  # [N*10, 3, 224, 224]
        # print(x.shape)
        x = self.pretrain.extract_features(x)  # [N*10, 1280, 7, 7]
        if "efficient" in self.pretrain_name:
            x = self._avg_pooling(x).squeeze()  # [N*10, 1280]
            # print(x.shape)
        x = x.view(N, T, x.shape[-1])  # [N, 10, 1280]
        x = x.mean(1)

        return x


class ResNetBackbone(nn.Module):
    def __init__(self):
        super(ResNetBackbone, self).__init__()
        resnet = torchvision.models.resnet50(pretrained=True)
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        self.avgpool = resnet.avgpool
        self._feature_dim = resnet.fc.in_features
        print(self._feature_dim)
        del resnet

    def extract_features(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return x


import h5py
class earlyfusion_dense(nn.Module):
    def __init__(self, num_classes, device, mode=0):
        super().__init__()
        self.mode = mode
        self.mode = mode
        self.device = device
        self.num_classes = num_classes
        self.video_model = Effi_vediodense(num_classes)
        # self.video_model.load_state_dict(torch.load(
        #     '/dssg/home/acct-stu/stu519/av_scene_classify_finetune/experiments/efficientnetb0/best_model.pt', "cpu")
        # )
        self.audio_model = Pannsbase_audiodense(num_classes, device)
        # self.audio_model.load_state_dict(torch.load(
        #     '/dssg/home/acct-stu/stu519/av_scene_classify_finetune/experiments/panns/best_model.pt', "cpu")
        # )

        if self.mode:
            self.audio_embed = nn.Sequential(
                nn.Linear(2048, 512),
                nn.BatchNorm1d(512),
                nn.ReLU(),
            )
            self.video_embed = nn.Sequential(
                nn.Linear(1280, 512),
                nn.BatchNorm1d(512),
                nn.ReLU(),
            )
            self.outputlayer = nn.Sequential(
                nn.Linear(1024, self.num_classes),
            )
        else:
            self.outputlayer = nn.Sequential(
                nn.Linear(1280+2048, 512),
                nn.BatchNorm1d(512),
                nn.ReLU(),
                nn.Linear(512, self.num_classes),
            )


    def forward(self, audio_x, video_x):
        # [N, 10, 3, 224, 224]
        # print(x.shape)
        video_emb = self.video_model.extract_feature(video_x)# [N, 512]
        audio_emb = self.audio_model.extract_feature(audio_x)# [N, 512]
        # hf = h5py.File('/dssg/home/acct-stu/stu519/av_scene_classify_finetune/data/extrashdata.hdf5', 'w')
        # hf.create_dataset("video", data=video_x[0].detach().cpu().numpy())
        # hf.create_dataset("audio", data=audio_x[0].detach().cpu().numpy())
        # print(audio_emb[0][0:20])
        # print(video_emb[0][0:20])
        # exit(0)

        if self.mode:
            audio_emb = self.audio_embed(audio_emb)
            video_emb = self.video_embed(video_emb)

        embed = torch.cat((audio_emb, video_emb), 1)
        output = self.outputlayer(embed)
        return output

    def extract_feature(self, audio_x, video_x):
        # [N, 10, 3, 224, 224]
        # print(x.shape)
        video_emb = self.video_model.extract_feature(video_x)  # [N, 2048]
        audio_emb = self.audio_model.extract_feature(audio_x)  # [N, 1280]

        return audio_emb, video_emb


class Latefusion_dense(nn.Module):
    def __init__(self, num_classes, device, mode=0):
        super().__init__()
        self.deeper = True
        self.mode = mode
        self.num_classes = num_classes
        self.video_model = Effi_vediodense(num_classes)
        self.video_model.load_state_dict(torch.load(
            '/dssg/home/acct-stu/stu519/av_scene_classify_finetune/experiments/efficientnetb0/best_model.pt', "cpu")
        )
        self.audio_model = Pannsbase_audiodense(num_classes, device)
        self.audio_model.load_state_dict(torch.load(
            '/dssg/home/acct-stu/stu519/av_scene_classify_finetune/experiments/panns/best_model.pt', "cpu")
        )
        if self.mode == 2:
            self.class_weight = nn.Parameter(torch.ones(10)*0.5)

    def forward(self, audio_x, video_x):
        # [N, 10, 3, 224, 224]
        # print(x.shape)
        video_out = self.video_model(video_x)
        audio_out = self.audio_model(audio_x)
        if self.mode == 0:
            output = video_out + audio_out
        elif self.mode == 1:
            output = torch.max(video_out, audio_out)
        elif self.mode == 2:
            output = video_out*self.class_weight + audio_out*(1-self.class_weight)

        return output
