import torch
torch.backends.cudnn.benchmark=True
torch.manual_seed(0)
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
from tqdm import tqdm
from Net.models import *
import numpy as np

class Transfer_ResNet38_formix(nn.Module):
    def __init__(self, sample_rate, window_size, hop_size, mel_bins, fmin, 
        fmax, classes_num, layer_fc = 2, freeze_base = None):
        """Classifier for a new task using pretrained Cnn14 as a sub module.
        """
        super(Transfer_ResNet38_formix, self).__init__()
        audioset_classes_num = 527
        
        self.base = ResNet38_formix(sample_rate, window_size, hop_size, mel_bins, fmin, 
            fmax, audioset_classes_num)

        self.layer_fc = layer_fc
        # Transfer to another task layer
        if(layer_fc == 1):
            self.fc_transfer = nn.Linear(2048, classes_num, bias=True)
        else:
            self.fc_transfer1 = nn.Linear(2048, 512, bias=True)
            self.fc_transfer2 = nn.Linear(512, classes_num, bias=True)

        if freeze_base:
            # Freeze AudioSet pretrained layers
            for param in self.base.parameters():
                param.requires_grad = False

        self.init_weights()

    def init_weights(self):
        if(self.layer_fc == 1):
            init_layer(self.fc_transfer)
        else:
            init_layer(self.fc_transfer1)
            init_layer(self.fc_transfer2)

    def load_from_pretrain(self, pretrained_checkpoint_path):
        checkpoint = torch.load(pretrained_checkpoint_path)
        self.base.load_state_dict(checkpoint['model'])

    def forward(self, input, index, mix_up=0):
        """Input: (batch_size, data_length)
        """
        #input = torch.FloatTensor(np.array(input).astype(np.float32)).cuda()

        output_dict, lam = self.base(input, index, mix_up)
        embedding = output_dict['embedding']

        if(self.layer_fc == 1):
            #clipwise_output =  torch.log_softmax(self.fc_transfer(embedding), dim=-1)
            clipwise_output =  self.fc_transfer(embedding)
        else:
            output1 = torch.relu(self.fc_transfer1(embedding))
            #clipwise_output =  torch.log_softmax(self.fc_transfer2(output1), dim=-1)
            clipwise_output =  self.fc_transfer2(output1)
        output_dict['result'] = clipwise_output

        return output_dict['result'], lam

class Transfer_ResNet54(nn.Module):
    def __init__(self, sample_rate, window_size, hop_size, mel_bins, fmin, 
        fmax, classes_num, layer_fc = 2, freeze_base = None):
        """Classifier for a new task using pretrained Cnn14 as a sub module.
        """
        super(Transfer_ResNet54, self).__init__()
        audioset_classes_num = 527
        
        self.base = ResNet54(sample_rate, window_size, hop_size, mel_bins, fmin, 
            fmax, audioset_classes_num)

        self.layer_fc = layer_fc
        # Transfer to another task layer
        if(layer_fc == 1):
            self.fc_transfer = nn.Linear(2048, classes_num, bias=True)
        else:
            self.fc_transfer1 = nn.Linear(2048, 512, bias=True)
            self.fc_transfer2 = nn.Linear(512, classes_num, bias=True)

        if freeze_base:
            # Freeze AudioSet pretrained layers
            for param in self.base.parameters():
                param.requires_grad = False

        self.init_weights()

    def init_weights(self):
        if(self.layer_fc == 1):
            init_layer(self.fc_transfer)
        else:
            init_layer(self.fc_transfer1)
            init_layer(self.fc_transfer2)

    def load_from_pretrain(self, pretrained_checkpoint_path):
        checkpoint = torch.load(pretrained_checkpoint_path)
        self.base.load_state_dict(checkpoint['model'])

    def forward(self, input, mixup_lambda=None):
        """Input: (batch_size, data_length)
        """
        #input = torch.FloatTensor(np.array(input).astype(np.float32)).cuda()
        output_dict = self.base(input, mixup_lambda)
        embedding = output_dict['embedding']

        if(self.layer_fc == 1):
            #clipwise_output =  torch.log_softmax(self.fc_transfer(embedding), dim=-1)
            clipwise_output =  self.fc_transfer(embedding)
        else:
            output1 = torch.relu(self.fc_transfer1(embedding))
            #clipwise_output =  torch.log_softmax(self.fc_transfer2(output1), dim=-1)
            clipwise_output =  self.fc_transfer2(output1)
        output_dict['result'] = clipwise_output
 
        return output_dict

class Transfer_ResNet22(nn.Module):
    def __init__(self, sample_rate, window_size, hop_size, mel_bins, fmin, 
        fmax, classes_num, layer_fc = 2, freeze_base = None):
        """Classifier for a new task using pretrained Cnn14 as a sub module.
        """
        super(Transfer_ResNet22, self).__init__()
        audioset_classes_num = 527
        
        self.base = ResNet22(sample_rate, window_size, hop_size, mel_bins, fmin, 
            fmax, audioset_classes_num)

        self.layer_fc = layer_fc
        # Transfer to another task layer
        if(layer_fc == 1):
            self.fc_transfer = nn.Linear(2048, classes_num, bias=True)
        else:
            self.fc_transfer1 = nn.Linear(2048, 512, bias=True)
            self.fc_transfer2 = nn.Linear(512, classes_num, bias=True)

        if freeze_base:
            # Freeze AudioSet pretrained layers
            for param in self.base.parameters():
                param.requires_grad = False

        self.init_weights()

    def init_weights(self):
        if(self.layer_fc == 1):
            init_layer(self.fc_transfer)
        else:
            init_layer(self.fc_transfer1)
            init_layer(self.fc_transfer2)

    def load_from_pretrain(self, pretrained_checkpoint_path):
        checkpoint = torch.load(pretrained_checkpoint_path)
        self.base.load_state_dict(checkpoint['model'])

    def forward(self, input, mixup_lambda=None):
        """Input: (batch_size, data_length)
        """
        #input = torch.FloatTensor(np.array(input).astype(np.float32)).cuda()
        output_dict = self.base(input, mixup_lambda)
        embedding = output_dict['embedding']

        if(self.layer_fc == 1):
            #clipwise_output =  torch.log_softmax(self.fc_transfer(embedding), dim=-1)
            clipwise_output =  self.fc_transfer(embedding)
        else:
            output1 = torch.relu(self.fc_transfer1(embedding))
            #clipwise_output =  torch.log_softmax(self.fc_transfer2(output1), dim=-1)
            clipwise_output =  self.fc_transfer2(output1)
        output_dict['result'] = clipwise_output
 
        return output_dict

class Transfer_CNN14(nn.Module):
    def __init__(self, sample_rate, window_size, hop_size, mel_bins, fmin, 
        fmax, classes_num = 10, layer_fc = 2, freeze_base = None):
        """Classifier for a new task using pretrained Cnn14 as a sub module.
        """
        super(Transfer_CNN14, self).__init__()
        audioset_classes_num = 527
        
        self.base = Cnn14_emb512(sample_rate, window_size, hop_size, mel_bins, fmin, 
            fmax, audioset_classes_num)

        self.layer_fc = layer_fc
        # Transfer to another task layer
        if(layer_fc == 1):
            self.fc_transfer = nn.Linear(512, classes_num, bias=True)
        else:
            self.fc_transfer1 = nn.Linear(512, 128, bias=True)
            self.fc_transfer2 = nn.Linear(128, classes_num, bias=True)

        if freeze_base:
            # Freeze AudioSet pretrained layers
            for param in self.base.parameters():
                param.requires_grad = False

        self.init_weights()

    def init_weights(self):
        if(self.layer_fc == 1):
            init_layer(self.fc_transfer)
        else:
            init_layer(self.fc_transfer1)
            init_layer(self.fc_transfer2)

    def load_from_pretrain(self, pretrained_checkpoint_path):
        checkpoint = torch.load(pretrained_checkpoint_path)
        self.base.load_state_dict(checkpoint['model'])

    def forward(self, input, mixup_lambda=None):
        """Input: (batch_size, data_length)
        """
        #input = torch.FloatTensor(np.array(input).astype(np.float32)).cuda()
        output_dict = self.base(input, mixup_lambda)
        embedding = output_dict['embedding']

        if(self.layer_fc == 1):
            #clipwise_output =  torch.log_softmax(self.fc_transfer(embedding), dim=-1)
            clipwise_output =  self.fc_transfer(embedding)
        else:
            output1 = torch.relu(self.fc_transfer1(embedding))
            #clipwise_output =  torch.log_softmax(self.fc_transfer2(output1), dim=-1)
            clipwise_output =  self.fc_transfer2(output1)
        output_dict['result'] = clipwise_output
 
        return output_dict

class Transfer_Wavegram(nn.Module):
    def __init__(self, sample_rate, window_size, hop_size, mel_bins, fmin, fmax):
        """Classifier for a new task using pretrained Cnn14 as a sub module.
        """
        super(Transfer_Wavegram, self).__init__()
        audioset_classes_num = 527
        
        self.base = Wavegram_Logmel_Cnn14(sample_rate, window_size, hop_size, mel_bins, fmin, 
            fmax, audioset_classes_num)

    def load_from_pretrain(self, pretrained_checkpoint_path):
        #transfer_model(pretrained_checkpoint_path, self.base)
        checkpoint = torch.load(pretrained_checkpoint_path)
        self.base.load_state_dict(checkpoint['model'])

    def forward(self, input):
        """Input: (batch_size, data_length)
        """
        output_dict = self.base(input)
        embedding = output_dict['embedding']
 
        return embedding


class Transfer_Fusion(nn.Module):
    def __init__(self, sample_rate, window_size, hop_size, mel_bins, fmin,
                 fmax, classes_num, layer_fc=2, freeze_base=None):
        """Classifier for a new task using pretrained Cnn14 as a sub module.
        """
        super(Transfer_Fusion, self).__init__()
        audioset_classes_num = 527

        self.base_wave = Wavegram_Logmel_Cnn14(sample_rate, window_size, hop_size, mel_bins, fmin,
                                          fmax, audioset_classes_num)
        self.base_cnn = Cnn10(self, sample_rate, window_size, hop_size, mel_bins, fmin, fmax, classes_num)

        self.layer_fc = layer_fc
        # Transfer to another task layer
        if (layer_fc == 1):
            self.fc_transfer = nn.Linear(2048+512, classes_num, bias=True)
        else:
            self.fc_transfer1 = nn.Linear(2048+512, 4096, bias=True)
            self.fc_transfer2 = nn.Linear(4096, 2048, bias=True)
            self.fc_transfer3 = nn.Linear(2048, 1024, bias=True)
            self.fc_transfer4 = nn.Linear(1024, 10, bias=True)

        if freeze_base:
            # Freeze AudioSet pretrained layers
            for param in self.base.parameters():
                param.requires_grad = False

        self.init_weights()

    def init_weights(self):
        if (self.layer_fc == 1):
            init_layer(self.fc_transfer)
        else:
            init_layer(self.fc_transfer1)
            init_layer(self.fc_transfer2)
            init_layer(self.fc_transfer3)
            init_layer(self.fc_transfer4)

    def load_from_pretrain(self, Wave_path, Cnn_path):
        transfer_model(Wave_path, self.base_wave)
        transfer_model(Cnn_path, self.base_cnn)
        #checkpoint = torch.load(pretrained_checkpoint_path)
        #self.base.load_state_dict(checkpoint['model'])

    def forward(self, input, mixup_lambda=None):
        """Input: (batch_size, data_length)
        """
        # input = torch.FloatTensor(np.array(input).astype(np.float32)).cuda()
        output_dict = self.base_wave(input, mixup_lambda)
        _, embedding_cnn = self.base_cnn(input)
        embedding_wave = output_dict['embedding']

        embedding = torch.cat((embedding_wave, embedding_cnn), dim=1)

        if (self.layer_fc == 1):
            # clipwise_output =  torch.log_softmax(self.fc_transfer(embedding), dim=-1)
            clipwise_output = self.fc_transfer(embedding)
        else:
            output1 = torch.relu(self.fc_transfer1(embedding))
            output1 = torch.relu(self.fc_transfer2(output1))
            output1 = torch.relu(self.fc_transfer3(output1))
            # clipwise_output =  torch.log_softmax(self.fc_transfer2(output1), dim=-1)
            clipwise_output = self.fc_transfer4(output1)

        return clipwise_output

def transfer_model(pretrained_file, model):
    '''
    只导入pretrained_file部分模型参数
    tensor([-0.7119,  0.0688, -1.7247, -1.7182, -1.2161, -0.7323, -2.1065, -0.5433,-1.5893, -0.5562]
    update:
        D.update([E, ]**F) -> None.  Update D from dict/iterable E and F.
        If E is present and has a .keys() method, then does:  for k in E: D[k] = E[k]
        If E is present and lacks a .keys() method, then does:  for k, v in E: D[k] = v
        In either case, this is followed by: for k in F:  D[k] = F[k]
    :param pretrained_file:
    :param model:
    :return:
    '''
    pretrained_dict = torch.load(pretrained_file,'cpu')  # get pretrained dict
    model_dict = model.state_dict()  # get model dict
    # 在合并前(update),需要去除pretrained_dict一些不需要的参数
    pretrained_dict = transfer_state_dict(pretrained_dict, model_dict)
    model_dict.update(pretrained_dict)  # 更新(合并)模型的参数
    model.load_state_dict(model_dict)
    return model


def transfer_state_dict(pretrained_dict, model_dict):
    '''
    根据model_dict,去除pretrained_dict一些不需要的参数,以便迁移到新的网络
    url: https://blog.csdn.net/qq_34914551/article/details/87871134
    :param pretrained_dict:
    :param model_dict:
    :return:
    '''
    # state_dict2 = {k: v for k, v in save_model.items() if k in model_dict.keys()}
    state_dict = {}
    for k, v in pretrained_dict.items():
        # if k[8:11]=='bn0':
        #     continue
        if k[8:] in model_dict.keys():
            # state_dict.setdefault(k, v)
            state_dict[k[8:]] = v
            print(k)
        else:
            print("Missing key(s) in state_dict :{}".format(k))
    return state_dict

class Transfer_2stage_Res38(nn.Module):
    def __init__(self, sample_rate, window_size, hop_size, mel_bins, fmin, 
        fmax, classes_num, layer_fc = 2, freeze_base = None):
        """Classifier for a new task using pretrained Cnn14 as a sub module.
        """
        super(Transfer_2stage_Res38, self).__init__()
        audioset_classes_num = 527
        
        self.base = ResNet38_formix(sample_rate, window_size, hop_size, mel_bins, fmin, 
            fmax, audioset_classes_num)

        self.layer_fc = layer_fc
        # Transfer to another task layer
        if(layer_fc == 1):
            self.fc_transfer = nn.Linear(2048, classes_num, bias=True)
        else:
            self.fc_transfer1 = nn.Linear(2048, 512, bias=True)
            self.fc_transfer2 = nn.Linear(512, classes_num, bias=True)
            self.fc_transfer3 = nn.Linear(2048, 512, bias=True)
            self.fc_transfer4 = nn.Linear(512, 3, bias=True)

        if freeze_base:
            # Freeze AudioSet pretrained layers
            for param in self.base.parameters():
                param.requires_grad = False

        self.init_weights()

    def init_weights(self):
        if(self.layer_fc == 1):
            init_layer(self.fc_transfer)
        else:
            init_layer(self.fc_transfer1)
            init_layer(self.fc_transfer2)
            init_layer(self.fc_transfer3)
            init_layer(self.fc_transfer4)

    def load_from_pretrain(self, pretrained_checkpoint_path):
        checkpoint = torch.load(pretrained_checkpoint_path)
        self.base.load_state_dict(checkpoint['model'])

    def forward(self, input, index, mix_up=0):
        """Input: (batch_size, data_length)
        """
        #input = torch.FloatTensor(np.array(input).astype(np.float32)).cuda()
        output_dict, lam = self.base(input, index, mix_up)
        embedding = output_dict['embedding']

        if(self.layer_fc == 1):
            #clipwise_output =  torch.log_softmax(self.fc_transfer(embedding), dim=-1)
            clipwise_output =  self.fc_transfer(embedding)
        else:
            output1 = torch.relu(self.fc_transfer1(embedding))
            #clipwise_output =  torch.log_softmax(self.fc_transfer2(output1), dim=-1)
            clipwise_output =  self.fc_transfer2(output1)
            output2 = torch.relu(self.fc_transfer3(embedding))
            clipwise_output1 =  self.fc_transfer4(output2)
        output_dict['10-class'] = clipwise_output
        output_dict['3-class'] = clipwise_output1

        return output_dict, lam

def to_one_hot(inp, num_classes):
    y_onehot = torch.FloatTensor(inp.size(0), num_classes).to(inp.device)
    y_onehot.zero_()
    y_onehot.scatter_(1, inp.unsqueeze(1).data, 1)
    return y_onehot

bce_loss = nn.BCELoss()
softmax = nn.Softmax(dim=1)
ce_loss = nn.CrossEntropyLoss()

class ManifoldMixupModel(nn.Module):
    def __init__(self, model, num_classes = 10, alpha = 0, mixup_layer = 0):
        super().__init__()
        self.model = model
        self.alpha = alpha
        self.lam = None
        self.num_classes = num_classes
        self. k = mixup_layer
        ##选择需要操作的层，在ResNet中各block的层名为layer1,layer2...所以可以写成如下。其他网络请自行修改
        self. module_list = []
        #base.resnet.layer1.2.bn2
        self.mix_layer = ['base.logmel_extractor',  'base.conv_block1',\
            'base.resnet.layer1','base.resnet.layer2', 'base.resnet.layer3',\
            'base.resnet.layer4', 'base.conv_block_after1','base.conv_block2',\
            'base.conv_block3','base.conv_block4', 'base.conv_block5', 'base.conv_block6']
        for n,m in self.model.named_modules():
            #if 'conv' in n:
            if n in self.mix_layer:
                print(n)
                self.module_list.append(m)

    def forward(self, x, target=None, target2=None):
        # 测试阶段target = None
        if target==None:
            out = self.model(x)
            return out
        else:
            if self.alpha <= 0:
                self.lam = 1
            else:
                np.random.seed(123)
                self.lam = np.random.beta(self.alpha, self.alpha)
            
            k = self.k
            # k = np.random.randint(0, len(self.module_list))
            self.indices = torch.randperm(target.size(0)).cuda()
            target_onehot = to_one_hot(target, self.num_classes)
            target_shuffled_onehot = target_onehot[self.indices]
            
            if k == -1:
                x = x * self.lam + x[self.indices] * (1 - self.lam)
                out = self.model(x)

            elif k==100:
                modifier_hook = self.module_list[2].register_forward_pre_hook(self.hook_pre)
                out = self.model(x)
                modifier_hook.remove()

            else:
                modifier_hook = self.module_list[k].register_forward_hook(self.hook_modify)
                #modifier_hook2 = self.module_list[1].register_forward_hook(self.hook_modify)
                out = self.model(x)
                modifier_hook.remove()
                #modifier_hook2.remove()
            target_reweighted = target_onehot* self.lam + target_shuffled_onehot * (1 - self.lam)
            #target_mix = torch.cat((target_onehot,target_reweighted))
            
            #loss = bce_loss(softmax(out), target_reweighted)
            #loss = bce_loss(softmax(out), target_mix)

            if target2==None:
                size = out.shape[0]
                loss = ce_loss(out[0:size//2], target) + self.lam * ce_loss(out[size//2:], target) + (1-self.lam) * ce_loss(out[size//2:], target[self.indices])
            else:
                size = out['class'].shape[0]
                loss = ce_loss(out['class'][0:size//2], target) + self.lam * ce_loss(out['class'][size//2:], target) \
                       + (1-self.lam) * ce_loss(out['class'][size//2:], target[self.indices])\
                       + ce_loss(out['domain'][0:size//2], target2) + self.lam * ce_loss(out['domain'][size//2:], target2) \
                       + (1-self.lam) * ce_loss(out['domain'][size//2:], target2[self.indices])
            return out, loss

    def hook_pre(self, module, inputs):
        return self.hook_modify(module, None, inputs[0])

    def hook_modify(self, module, input, output):
        output_mix = self.lam * output + (1 - self.lam) * output[self.indices]
        output = torch.cat((output, output_mix)).cuda()
        return output




class Transfer_ResNet38(nn.Module):
    def __init__(self, sample_rate, window_size, hop_size, mel_bins, fmin, 
        fmax):
        """Classifier for a new task using pretrained Cnn14 as a sub module.
        """
        super(Transfer_ResNet38, self).__init__()
        audioset_classes_num = 527
        
        self.base = ResNet38(sample_rate, window_size, hop_size, mel_bins, fmin, 
            fmax, audioset_classes_num)
        # Transfer to another task layer


    def load_from_pretrain(self, pretrained_checkpoint_path):
        checkpoint = torch.load(pretrained_checkpoint_path)
        self.base.load_state_dict(checkpoint['model'])

    def forward(self, input):
        """Input: (batch_size, data_length)
        """
        #input = torch.FloatTensor(np.array(input).astype(np.float32)).cuda()
        output_dict = self.base(input)

        return output_dict['embedding']

class PatchUpModel(nn.Module):
    def __init__(self, model, num_classes = 10, block_size=7, gamma=.9, patchup_type='hard',keep_prob=.9, mixup_layer=1):
        super().__init__()
        self.patchup_type = patchup_type
        self.block_size = block_size
        self.gamma = gamma
        self.gamma_adj = None
        self.kernel_size = (block_size, block_size)
        self.stride = (1, 1)
        self.padding = (block_size // 2, block_size // 2)
        self.computed_lam = None
        self.k = mixup_layer
        
        self.model = model
        self.num_classes = num_classes
        self. module_list = []
        self.mix_layer = ['base.logmel_extractor','base.conv_block1',\
             'base.resnet.layer1','base.resnet.layer2', 'base.resnet.layer3',\
                 'base.resnet.layer4', 'base.conv_block_after1', 'base.conv_block2',\
            'base.conv_block3','base.conv_block4', 'base.conv_block5', 'base.conv_block6']
        for n,m in self.model.named_modules():
            #if 'conv' in n:
            if n in self.mix_layer:
                print(n)
                self.module_list.append(m)

    def adjust_gamma(self, x):
        #return self.gamma * x.shape[-1] ** 2 / \
        #       (self.block_size ** 2 * (x.shape[-1] - self.block_size + 1) ** 2)
        return self.gamma * x.shape[-1] * x.shape[-2] / \
               (self.block_size ** 2 * (x.shape[-1] - self.block_size + 1) * (x.shape[-2] - self.block_size + 1))

    def forward(self, x, target=None):
        if target==None:
            out = self.model(x)
            return out
        else:
            self.lam = np.random.beta(2.0, 2.0)
            k = self.k 
            #k = np.random.randint(0, 5)
            self.indices = torch.randperm(target.size(0)).cuda()
            self.target_onehot = to_one_hot(target, self.num_classes)
            self.target_shuffled_onehot = self.target_onehot[self.indices]
            
            if k == -1:  #CutMix
                W,H = x.size(2),x.size(3)
                cut_rat = np.sqrt(1. - self.lam)
                cut_w = np.int(W * cut_rat)
                cut_h = np.int(H * cut_rat)
                cx = np.random.randint(W)
                cy = np.random.randint(H)
        
                bbx1 = np.clip(cx - cut_w // 2, 0, W)
                bby1 = np.clip(cy - cut_h // 2, 0, H)
                bbx2 = np.clip(cx + cut_w // 2, 0, W)
                bby2 = np.clip(cy + cut_h // 2, 0, H)
                
                x[:, :, bbx1:bbx2, bby1:bby2] = x[self.indices, :, bbx1:bbx2, bby1:bby2]
                lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (W * H))
                out = self.model(x)
                loss = bce_loss(softmax(out), self.target_onehot) * lam +\
                    bce_loss(softmax(out), self.target_shuffled_onehot) * (1. - lam)

            elif k==100:
                modifier_hook = self.module_list[2].register_forward_pre_hook(self.hook_pre)
                out = self.model(x)
                modifier_hook.remove()
                size = out.shape[0]
                loss = ce_loss(out[0:size // 2], target) + ce_loss(out[size // 2:],target) * self.total_unchanged_portion + \
                       ce_loss(out[size // 2:], target[self.indices]) * (1 - self.total_unchanged_portion)

            else:
                modifier_hook = self.module_list[k].register_forward_hook(self.hook_modify)
                out = self.model(x)
                modifier_hook.remove()
                
                # 当loss为CrossEntropy时，两种mixup的实现才一样
                # 但在这里是BCE
                '''
                loss = 1.0 * bce_loss(softmax(out), self.target_a) * self.total_unchanged_portion + \
                     bce_loss(softmax(out), self.target_b) * (1. - self.total_unchanged_portion) + \
                    1.0 * bce_loss(softmax(out), self.target_reweighted)
                '''
                '''
                loss = ce_loss(out, target) * self.total_unchanged_portion + \
                    ce_loss(out, target[self.indices]) * (1 - self.total_unchanged_portion)
                '''
                size = out.shape[0]
                loss = ce_loss(out[0:size//2], target) + ce_loss(out[size//2:], target) * self.total_unchanged_portion + \
                    ce_loss(out[size//2:], target[self.indices]) * (1 - self.total_unchanged_portion)
            return out, loss

    def hook_pre(self, module,inputs):
        return self.hook_modify(module, None, inputs[0])

    def hook_modify(self, module, input, output):
        self.gamma_adj = self.adjust_gamma(output)
        p = torch.ones_like(output[0]) * self.gamma_adj
        m_i_j = torch.bernoulli(p)
        mask_shape = len(m_i_j.shape)
        m_i_j = m_i_j.expand(output.size(0), m_i_j.size(0), m_i_j.size(1), m_i_j.size(2))
        # 核必须是奇数才能这么干，但核必然是奇数的
        holes = F.max_pool2d(m_i_j, self.kernel_size, self.stride, self.padding)#
        mask = 1 - holes
        unchanged = mask * output
        if mask_shape == 1:
            total_feats = output.size(1)
        else:
            # total_feats = output.size(1) * (output.size(2) ** 2)
            total_feats = output.size(1) * (output.size(2) * output.size(3))
        total_changed_pixels = holes[0].sum()
        total_changed_portion = total_changed_pixels / total_feats
        self.total_unchanged_portion = (total_feats - total_changed_pixels) / total_feats
        if self.patchup_type == 'hard':
            self.target_reweighted = self.total_unchanged_portion * self.target_onehot +\
                total_changed_portion * self.target_shuffled_onehot
            patches = holes * output[self.indices]
            self.target_b = self.target_onehot[self.indices]
        elif self.patchup_type == 'soft':
            self.target_reweighted = self.total_unchanged_portion * self.target_onehot +\
                self.lam * total_changed_portion * self.target_onehot +\
                (1 - self.lam) * total_changed_portion * self.target_shuffled_onehot
            patches = holes * output
            patches = patches * self.lam + patches[self.indices] * (1 - self.lam)
            self.target_b = self.lam * self.target_onehot + (1 - self.lam) * self.target_shuffled_onehot
        else:
            raise ValueError("patchup_type must be \'hard\' or \'soft\'.")
        
        output = torch.cat((output, unchanged + patches)).cuda()
        self.target_a = self.target_onehot
        return output

class PatchUp_multi_layers(nn.Module):
    def __init__(self, model, num_classes = 10, block_size=7, gamma=.9, patchup_type='hard',keep_prob=.9, mixup_layer=2):
        super().__init__()
        self.patchup_type = patchup_type
        self.block_size = block_size
        self.gamma = gamma
        self.gamma_adj = None
        self.kernel_size = (block_size, block_size)
        self.stride = (1, 1)
        self.padding = (block_size // 2, block_size // 2)
        self.computed_lam = None
        
        self.model = model
        self.num_classes = num_classes
        self. module_list = []
        self.mix_layer = ['base.logmel_extractor','base.conv_block1',\
             'base.resnet.layer1','base.resnet.layer2', 'base.resnet.layer3',\
                 'base.resnet.layer4', 'base.conv_block_after1.bn2']
        for n,m in self.model.named_modules():
            #if 'conv' in n:
            if n in self.mix_layer:
                print(n)
                self.module_list.append(m)

    def adjust_gamma(self, x):
        return self.gamma * x.shape[-1] ** 2 / \
               (self.block_size ** 2 * (x.shape[-1] - self.block_size + 1) ** 2)

    def forward(self, x, target=None):
        if target==None:
            out = self.model(x)
            return out
        else:

            self.lam = np.random.beta(2.0, 2.0)
            k = np.random.randint(0, len(self.module_list))
            self.indices = torch.randperm(target.size(0)).cuda()
            self.target_onehot = to_one_hot(target, self.num_classes)
            self.target_shuffled_onehot = self.target_onehot[self.indices]
            
            modifier_hook2 = self.module_list[2].register_forward_hook(self.hook_modify)
            modifier_hook3 = self.module_list[3].register_forward_hook(self.hook_modify)
            out = self.model(x)
            modifier_hook2.remove()
            modifier_hook3.remove()
            
            loss = bce_loss(softmax(out), self.target_reweighted)
            return out, loss
        
    def hook_modify(self, module, input, output):
        self.gamma_adj = self.adjust_gamma(output)
        p = torch.ones_like(output[0]) * self.gamma_adj
        m_i_j = torch.bernoulli(p)
        mask_shape = len(m_i_j.shape)
        m_i_j = m_i_j.expand(output.size(0), m_i_j.size(0), m_i_j.size(1), m_i_j.size(2))
        # 核必须是奇数才能这么干，但核必然是奇数的
        holes = F.max_pool2d(m_i_j, self.kernel_size, self.stride, self.padding)#
        mask = 1 - holes
        unchanged = mask * output
        if mask_shape == 1:
            total_feats = output.size(1)
        else:
            # total_feats = output.size(1) * (output.size(2) ** 2)
            total_feats = output.size(1) * (output.size(2) * output.size(3))
        total_changed_pixels = holes[0].sum()
        total_changed_portion = total_changed_pixels / total_feats
        self.total_unchanged_portion = (total_feats - total_changed_pixels) / total_feats
        if self.patchup_type == 'hard':
            self.target_reweighted = self.total_unchanged_portion * self.target_onehot +\
                total_changed_portion * self.target_shuffled_onehot
            patches = holes * output[self.indices]
            self.target_b = self.target_onehot[self.indices]
        elif self.patchup_type == 'soft':
            self.target_reweighted = self.total_unchanged_portion * self.target_onehot +\
                self.lam * total_changed_portion * self.target_onehot +\
                (1 - self.lam) * total_changed_portion * self.target_shuffled_onehot
            patches = holes * output
            patches = patches * self.lam + patches[self.indices] * (1 - self.lam)
            self.target_b = self.lam * self.target_onehot + (1 - self.lam) * self.target_shuffled_onehot
        else:
            raise ValueError("patchup_type must be \'hard\' or \'soft\'.")
        
        output = unchanged + patches
        self.target_a = self.target_onehot

        self.lam = np.random.beta(2.0, 2.0)
        self.indices = torch.randperm(output.size(0)).cuda()
        self.target_onehot = self.target_reweighted
        self.target_shuffled_onehot = self.target_onehot[self.indices]

        return output

class Manifold_multi_layers(nn.Module):
    def __init__(self, model, num_classes = 10, alpha = 1, mixup_layer=4):
        super().__init__()
        self.model = model
        self.alpha = alpha
        self.lam = None
        self.num_classes = num_classes
        ##选择需要操作的层，在ResNet中各block的层名为layer1,layer2...所以可以写成如下。其他网络请自行修改
        self.module_list = []
        self.mix_layer = ['base.logmel_extractor',  'base.conv_block1',\
             'base.resnet.layer1','base.resnet.layer2', 'base.resnet.layer3',\
                 'base.resnet.layer4', 'base.conv_block_after1']
        for n,m in self.model.named_modules():
            #if 'conv' in n:
            if n in self.mix_layer:
                print(n)
                self.module_list.append(m)

    def forward(self, x, target=None):
        if target==None:
            out = self.model(x)
            return out
        else:
            self.target_onehot = to_one_hot(target, self.num_classes)
            modifier_hook2 = self.module_list[2].register_forward_hook(self.hook_modify)
            modifier_hook4 = self.module_list[4].register_forward_hook(self.hook_modify)
            out = self.model(x)
            modifier_hook2.remove()
            modifier_hook4.remove()
            
            loss = bce_loss(softmax(out), self.target_reweighted)
            return out, loss
        
    def hook_modify(self, module, input, output):
        self.indices = torch.randperm(output.size(0)).cuda()
        self.lam = np.random.beta(self.alpha, self.alpha)
        self.target_shuffled_onehot = self.target_onehot[self.indices]
        self.target_reweighted = self.target_onehot* self.lam + self.target_shuffled_onehot * (1 - self.lam)
        self.target_onehot = self.target_reweighted
        output = self.lam * output + (1 - self.lam) * output[self.indices]
        return output


class Transfer_CNN10(nn.Module):
    def __init__(self, sample_rate, window_size, hop_size, mel_bins, fmin,
                 fmax, classes_num, layer_fc=2, freeze_base=None):
        """Classifier for a new task using pretrained Cnn14 as a sub module.
        """
        super(Transfer_CNN10, self).__init__()
        audioset_classes_num = 527

        self.base = Cnn10(sample_rate, window_size, hop_size, mel_bins, fmin,
                                          fmax, audioset_classes_num)

        self.layer_fc = layer_fc
        # Transfer to another task layer
        if (layer_fc == 1):
            self.fc_transfer = nn.Linear(2048, classes_num, bias=True)
        else:
            self.fc_transfer1 = nn.Linear(2048, 4096, bias=True)
            self.fc_transfer2 = nn.Linear(4096, 2048, bias=True)
            self.fc_transfer3 = nn.Linear(2048, 1024, bias=True)
            self.fc_transfer4 = nn.Linear(1024, 10, bias=True)

        if freeze_base:
            # Freeze AudioSet pretrained layers
            for param in self.base.parameters():
                param.requires_grad = False

        self.init_weights()

    def init_weights(self):
        if (self.layer_fc == 1):
            init_layer(self.fc_transfer)
        else:
            init_layer(self.fc_transfer1)
            init_layer(self.fc_transfer2)
            init_layer(self.fc_transfer3)
            init_layer(self.fc_transfer4)

    def load_from_pretrain(self, pretrained_checkpoint_path):
        # transfer_model(pretrained_checkpoint_path, self.base)
        checkpoint = torch.load(pretrained_checkpoint_path)
        self.base.load_state_dict(checkpoint['model'])

    def forward(self, input, mixup_lambda=None):
        """Input: (batch_size, data_length)
        """
        # input = torch.FloatTensor(np.array(input).astype(np.float32)).cuda()
        output_dict = self.base(input, mixup_lambda)
        embedding = output_dict['embedding']

        if (self.layer_fc == 1):
            # clipwise_output =  torch.log_softmax(self.fc_transfer(embedding), dim=-1)
            clipwise_output = self.fc_transfer(embedding)
        else:
            output1 = torch.relu(self.fc_transfer1(embedding))
            output1 = torch.relu(self.fc_transfer2(output1))
            output1 = torch.relu(self.fc_transfer3(output1))
            # clipwise_output =  torch.log_softmax(self.fc_transfer2(output1), dim=-1)
            clipwise_output = self.fc_transfer4(output1)
        output_dict['result'] = clipwise_output

        return output_dict['result']
