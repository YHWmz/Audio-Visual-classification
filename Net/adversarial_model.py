import torch
from torch import nn
from Net.models import ResNet38, init_layer, init_bn, Wavegram_Logmel_Cnn14

# class GRL(torch.autograd.Function):
#     def __init__(self, lam = 1):
#         super(GRL, self).__init__()
#         self.lam = lam

#     def forward(self, x):
#         print('forward succeed!!!!!!!!!!!!!!!!!!')
#         return x.view_as(x)

#     def backward(self, grad_output):
#         grad_input = grad_output.clone() * (-self.lam)
#         return grad_input 

#     def set_lambda(self, lam):
#         self.lam = lam

class GRL(torch.autograd.Function):
    """
    Extension of grad reverse layer  梯度反转层
    """
    @staticmethod
    def forward(ctx, x, constant=1):
        #print(type(ctx),type(x),type(constant))
        ctx.constant = constant
        return x.view_as(x)
    
    @staticmethod
    def backward(ctx, grad_output):
        #print("before:{}".format(grad_output))
        grad_output = grad_output.neg() * ctx.constant
        #print("after:{}".format(grad_output))
        return grad_output, None


class classifier(nn.Module):
    def __init__(self, classnum):
        super(classifier, self).__init__()
        self.fc1 = nn.Linear(2048, 4096, bias=True)
        self.fc2 = nn.Linear(4096, 2048, bias=True)
        self.fc3 = nn.Linear(2048, 1024, bias=True)
        self.fc4 = nn.Linear(1024, classnum, bias=True)
        
        self.init_weights()
        
    def init_weights(self):
        init_layer(self.fc1)
        init_layer(self.fc2)
        init_layer(self.fc3)
        init_layer(self.fc4)

    def forward(self, input):
        x = torch.relu(self.fc1(input))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        return x

class Adversarial_ResNet38(nn.Module):
    def __init__(self, sample_rate, window_size, hop_size, mel_bins, fmin, 
        fmax, freeze_base = None):
        """Classifier for a new task using pretrained Cnn14 as a sub module.
        """
        super(Adversarial_ResNet38, self).__init__()
        audioset_classes_num = 527
        
        self.base = ResNet38(sample_rate, window_size, hop_size, mel_bins, fmin, 
            fmax, audioset_classes_num)

        self.fc_class = classifier(10)
        self.fc_domain = classifier(6)

        self.GRL = GRL()
        
        #self.fc_class.fc1.register_full_backward_hook(hook_backward_fn)
        #self.fc_domain.fc1.register_full_backward_hook(hook_backward_fn)
        #self.fc_class.fc1.weight.register_hook(hook_print_grad)
        #self.fc_domain.fc1.weight.register_hook(hook_print_grad)
        #self.base.resnet.layer4[2].conv2.weight.register_hook(hook_print_grad)

        print('done')

        if freeze_base:
            # Freeze AudioSet pretrained layers
            for param in self.base.parameters():
                param.requires_grad = False
        

    def load_from_pretrain(self, pretrained_checkpoint_path):
        checkpoint = torch.load(pretrained_checkpoint_path)
        self.base.load_state_dict(checkpoint['model'])

    def forward(self, input):
        """
        Input: (batch_size, data_length)
        """
        #print(input.dtype)
        output_dict = self.base(input)
        embedding = output_dict['embedding']

        output_dict['class'] = self.fc_class(embedding)
        
        reversal_emb = self.GRL.apply(embedding, 1)
        output_dict['domain'] =  self.fc_domain(reversal_emb)

        return output_dict


class Adversarial_Wavegram(nn.Module):
    def __init__(self, sample_rate, window_size, hop_size, mel_bins, fmin,
                 fmax, freeze_base=None):
        """Classifier for a new task using pretrained Cnn14 as a sub module.
        """
        super(Adversarial_Wavegram, self).__init__()
        audioset_classes_num = 527

        self.base = Wavegram_Logmel_Cnn14(sample_rate, window_size, hop_size, mel_bins, fmin,
                             fmax, audioset_classes_num)

        self.fc_class = classifier(10)
        self.fc_domain = classifier(6)

        self.GRL = GRL()

        # self.fc_class.fc1.register_full_backward_hook(hook_backward_fn)
        # self.fc_domain.fc1.register_full_backward_hook(hook_backward_fn)
        # self.fc_class.fc1.weight.register_hook(hook_print_grad)
        # self.fc_domain.fc1.weight.register_hook(hook_print_grad)
        # self.base.resnet.layer4[2].conv2.weight.register_hook(hook_print_grad)

        print('done')

        if freeze_base:
            # Freeze AudioSet pretrained layers
            for param in self.base.parameters():
                param.requires_grad = False

    def load_from_pretrain(self, pretrained_checkpoint_path):
        checkpoint = torch.load(pretrained_checkpoint_path)
        self.base.load_state_dict(checkpoint['model'])

    def forward(self, input):
        """
        Input: (batch_size, data_length)
        """
        # print(input.dtype)
        output_dict = self.base(input)
        embedding = output_dict['embedding']

        output_dict['class'] = self.fc_class(embedding)

        reversal_emb = self.GRL.apply(embedding, 1)
        output_dict['domain'] = self.fc_domain(reversal_emb)

        return output_dict

def hook_print_grad(grad):
    print(grad.shape)
    print('grad is :{}'.format(grad))
    return grad

def hook_backward_fn(module, grad_input, grad_output):
    print(f"module: {module}")
    print(f"grad_output: {len(grad_output)}")
    print("*"*20)

def to_one_hot(inp, num_classes): 
    y_onehot = torch.FloatTensor(inp.size(0), num_classes).to(inp.device)
    y_onehot.zero_()
    y_onehot.scatter_(1, inp.unsqueeze(1).data, 1)
    return y_onehot

import numpy as np
bce_loss = nn.BCELoss()
softmax = nn.Softmax(dim=1)
ce_loss = nn.CrossEntropyLoss()

class ad_PatchUpModel(nn.Module):
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
        self.mix_layer = ['base.logmel_extractor', 'base.bn0', 'base.conv_block1.bn2',\
             'base.resnet.layer1','base.resnet.layer2', 'base.resnet.layer3',\
                 'base.resnet.layer4', 'base.conv_block_after1.bn2']
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
            k = self.k #np.random.randint(-1, len(self.module_list))
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
