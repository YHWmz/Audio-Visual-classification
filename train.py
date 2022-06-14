from pathlib import Path
import os
import argparse
import pdb
import time
import random
os.environ["CUDA_VISIBLE_DEVICES"]="3"
from tqdm import tqdm
import yaml
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
import torch
from torch.utils.data import DataLoader
import torch.optim as optim

from dataset import SceneDataset, InferenceAudioDataset, collate, InferenceVideoDataset, fusion_collate, anvDataset
import models
import utils
import vggish

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

parser = argparse.ArgumentParser(description='training networks')
parser.add_argument('--config_file', type=str, required=True)
parser.add_argument('--seed', type=int, default=0, required=False,
                    help='set the seed to reproduce result')
args = parser.parse_args()

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

with open(args.config_file, "r") as reader:
    config = yaml.load(reader, Loader=yaml.FullLoader)

#[10, 96, 64]
if config['modality'] == 'a':
    if config["audio_pretrain"] == "vggish":
        audio_train_ds = InferenceAudioDataset(config["data"]["train"]["file_path"],
                             config["data"]["train"]['audio_feature'],
                             classes_dic)
        train_dataloader = DataLoader(audio_train_ds, batch_size=config["data"]["dataloader_args"]['batch_size'],
                                    num_workers=config["data"]["dataloader_args"]['num_workers'],
                                    shuffle=True,
                                    collate_fn=collate)
        audio_val_ds = InferenceAudioDataset(config["data"]["val"]["file_path"],
                                               config["data"]["val"]['audio_feature'],
                                               classes_dic)
        val_dataloader = DataLoader(audio_val_ds, batch_size=config["data"]["dataloader_args"]['batch_size'],
                                    num_workers=config["data"]["dataloader_args"]['num_workers'],
                                    shuffle=True,
                                    collate_fn=collate)
    elif config["audio_pretrain"] == "panns":
        audio_train_ds = InferenceAudioDataset(config["data"]["train"]["file_path"],
                                               config["data"]["train"]['raw_audio'],
                                               classes_dic)
        train_dataloader = DataLoader(audio_train_ds, batch_size=config["data"]["dataloader_args"]['batch_size'],
                                      num_workers=config["data"]["dataloader_args"]['num_workers'],
                                      shuffle=True,
                                      collate_fn=collate)
        audio_val_ds = InferenceAudioDataset(config["data"]["val"]["file_path"],
                                             config["data"]["val"]['raw_audio'],
                                             classes_dic)
        val_dataloader = DataLoader(audio_val_ds, batch_size=config["data"]["dataloader_args"]['batch_size'],
                                    num_workers=config["data"]["dataloader_args"]['num_workers'],
                                    shuffle=True,
                                    collate_fn=collate)
    elif config["audio_pretrain"] == "psla":
        audio_train_ds = InferenceAudioDataset(config["data"]["train"]["file_path"],
                                               config["data"]["train"]['psla_audio_feature'],
                                               classes_dic)
        train_dataloader = DataLoader(audio_train_ds, batch_size=config["data"]["dataloader_args"]['batch_size'],
                                      num_workers=config["data"]["dataloader_args"]['num_workers'],
                                      shuffle=True,
                                      collate_fn=collate)
        audio_val_ds = InferenceAudioDataset(config["data"]["val"]["file_path"],
                                             config["data"]["val"]['psla_audio_feature'],
                                             classes_dic)
        val_dataloader = DataLoader(audio_val_ds, batch_size=config["data"]["dataloader_args"]['batch_size'],
                                    num_workers=config["data"]["dataloader_args"]['num_workers'],
                                    shuffle=True,
                                    collate_fn=collate)

if config['modality'] == 'v':
    video_train_ds = InferenceVideoDataset(config["data"]["train"]["file_path"],
                                           config["data"]["train"]['video_feature'],
                                           classes_dic)
    train_dataloader = DataLoader(video_train_ds, batch_size=8,
                                  num_workers=config["data"]["dataloader_args"]['num_workers'],
                                  shuffle=True,
                                  collate_fn=collate)
    video_val_ds = InferenceVideoDataset(config["data"]["val"]["file_path"],
                                         config["data"]["val"]['video_feature'],
                                         classes_dic)
    val_dataloader = DataLoader(video_val_ds, batch_size=8,
                                num_workers=config["data"]["dataloader_args"]['num_workers'],
                                shuffle=True,
                                collate_fn=collate)

if config['modality'] == 'anv':
    train_ds = anvDataset(config["data"]["train"]["file_path"],
                          config["data"]["train"]['raw_audio'],
                          config["data"]["train"]['video_feature'],
                          classes_dic)
    train_dataloader = DataLoader(train_ds, batch_size=8,
                                  num_workers=config["data"]["dataloader_args"]['num_workers'],
                                  shuffle=True,
                                  collate_fn=fusion_collate)
    val_ds = anvDataset(config["data"]["val"]["file_path"],
                        config["data"]["val"]['raw_audio'],
                        config["data"]["val"]['video_feature'],
                        classes_dic)
    val_dataloader = DataLoader(val_ds, batch_size=8,
                                num_workers=config["data"]["dataloader_args"]['num_workers'],
                                shuffle=True,
                                collate_fn=fusion_collate)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if config['modality'] == 'a':
    if config["audio_pretrain"] == "vggish":
        model = models.VGGbase_audiodense(config["num_classes"])
    elif config["audio_pretrain"] == "panns":
        model = models.Pannsbase_audiodense(config["num_classes"], device)
    elif config["audio_pretrain"] == "psla":
        model = models.Pslabase_audiodense(config["num_classes"])

    print(model)

if config['modality'] == 'v':
    model = models.Effi_vediodense(config["num_classes"])
    print(model)

if config['modality'] == 'anv':
    if config['fusion'] == 'late':
        model = models.Latefusion_dense(config["num_classes"], device, mode=config['mode'])
    else:
        model = models.earlyfusion_dense(config["num_classes"], device, mode=config['mode'])


output_dir = config["output_dir"]
Path(output_dir).mkdir(exist_ok=True, parents=True)
logging_writer = utils.getfile_outlogger(os.path.join(output_dir, "train.log"))

model = model.to(device)

loss_fn = torch.nn.CrossEntropyLoss()

# if config['modality'] == 'a':
#     if model.deeper:
#         cls_params = list(model.audio_embed.parameters())+list(model.outputlayer.parameters())
#     else:
#         cls_params = model.audio_cls.parameters()
# elif config['modality'] == 'v':
#     if model.deeper:
#         cls_params = list(model.vedio_embed.parameters())+list(model.outputlayer.parameters())
#     else:
#         cls_params = model.vedio_cls.parameters()

# lr = config["optimizer"]["args"]['lr']
# optimizer = torch.optim.Adam([{'params': model.pretrain.parameters(), 'lr': lr*0.5}, {'params': cls_params, 'lr': lr}], weight_decay=config["optimizer"]["args"]['weight_decay'])

# cls_params = list(model.audio_model.parameters()) + list(model.video_embed.parameters())+list(model.audio_embed.parameters())+list(model.outputlayer.parameters())
optimizer = getattr(optim, config["optimizer"]["type"])(
    model.parameters(),
    # cls_params,
    **config["optimizer"]["args"])

lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,
    **config["lr_scheduler"]["args"])

print('-----------start training-----------')


def train(epoch, data_loader, modality):
    model.train()
    train_loss = 0.
    start_time = time.time()
    count = len(data_loader) * (epoch - 1)
    loader = tqdm(data_loader)
    for batch_idx, batch in enumerate(loader):
        count = count + 1
        if modality == 'anv':
            audio, video, target, fname = batch
            audio = audio.to(device)
            video = video.to(device)
            target = target.to(device)
        else:
            audio, target, fname = batch
            audio = audio.to(device)
            target = target.to(device)

        # training
        optimizer.zero_grad()

        if modality == 'anv':
            # print(audio[0][111:131])
            # print(video[0, 3, 2, 14, 111:131])
            # print(fname)
            logit = model(audio, video)
            # print(logit[0])
        else:
            logit = model(audio)
        loss = loss_fn(logit, target)
        loss.backward()

        train_loss += loss.item()
        optimizer.step()

        if (batch_idx + 1) % 100 == 0:
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | ms/batch {:5.2f} | loss {:5.2f} |'.format(
                epoch, batch_idx + 1, len(data_loader),
                elapsed * 1000 / (batch_idx + 1), loss.item()))

    train_loss /= (batch_idx + 1)
    logging_writer.info('-' * 99)
    logging_writer.info('| epoch {:3d} | time: {:5.2f}s | training loss {:5.2f} |'.format(
        epoch, (time.time() - start_time), train_loss))
    return train_loss

def validate(epoch, data_loader, modality):
    model.eval()
    validation_loss = 0.
    start_time = time.time()
    # data loading
    cv_loss = 0.
    targets = []
    preds = []
    with torch.no_grad():
        for batch_idx, batch in enumerate(data_loader):
            if modality == 'anv':
                audio, video, target, fname = batch
                audio = audio.to(device)
                video = video.to(device)
                target = target.to(device)
                logit = model(audio, video)
            else:
                audio, target, _ = batch
                audio = audio.to(device)
                target = target.to(device)
                logit = model(audio)

            loss = loss_fn(logit, target)
            pred = torch.argmax(logit, 1)
            targets.append(target.cpu().numpy())
            preds.append(pred.cpu().numpy())
            cv_loss += loss.item()

    cv_loss /= (batch_idx+1)
    preds = np.concatenate(preds, axis=0)
    targets = np.concatenate(targets, axis=0)
    accuracy = accuracy_score(targets, preds)
    logging_writer.info('| epoch {:3d} | time: {:5.2f}s | cv loss {:5.2f} | cv accuracy: {:5.2f} |'.format(
            epoch, time.time() - start_time, cv_loss, accuracy))
    logging_writer.info('-' * 99)

    return cv_loss


training_loss = []
cv_loss = []

with open(os.path.join(output_dir, 'config.yaml'), "w") as writer:
    yaml.dump(config, writer, default_flow_style=False)

not_improve_cnt = 0
for epoch in range(1, config["epoch"]):
    print('epoch', epoch)
    training_loss.append(train(epoch, train_dataloader, config['modality']))
    cv_loss.append(validate(epoch, val_dataloader, config['modality']))
    print('-' * 99)
    print('epoch', epoch, 'training loss: ', training_loss[-1], 'cv loss: ', cv_loss[-1])

    if cv_loss[-1] == np.min(cv_loss):
        # save current best model
        torch.save(model.state_dict(), os.path.join(output_dir, 'best_model.pt'))
        print('best validation model found and saved.')
        print('-' * 99)
        not_improve_cnt = 0
    else:
        not_improve_cnt += 1
    
    lr_scheduler.step(cv_loss[-1])
    
    if not_improve_cnt == config["early_stop"]:
        break


minmum_cv_index = np.argmin(cv_loss)
minmum_loss = np.min(cv_loss)
plt.plot(training_loss, 'r')
#plt.hold(True)
plt.plot(cv_loss, 'b')
plt.axvline(x=minmum_cv_index, color='k', linestyle='--')
plt.plot(minmum_cv_index, minmum_loss,'r*')

plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train_loss', 'val_loss'], loc='upper left')
plt.savefig(os.path.join(output_dir, 'loss.png'))
