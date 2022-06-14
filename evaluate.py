import os
import argparse

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
import yaml
from sklearn.metrics import accuracy_score, confusion_matrix, log_loss, classification_report
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sn

from dataset import *
import models
torch.multiprocessing.set_sharing_strategy('file_system')

parser = argparse.ArgumentParser(description='evaluation')
parser.add_argument('--experiment_path', type=str, required=True)

args = parser.parse_args()

with open(os.path.join(args.experiment_path, "config.yaml"), "r") as reader:
    config = yaml.load(reader, Loader=yaml.FullLoader)

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


if config["audio_pretrain"] == "vggish":
    tt_ds = InferenceAudioDataset(config["data"]["test"]["file_path"],
                                  config["data"]["test"]['audio_feature'],
                                  classes_dic)
    config["data"]["dataloader_args"]["batch_size"] = 1
    tt_dataloader = DataLoader(tt_ds, batch_size=config["data"]["dataloader_args"]['batch_size'],
                               num_workers=config["data"]["dataloader_args"]['num_workers'],
                               collate_fn=collate
                               )
elif config["audio_pretrain"] == "panns":
    tt_ds = InferenceAudioDataset(config["data"]["test"]["file_path"],
                                  config["data"]["test"]['raw_audio'],
                                  classes_dic)
    config["data"]["dataloader_args"]["batch_size"] = 1
    tt_dataloader = DataLoader(tt_ds, batch_size=config["data"]["dataloader_args"]['batch_size'],
                               num_workers=config["data"]["dataloader_args"]['num_workers'],
                               collate_fn=collate
                               )
elif config["audio_pretrain"] == "psla":
    tt_ds = InferenceAudioDataset(config["data"]["test"]["file_path"],
                                  config["data"]["test"]['psla_audio_feature'],
                                  classes_dic)
    config["data"]["dataloader_args"]["batch_size"] = 1
    tt_dataloader = DataLoader(tt_ds, batch_size=config["data"]["dataloader_args"]['batch_size'],
                               num_workers=config["data"]["dataloader_args"]['num_workers'],
                               collate_fn=collate
                               )

if config['modality'] == 'anv':
    tt_ds = anvDataset(config["data"]["test"]["file_path"],
                          config["data"]["test"]['raw_audio'],
                          config["data"]["test"]['video_feature'],
                          classes_dic)
    config["data"]["dataloader_args"]["batch_size"] = 1
    tt_dataloader = DataLoader(tt_ds, batch_size=config["data"]["dataloader_args"]["batch_size"],
                                  num_workers=config["data"]["dataloader_args"]['num_workers'],
                                  collate_fn=fusion_collate)

if config['modality'] == 'v':
    tt_ds = InferenceVideoDataset(config["data"]["test"]["file_path"],
                                  config["data"]["test"]['video_feature'],
                                  classes_dic)
    config["data"]["dataloader_args"]["batch_size"] = 1
    tt_dataloader = DataLoader(tt_ds, batch_size=config["data"]["dataloader_args"]["batch_size"],
                                  num_workers=config["data"]["dataloader_args"]['num_workers'],
                                  shuffle=True,
                                  collate_fn=collate)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if config['modality'] == 'a':
    if config["audio_pretrain"] == "vggish":
        model = models.VGGbase_audiodense(config["num_classes"])
    elif config["audio_pretrain"] == "panns":
        model = models.Pannsbase_audiodense(config["num_classes"], device)
    elif config["audio_pretrain"] == "psla":
        model = models.Pslabase_audiodense(config["num_classes"])

    # print(model)
if config['modality'] == 'v':
    model = models.Effi_vediodense(config["num_classes"])
    # print(model)

if config['modality'] == 'anv':
    if config['fusion'] == 'late':
        model = models.Latefusion_dense(config["num_classes"], device, mode=config['mode'])
    else:
        model = models.earlyfusion_dense(config["num_classes"], device, mode=config['mode'])

model.load_state_dict(torch.load(
    os.path.join(args.experiment_path, "best_model.pt"), "cpu")
)

model = model.to(device).eval()

targets = []
probs = []
preds = []
aids = []

with torch.no_grad():
    tt_dataloader = tqdm(tt_dataloader)
    for batch_idx, batch in enumerate(tt_dataloader):
        if config['modality'] == 'anv':
            audio, video, target, fnames = batch
            audio = audio.to(device)
            video = video.to(device)
            target = target.to(device)
        else:
            audio, target, fnames = batch
            audio = audio.to(device)
            target = target.to(device)

        logit = model(audio, video)
        pred = torch.argmax(logit, 1)
        targets.append(target.cpu().numpy())
        probs.append(torch.softmax(logit, 1).cpu().numpy())
        preds.append(pred.cpu().numpy())
        aids.append(np.array(fnames))


targets = np.concatenate(targets, axis=0)
preds = np.concatenate(preds, axis=0)
probs = np.concatenate(probs, axis=0)
aids = np.concatenate(aids, axis=0)

writer = open(os.path.join(args.experiment_path, "result.txt"), "w")
cm = confusion_matrix(targets, preds)
keys = ['airport',
        'bus',
        'metro',
        'metro_station',
        'park',
        'public_square',
        'shopping_mall',
        'street_pedestrian',
        'street_traffic',
        'tram']

scenes_pred = [keys[pred] for pred in preds]
scenes_label = [keys[target] for target in targets]
pred_dict = {"aid": aids, "scene_pred": scenes_pred, "scene_label": scenes_label}
for idx, key in enumerate(keys):
    pred_dict[key] = probs[:, idx]
pd.DataFrame(pred_dict).to_csv(os.path.join(args.experiment_path, "prediction.csv"),
                               index=False,
                               sep="\t",
                               float_format="%.3f")


print(classification_report(targets, preds, target_names=keys), file=writer)

df_cm = pd.DataFrame(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis],
    index=keys, columns=keys)
plt.figure(figsize=(15, 12))
sn.heatmap(df_cm, annot=True)
plt.savefig(os.path.join(args.experiment_path, 'cm.png'))

acc = accuracy_score(targets, preds)
print('  ', file=writer)
print(f'accuracy: {acc:.3f}', file=writer)
logloss = log_loss(targets, probs)
print(f'overall log loss: {logloss:.3f}', file=writer)
