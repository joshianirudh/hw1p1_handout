import torch
import torch.nn as nn
import numpy as np
from torchsummaryX import summary
import sklearn
import gc
import zipfile
import bisect
import pandas as pd
from tqdm.auto import tqdm
import os
import datetime
import wandb
import yaml
import torchaudio.transforms as tat
import torchaudio
device =  '' if torch.cuda.is_available() else 'cpu'
print("Device: ", device)


PHONEMES = [
            '[SIL]',   'AA',    'AE',    'AH',    'AO',    'AW',    'AY',
            'B',     'CH',    'D',     'DH',    'EH',    'ER',    'EY',
            'F',     'G',     'HH',    'IH',    'IY',    'JH',    'K',
            'L',     'M',     'N',     'NG',    'OW',    'OY',    'P',
            'R',     'S',     'SH',    'T',     'TH',    'UH',    'UW',
            'V',     'W',     'Y',     'Z',     'ZH',    '[SOS]', '[EOS]']


config = {
    'Name': '', # Write your name here
    'subset': 1.0, # Subset of dataset to use (1.0 == 100% of data)
    'context': 50,
    'archetype': 'diamond', # Default Values: pyramid, diamond, inverse-pyramid,cylinder
    'activations': 'GELU',
    'learning_rate': 0.001,
    'dropout': 0.25,
    'optimizers': 'SGD',
    'scheduler': 'ReduceLROnPlateau',
    'epochs': 14,
    'batch_size': 2048,
    'weight_decay': 0.05,
    'weight_initialization': None, # e.g kaiming_normal, kaiming_uniform, uniform, xavier_normal or xavier_uniform
    'augmentations': 'Both', # Options: ["FreqMask", "TimeMask", "Both", null]
    'freq_mask_param': 4,
    'time_mask_param': 8
 }



class AudioTestDataset(torch.utils.data.Dataset):

    def __init__(self, root, context=0, partition= "test-clean"): # Feel free to add more arguments

        self.context = context
        self.subset = config['subset']
        self.freq_masking = tat.FrequencyMasking(100)
        self.time_masking = tat.TimeMasking(100)

        self.mfcc_dir  =  os.path.join(root, partition, 'mfcc')

        mfcc_names = sorted(os.listdir(self.mfcc_dir))

        subset_size = int(self.subset * len(mfcc_names))

        mfcc_names = mfcc_names[:subset_size]


        self.mfccs= []

        for i in tqdm(range(len(mfcc_names))):

            mfcc = np.load(os.path.join(self.mfcc_dir, mfcc_names[i]))
            mfccs_normalized = mfcc - np.mean(mfcc, axis=0, keepdims=True)

            mfccs_normalized = torch.tensor(mfccs_normalized, dtype=torch.float32)

            self.mfccs.append(mfccs_normalized)

        self.mfccs = torch.cat(self.mfccs, dim=0)
        self.length = len(self.mfccs)
        self.mfccs = nn.functional.pad(self.mfccs, (0, 0, self.context, self.context)) # TODO


    def __len__(self):
        return self.length

    def collate_fn(self, batch):
      return torch.stack(batch, dim=0)

    def __getitem__(self, ind):
        # TODO: Based on context and offset, return a frame at given index with context frames to the left, and right.
        start = idx
        end = idx + 2 * self.context + 1
        frames = self.mfccs[start:end, :]

        # After slicing, you get an array of shape 2*context+1 x 28.

        return frames

ROOT = "11785-s25-hw1p2"

train_data = AudioDataset(ROOT, context = config["context"], partition= "train-clean-100")

val_data = AudioDataset(ROOT, context = config["context"], partition="dev-clean")

test_data = AudioTestDataset(ROOT, context = config["context"])

for i, j in train_data:
    print(i.shape, j.shape)
    print(j)
    break

train_loader = torch.utils.data.DataLoader(
    dataset     = train_data,
    num_workers = 0,
    batch_size  = config['batch_size'],
    pin_memory  = True,
    shuffle     = True,
    collate_fn = train_data.collate_fn
)

val_loader = torch.utils.data.DataLoader(
    dataset     = val_data,
    num_workers = 0,
    batch_size  = config['batch_size'],
    pin_memory  = True,
    shuffle     = False
)

test_loader = torch.utils.data.DataLoader(
    dataset     = test_data,
    num_workers = 0,
    batch_size  = config['batch_size'],
    pin_memory  = True,
    shuffle     = False
)


print("Batch size     : ", config['batch_size'])
print("Context        : ", config['context'])
print("Input size     : ", (2*config['context']+1)*28)
print("Output symbols : ", len(PHONEMES))

print("Train dataset samples = {}, batches = {}".format(train_data.__len__(), len(train_loader)))
print("Validation dataset samples = {}, batches = {}".format(val_data.__len__(), len(val_loader)))
print("Test dataset samples = {}, batches = {}".format(test_data.__len__(), len(test_loader)))


# Testing code to check if your validation data loaders are working
all = []
for i, data in enumerate(val_loader):
    frames, phoneme = data
    all.append(phoneme)
    break


class Network(nn.Module):
    def __init__(self, input_size, output_size):
        super(Network, self).__init__()

        self.model = nn.Sequential(
            torch.nn.Linear(input_size, 512),
            nn.BatchNorm1d(512),
            torch.nn.ReLU(),
            nn.Dropout(0.2),
            torch.nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            torch.nn.ReLU(),
            nn.Dropout(0.3),
            torch.nn.Linear(1024, 2048),
            nn.BatchNorm1d(2048),
            torch.nn.ReLU(),
            nn.Dropout(0.3),
            torch.nn.Linear(2048, 2048),
            nn.BatchNorm1d(2048),
            torch.nn.ReLU(),
            nn.Dropout(0.3),
            torch.nn.Linear(2048, 1024),
            nn.BatchNorm1d(1024),
            torch.nn.ReLU(),
            nn.Dropout(0.3),
            torch.nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            torch.nn.ReLU(),
            nn.Dropout(0.2),
            torch.nn.Linear(512, output_size)
        )

        if config['weight_initialization'] is not None:
            self.initialize_weights()

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Linear):
                if config["weight_initialization"] == "xavier_normal":
                    torch.nn.init.xavier_normal_(m.weight)
                elif config["weight_initialization"] == "xavier_uniform":
                    torch.nn.init.xavier_uniform_(m.weight)
                elif config["weight_initialization"] == "kaiming_normal":
                    torch.nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                elif config["weight_initialization"] == "kaiming_uniform":
                    torch.nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                elif config["weight_initialization"] == "uniform":
                    torch.nn.init.uniform_(m.weight)
                else:
                    raise ValueError("Invalid weight_initialization value")

                # Initialize bias to 0
                m.bias.data.fill_(0)


    def forward(self, x):

        # Flatten to a 1D vector for each data point
        x = torch.flatten(x, start_dim=1)  # Keeps batch size, flattens the rest

        return self.model(x)



INPUT_SIZE  = (2*config['context'] + 1) * 28 # Why is this the case?
model = Network(INPUT_SIZE, len(train_data.phonemes)).to(device)
print(summary(model, frames.to(device)))


criterion = torch.nn.CrossEntropyLoss() # Defining Loss function.
# We use CE because the task is multi-class classification

# Choose an appropriate optimizer of your choice
optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'])

# Recommended : Define Scheduler for Learning Rate,
# including but not limited to StepLR, MultiStep, CosineAnnealing, CosineAnnealingWithWarmRestarts, ReduceLROnPlateau, etc.
# You can refer to Pytorch documentation for more information on how to use them.
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)

# Is your training time very high?
# Look into mixed precision training if your GPU (Tesla T4, V100, etc) can make use of it
# Refer - https://pytorch.org/docs/stable/notes/amp_examples.html
# Mixed Precision Training with AMP for speedup
scaler = torch.amp.GradScaler('cuda', enabled=True)

"""# Training and Validation Functions

This section covers the training, and validation functions for each epoch of running your experiment with a given model architecture. The code has been provided to you, but we recommend going through the comments to understand the workflow to enable you to write these loops for future HWs.
"""


def eval(model, dataloader):

    model.eval() # set model in evaluation mode
    vloss, vacc = 0, 0 # Monitoring loss and accuracy
    batch_bar   = tqdm(total=len(val_loader), dynamic_ncols=True, position=0, leave=False, desc='Val')

    for i, (frames, phonemes) in enumerate(dataloader):

        ### Move data to device (ideally GPU)
        frames      = frames.to(device)
        phonemes    = phonemes.to(device)

        # makes sure that there are no gradients computed as we are not training the model now
        with torch.inference_mode():
            ### Forward Propagation
            logits  = model(frames)
            ### Loss Calculation
            loss    = criterion(logits, phonemes)

        vloss += loss.item()
        vacc  += torch.sum(torch.argmax(logits, dim= 1) == phonemes).item()/logits.shape[0]

        # Do you think we need loss.backward() and optimizer.step() here?

        batch_bar.set_postfix(loss="{:.04f}".format(float(vloss / (i + 1))),
                              acc="{:.04f}%".format(float(vacc*100 / (i + 1))))
        batch_bar.update()

        ### Release memory
        del frames, phonemes, logits
        torch.cuda.empty_cache()

    batch_bar.close()
    vloss   /= len(val_loader)
    vacc    /= len(val_loader)

    return vloss, vacc



def test(model, test_loader):
    ### What you call for model to perform inference?
    model.eval()  # TODO train or eval?

    ### List to store predicted phonemes of test data
    test_predictions = []

    ### Which mode do you need to avoid gradients?
    with torch.no_grad():  # TODO
        for i, mfccs in enumerate(tqdm(test_loader)):
            mfccs = mfccs.to(device)

            logits = model(mfccs)

            ### Get most likely predicted phoneme with argmax
            predicted_phonemes = torch.argmax(logits, dim=0)

            ### How do you store predicted_phonemes with test_predictions? Hint, look at eval
            # Remember the phonemes were converted to their corresponding integer indices earlier, and the results of the argmax is a list of the indices of the predicted phonemes.
            # So how do you get and store the actual predicted phonemes
            # TODO: Store predicted_phonemes
            predicted_indices = torch.argmax(logits, dim=1)  # Shape: (batch_size,)
            test_predictions.append(predicted_indices.cpu())
    all_indices = torch.cat(test_predictions, dim=0)  # Shape: (total_samples,)

    # Convert numerical indices to phoneme strings
    phoneme_predictions = [PHONEMES[idx] for idx in all_indices.numpy()]
    return phoneme_predictions

predictions = test(model, test_loader)

### Create CSV file with predictions
with open("./submission.csv", "w+") as f:
    f.write("id,label\n")
    for i in range(len(predictions)):
        f.write("{},{}\n".format(i, predictions[i]))


