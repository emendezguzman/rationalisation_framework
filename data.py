# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

#Import Packages

import json
import torch

#Load JSON File (Text, Labels, Human Rationales)
def read_json_dataset(json_file):

  #Read JSON
  with open(json_file, 'r') as f:
    data = json.load(f)

  #Retrieve Texts, Input IDS and Attention for Human Rationales
  texts = data['texts']
  input_ids = data['input_ids']
  human_attention = data['human_rationales']
  model_name = data['model']

  #Retrieve Labels
  labels = []

  for item in data['labels']:
    labels.append([float(i) for i in item])

  return texts, input_ids, human_attention, labels, model_name

#Data Class (Rationale Dataset -> Sender_Input, Labels, Receiver_Input)
class RationaleDataset(torch.utils.data.Dataset):
  def __init__(self, input_ids, human_attention, labels):
    self.input_ids = input_ids
    self.human_attention = human_attention
    self.labels = labels

  def __getitem__(self, idx):
    sender_input = torch.tensor(self.input_ids[idx])
    labels = torch.tensor(self.labels[idx])
    receiver_input = torch.tensor(self.input_ids[idx])
    _aux_input = {'attention': torch.tensor(self.human_attention[idx])}

    return sender_input, labels, receiver_input, _aux_input

  def __len__(self):
    return len(self.labels)