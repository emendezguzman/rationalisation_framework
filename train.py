# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

#Import Packages
from data import read_json_dataset, RationaleDataset
from archs import Sender, Receiver, RationaleWrapper
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import egg.core as core
import torch
from transformers import BertModel
from torch.nn import functional as F

#EGG-level parameters
opts = core.init(params=['--random_seed=7',
                         '--lr=1e-3',
                         '--batch_size=4',
                         '--optimizer=adam'])

#Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#Read JSON File
json_file = '/content/full_dataset_bert-base-uncased_20230822.json'
texts, input_ids, human_attention, labels, model_name = read_json_dataset(json_file)

#Create Train and Test Data
train_size = 0.8
train_input_ids, test_input_ids, train_human_attention, test_human_attention, train_labels, test_labels = train_test_split(input_ids, human_attention, labels,
                                                                                                                           train_size=train_size)
print("Train Dataset (Items):", len(train_input_ids))
print("Test Dataset (Items):", len(test_input_ids))

#Create RationaleDataset Objects
train_dataset = RationaleDataset(train_input_ids, train_human_attention, train_labels)
test_dataset = RationaleDataset(test_input_ids, test_human_attention, test_labels)

#Create DataLoaders
kwargs = {'num_workers': 1, 'pin_memory': True} if torch.cuda.is_available() else {}
batch_size = opts.batch_size
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, **kwargs)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, **kwargs)

#Download Transformer Model
model = BertModel.from_pretrained(model_name, output_hidden_states=True)

#Sparsity Regularisation Parameters (Hyperparameters - Sparsity = 0.0003 / Coherence = 2.0)
sparsity = 0.003
coherence = 2.0
coherent_factor = sparsity * coherence

#Loss Function
def loss(sender_input, _message, _receiver_input, receiver_output, _labels, _aux_input=None):
  """
    :param _message: Rationale (Shape -> [B, T])
    :param receiver_output: Labels Predictions (Shape -> [B, 11])
    :param _labels: Labels (Shape -> [B, 11])
    :return: Loss (Shape -> [B])
    """

  #Classification Loss for p(y|x,z)
  class_cost = F.cross_entropy(receiver_output, _labels, reduction='none')

  #Length Regularisation
  zsum = _message.sum((1,2))
  zsum_cost = sparsity * zsum

  # #Sparcity Regularisation
  zdiff = _message[:, 1:] - _message[:, :-1]
  zdiff = zdiff.abs().sum((1,2))
  zdiff_cost = coherent_factor * zdiff

  #Rationale Classification
  rat_cost = F.binary_cross_entropy_with_logits(_message.float(), _aux_input['attention'].float(), reduction='none').sum(dim=(1,2))

  #Overall Loss
  loss = class_cost + zsum_cost + zdiff_cost + rat_cost

  return loss, {}

#Initialise Sender and Receiver
sender = Sender(model=model)
sender = RationaleWrapper(sender)
receiver = Receiver(model=model)
receiver = core.ReinforceDeterministicWrapper(receiver)
optimizer = core.build_optimizer(receiver.parameters())

#Game Instance
game = core.SymbolGameReinforce(sender, receiver, loss, sender_entropy_coeff=0.05, receiver_entropy_coeff=0.0)
optimizer = torch.optim.Adam(game.parameters())

trainer = core.Trainer(
    game=game, optimizer=optimizer, train_data=train_loader,
    validation_data=test_loader
)

#Training
n_epochs = 5
trainer.train(n_epochs)