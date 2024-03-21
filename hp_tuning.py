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

#Create Dev Data
dev_size = 0.1
train_train_input_ids, dev_input_ids, train_train_human_attention, dev_human_attention, train_train_labels, dev_labels = train_test_split(train_input_ids, train_human_attention, train_labels,
                                                                                                                                          test_size=dev_size)

#Train, Dev and Test Set
print("Train Dataset (Items):", len(train_train_input_ids))
print("Dev Dataset (Items):", len(dev_input_ids))
print("Test Dataset (Items):", len(test_input_ids))

# Create RationaleDataset Objects
train_dataset = RationaleDataset(train_train_input_ids, train_train_human_attention, train_train_labels)
dev_dataset = RationaleDataset(dev_input_ids, dev_human_attention, dev_labels)
test_dataset = RationaleDataset(test_input_ids, test_human_attention, test_labels)

#Create DataLoaders
kwargs = {'num_workers': 1, 'pin_memory': True} if torch.cuda.is_available() else {}
batch_size = opts.batch_size
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, **kwargs)
dev_loader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=True, **kwargs)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, **kwargs)

#Rationale Wrapper HP
class RationaleWrapper(nn.Module):
    """
    This is a special Reinforce Wrapper for an agent.
    It takes an input text and return samples from p(z|x)

    Reinforce Wrapper for an agent. Assumes that the during the forward,
    the wrapped agent returns log-probabilities over the potential outputs. During training, the wrapper
    transforms them into a tuple of (sample from the Bernoulli, log-prob of the sample, entropy for the sample).
    Eval-time the sample is replaced with p(z|x) > 0.5 (Hyperparameter).
    """

    def __init__(self, agent, rationale_threshold):

        super(RationaleWrapper, self).__init__()
        self.agent = agent
        self.rationale_threshold = rationale_threshold

    def forward(self, *args, **kwargs):
        z_dist = self.agent(*args, **kwargs)

        if self.training:
            z = z_dist.sample()
        else:
            #z = z_dist.sample()
            z = torch.where(z_dist.probs > self.rationale_threshold, 1, 0)

        log_prob = z_dist.log_prob(z.float()).mean((1,2))
        entropy = z_dist.entropy().mean((1,2))
        z = z.squeeze(-1)

        return z, log_prob, entropy

#Download Transformer Model
model = BertModel.from_pretrained(model_name, output_hidden_states=True)

#Sweep Configuration
sweep_config = {
    "method": "random",
    "metric": {"goal": "minimize", "name": "validation_loss"},
    "parameters": {
        "regularisation_length": {"values": [0.03, 0.06, 0.09, 0.12, 0.15]},
        "regularisation_sparsity": {"values": [0.06, 0.12, 0.18, 0.24, 0.30]},
        "entropy_coefficient": {"values": [0.05, 0.10, 0.15, 0.20, 0.25]},
        "rationale_threshold": {"values": [0.40, 0.45, 0.50, 0.55, 0.60]},
    },
}

#Logging W&B
wandb.login()

#Sweep Initialisation
project_name = 'uom_supervised_rationalisation'
sweep_id = wandb.sweep(sweep_config, project=project_name)

#Training Function
def train():

  #Initialising a new wandb run
  wandb.init()

  #Access Hyperparameters
  config = wandb.config
  rationale_threshold = config.rationale_threshold
  regularisation_length = config.regularisation_length
  regularisation_sparsity = config.regularisation_sparsity
  entropy_coefficient = config.entropy_coefficient

  #Initialise Sender and Receiver
  sender = Sender(model=model)
  sender = RationaleWrapper(sender, rationale_threshold=rationale_threshold)
  receiver = Receiver(model=model)
  receiver = core.ReinforceDeterministicWrapper(receiver)
  optimizer = core.build_optimizer(receiver.parameters())

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
    zsum_cost = regularisation_length * zsum

    # #Sparcity Regularisation
    zdiff = _message[:, 1:] - _message[:, :-1]
    zdiff = zdiff.abs().sum((1,2))
    zdiff_cost = regularisation_sparsity * zdiff

    #Rationale Classification
    rat_cost = F.binary_cross_entropy_with_logits(_message.float(), _aux_input['attention'].float(), reduction='none').sum(dim=(1,2))

    #Overall Loss
    loss = class_cost + zsum_cost + zdiff_cost + rat_cost

    return loss, {}

  #Game Instance
  game = core.SymbolGameReinforce(sender, receiver, loss, sender_entropy_coeff=entropy_coefficient, receiver_entropy_coeff=0.0)
  optimizer = torch.optim.Adam(game.parameters())
  trainer = core.Trainer(
    game=game, optimizer=optimizer, train_data=train_loader,
    validation_data=test_loader)

  #Training
  n_epochs = 25
  trainer.train(n_epochs)

  #Sync wandb
  wandb.join()

#Correr sweep
wandb.agent(sweep_id, train, count=10)