# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

#Import Packages
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn import Linear,Sequential,Dropout
from torch.nn import ModuleList
from torch.distributions.bernoulli import Bernoulli

#Encoder
class Sender(nn.Module):
  """
  Module to encode a sequence into a rationale
  """

  def __init__(self,
               model,
               num_labels: int=11):

    super(Sender, self).__init__()
    #Transormer Model
    self.model = model
    #Rationale prediction head for each label
    self.rationale_heads = nn.ModuleList([nn.Linear(self.model.config.hidden_size, 1) for _ in range(num_labels)])

  def forward(self, x, _aux_input=None):
    """
    :param x: Inputs_ids (Shape -> [B,T])
    :return: Gate distribution and Mask
    """

    #Transformer outputs
    outputs = self.model(x)

    #Extract the hidden states
    hidden_states = outputs.last_hidden_state

    #Compute Bernoulli for each label using the respective rationale heads
    logits = torch.stack([rationale_head(hidden_states).squeeze(-1) for rationale_head in self.rationale_heads], dim=1)
    z_dist = Bernoulli(logits=logits)

    return z_dist

# Decoder
class Receiver(nn.Module):
  """
  Module to predict labels based on the rationales
  """

  def __init__(self,
               model,
               num_labels: int = 11,
               dropout: float = 0.1):
      super(Receiver, self).__init__()
      # Transformer Model
      self.model = model
      # Final classification layer
      self.output_layer = nn.Sequential(nn.Dropout(p=dropout),
                                        nn.Linear(self.model.config.hidden_size, num_labels),
                                        nn.Sigmoid())

  def forward(self, z, x, _aux_input=None):
      """
      :param z: Rationale (Shape -> [B, T])
      :param x: Sequence of word embeddings (Shape -> [B, T, E])
      :return: Label prediction (Shape -> [B,output_size])
      """

      # Transpose the rationale
      # z = z.transpose(1,2)

      # Pass each label-specific rationale through the decoder

      outputs = []

      for i in range(z.size(1)):
          # Select label-specific rationale
          rationale = z[:, i, :]

          # Encode the text
          model_outputs = self.model(x, attention_mask=rationale)
          pooled_output = model_outputs.pooler_output

          # Prediction from final states
          y = self.output_layer(pooled_output)
          y_label = y[:, i]

          outputs.append(y_label)

      # Concatenate the output probabilities along the label dimension
      output_probs = torch.stack(outputs, dim=1)

      return output_probs

# Rationale Wrapper
class RationaleWrapper(nn.Module):
  """
  This is a special Reinforce Wrapper for an agent.
  It takes an input text and return samples from p(z|x)

  Reinforce Wrapper for an agent. Assumes that the during the forward,
  the wrapped agent returns log-probabilities over the potential outputs. During training, the wrapper
  transforms them into a tuple of (sample from the Bernoulli, log-prob of the sample, entropy for the sample).
  Eval-time the sample is replaced with p(z|x) > 0.5 (Hyperparameter).
  """

  def __init__(self, agent):

      super(RationaleWrapper, self).__init__()
      self.agent = agent

  def forward(self, *args, **kwargs):
      z_dist = self.agent(*args, **kwargs)

      if self.training:
          z = z_dist.sample()
      else:
          # z = z_dist.sample()
          z = torch.where(z_dist.probs > 0.25, 1, 0)

      log_prob = z_dist.log_prob(z.float()).mean((1, 2))
      entropy = z_dist.entropy().mean((1, 2))
      z = z.squeeze(-1)

      return z, log_prob, entropy