# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

#Import Packages
import json
import re
from huggingface_hub import notebook_login
from transformers import AutoTokenizer, BertTokenizer, PreTrainedTokenizerFast
import numpy as np

class RationaleDataset():

    #OBJECT CREATION

    def __init__(self, texts, labels, human_rationales, num_labels, labels_list):
      self.texts = texts
      self.labels = labels
      self.human_rationales = human_rationales
      self.num_labels = num_labels
      self.name_labels = labels_list

    @classmethod
    def from_json(cls, data_path: str, annotator_email = None):
      """

      """

      #Read JSON File
      with open(data_path, 'r') as f:
          data = json.load(f)

      #Retrieve Text
      texts = list()
      for article in data['examples']:
          texts.append(article['content'])

      #Create and Sort Labels List
      labels_list = list()
      for tag in data['schema']['tags']:
          labels_list.append(tag['name'])

      labels_list.sort()

      #Generate Label Vector
      num_labels = len(labels_list)
      labels = list()

      #One Annotator per Example
      if annotator_email == None:
        for article in data['examples']:
            label = [0] * num_labels
            for annotation in article['annotations']:
                index = labels_list.index(annotation['tag'])
                if label[index] == 0:
                    label[index] = 1
            labels.append(label)

      #Multiple Annotators per Example (Ground Truth = Annotator Email)
      else:
        for article in data['examples']:
          label = [0] * num_labels
          for annotation in article['annotations']:
            if annotation['annotated_by'][0]['annotator'] == annotator_email:
              index = labels_list.index(annotation['tag'])
              if label[index] == 0:
                label[index] = 1
          labels.append(label)

      #Retrieve Rationales
      human_rationales = list()

      #1 Annotator
      if annotator_email == None:
        for article in data['examples']:
            rationale = dict()
            for annotation in article['annotations']:
                if rationale.get(annotation['tag']) == None:
                    rationale[annotation['tag']] = [(annotation['start'], annotation['end'], annotation['value'])]
                else:
                    rationale[annotation['tag']].append((annotation['start'], annotation['end'], annotation['value']))

            human_rationales.append(rationale)

      else:
        for article in data['examples']:
            rationale = dict()
            for annotation in article['annotations']:
              if annotation['annotated_by'][0]['annotator'] == annotator_email:
                if rationale.get(annotation['tag']) == None:
                    rationale[annotation['tag']] = [(annotation['start'], annotation['end'], annotation['value'])]
                else:
                    rationale[annotation['tag']].append((annotation['start'], annotation['end'], annotation['value']))

            human_rationales.append(rationale)

      return cls(texts, labels, human_rationales, num_labels, labels_list)

    #FILTERING EXAMPLES (NO LABEL)
    def removing_no_labels(self):

      texts_labelled = list()
      labels_labelled = list()
      human_rationales_labelled = list()

      for i in range(len(self.labels)):

        if self.labels[i] != [0] * self.num_labels:

          texts_labelled.append(self.texts[i])
          labels_labelled.append(self.labels[i])
          human_rationales_labelled.append(self.human_rationales[i])

      self.texts = texts_labelled
      self.labels = labels_labelled
      self.human_rationales = human_rationales_labelled

    #GENERATING INPUT IDS
    def generating_encodings(self, tokenizer, max_length):
      """

      """
      #Adding PAD Token
      if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})

      input_ids = list()
      offset_mapping = list()

      for text in self.texts:

        encodings = tokenizer(text,
                              add_special_tokens=False,
                              max_length=max_length,
                              padding='max_length',
                              truncation=True,
                              return_offsets_mapping=True)

        encodings_input_ids = encodings['input_ids']
        encodings_offset_mapping = encodings['offset_mapping']

        input_ids.append(encodings_input_ids)
        offset_mapping.append(encodings_offset_mapping)

      return input_ids, offset_mapping

    #CONSOLIDATING HUMAN RATIONALES INDEXES (Multi-Label)
    def consolidate_human_rationales(self):

      #Auxiliary List
      consolidated_human_rationales = list()

      #Going through every example
      for i in range(len(self.human_rationales)):

        indexes_item = list()
        label_indexes = dict()

      #Going through every label
        for label in self.name_labels:

          #Checking rationales per label
          if self.human_rationales[i].get(label):
            indexes = [(item[0], item[1]) for item in self.human_rationales[i][label]]

          else:
            indexes = []

          #Storing indexes per label
          label_indexes[label] = indexes

        #Storing indexes per item
        consolidated_human_rationales.append(label_indexes)

      return consolidated_human_rationales

    #CREATING HUMAN RATIONALES ATTENTION MASK (Multi-Label)
    @staticmethod
    def generate_rationales_attention(offsets_mapping, consolidated_human_rationales):

      #Auxiliary list
      rationales_attentions = list()

      #Going through every example
      for i in range(len(offsets_mapping)):

        #Auxiliary list
        rationales_attentions_item = list()

        #Going through every label
        for label in consolidated_human_rationales[i].keys():

          attention_mask_label = [0] * len(offsets_mapping[i])

          for j, (start, end) in enumerate(offsets_mapping[i]):

            #Checking rationales per label
            if len(consolidated_human_rationales[i][label]) > 0:

              for k in range(len(consolidated_human_rationales[i][label])):

                rationale_start = consolidated_human_rationales[i][label][k][0]
                rationale_end = consolidated_human_rationales[i][label][k][1]

                if (start >= rationale_start and end <= rationale_end):

                  attention_mask_label[j] = 1

          #Storing rationale mask per label
          rationales_attentions_item.append(attention_mask_label)

        #Storing rationale mask per item
        rationales_attentions.append(rationales_attentions_item)

      return rationales_attentions