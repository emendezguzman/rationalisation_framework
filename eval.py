# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

#Import Libraries
from torchmetrics.classification import MultilabelF1Score

class MLEvaluator():

  #Create Object
  def __init__(self, input_ids, machine_rationales, labels_prob, human_rationales, labels, pred_threshold, model):
    self.input_ids = input_ids
    self.machine_rationales = machine_rationales
    self.labels_prob = labels_prob
    self.human_rationales = human_rationales
    self.labels = labels
    self.pred_labels = np.where(self.labels_prob >= pred_threshold, 1.0, 0.0)
    self.model = model

  #Evaluate Predictive Performance
  def evaluate_multilabel_performance(self):

    #Auxilary Dict
    metrics = dict()

    #Calculate Metrics
    metrics['LRAP'] = round(label_ranking_average_precision_score(self.labels, self.labels_prob))
    metrics['EMR'] = round(accuracy_score(self.labels, self.pred_labels),4)
    metrics['hamming_loss'] = round(hamming_loss(self.labels, self.pred_labels),4)
    metrics['f1_score'] = round(f1_score(self.labels, self.pred_labels, average='weighted', zero_division=0),4)
    metrics['f1_score_label'] = f1_score(self.labels, self.pred_labels, average=None, zero_division=0)

    return metrics

  #Calculate IoU at Token Level
  @staticmethod
  def calculate_rationale_match(machine_rationale, human_rationale, iou_threshold):

    #Formatting Rationales as Boolean Variables
    machine_rationale = machine_rationale.astype(bool)
    human_rationale = human_rationale.astype(bool)

    #Calculate IoU
    intersection = np.logical_and(machine_rationale, human_rationale)
    union = np.logical_or(machine_rationale, human_rationale)
    iou = np.sum(intersection) / np.sum(union)

    #Evaluating IoU against Threshold
    if iou >= iou_threshold:
      rationale_match = 1
    else:
      rationale_match = 0

    return rationale_match

  #Evaluate Plausibility - Multi Label
  def evaluate_plausibility(self, iou_threshold=0.5):

    #Create Auxiliary Dictionary
    matches_dict = dict()

    #Going through every example
    for i in range(self.human_rationales.shape[0]):

      #Going through every label
      for j in range(self.labels.shape[1]):

        #Checking that the prediction was correct at label level
        if self.labels[i,j] == self.pred_labels[i,j] == 1:

          #Calculate rationale match at a label level
          rationale_match = MLEvaluator.calculate_rationale_match(self.machine_rationales[i, j, :], self.human_rationales[i, j, :], iou_threshold)

          #Store match in label dictionary
          if matches_dict.get(j):
            matches_dict[j].append(rationale_match)

          else:
            matches_dict[j] = [rationale_match]

    return matches_dict

  #Calculate F1 Score at Label Level
  @staticmethod
  def calculate_plausibility_f1_score(matches_dict: dict):

    #Create Auxiliary Dictionary
    plausibility_f1_score = dict()
    n_examples = 0
    f1_score_agg = 0

    #Going through every label
    for label in matches_dict.keys():

      n_examples_label = len(matches_dict[label])
      ground_truth = n_examples_label * [1]
      plausibility_f1_score[label] = f1_score(ground_truth, matches_dict[label])

      #Update n_examples and f1_score_agg
      n_examples += n_examples_label
      f1_score_agg += plausibility_f1_score[label] * n_examples_label

    #Calculate Weighted F1 Score
    f1_score_weighted = f1_score_agg / n_examples

    return f1_score_weighted, plausibility_f1_score

  #Evaluate Explanation Quality

  #Calculate Predictions (Full Text)
  def calculate_pred_full_text(self):

    with torch.no_grad():

      input_ids = torch.tensor(self.input_ids).to(device)
      attention_ones = torch.ones(self.machine_rationales.shape).to(device)
      y_pred_full_text = self.model(attention_ones, input_ids)

    return y_pred_full_text.cpu()

  #Calculate Predictions (Full Text / Rationales)
  def calculate_pred_rationale_comp(self):

    with torch.no_grad():

      input_ids = torch.tensor(self.input_ids).to(device)
      attention_ones = torch.ones(self.machine_rationales.shape)
      machine_rationales = torch.tensor(self.machine_rationales)
      attention_comp_rationales = (attention_ones - machine_rationales).to(device)
      y_pred_comp_rationales = self.model(attention_comp_rationales, input_ids)

    return y_pred_comp_rationales.cpu()

  #Comprehensiveness (Multi-Label)
  def calculate_comprehensiveness(self):

    #Calculate Prediction Full Text
    y_pred_full_text = MLEvaluator.calculate_pred_full_text(self)

    #Calculate Prediction Prediction Full Text / Rationales
    y_pred_comp_rationales = MLEvaluator.calculate_pred_rationale_comp(self)

    #Identifying matches between Labels and Predicted Labels
    matches = np.logical_and(self.labels, self.pred_labels)
    n_label = matches.sum(0)

    #Calculate Comprehensiveness per Label
    comp_per_label = list()

    for i in range(matches.shape[1]):

      #Calculate Diff between y_red and y_pred_comp_rationales
      diff = y_pred_full_text[:, i] - y_pred_comp_rationales[:, i]
      diff_matches = diff[matches[:, i]]

      #Calculate max(0, diff between probabilities)
      comp = torch.max(diff_matches, torch.zeros_like(diff_matches))

      #Calculate average comprehensiveness per label
      comp_label = round(torch.mean(comp).cpu().numpy().item(),4)

      #Checking whether there is no match and replace with zero
      if np.isnan(comp_label) == True:
        comp_per_label.append(0)
      else:
        comp_per_label.append(comp_label)

    wei_average_comp = round(np.dot(n_label, comp_per_label) / n_label.sum(),4)

    return wei_average_comp, comp_per_label

  #Sufficiency
  def calculate_sufficiency(self):

    #Calculate Prediction Full Text
    y_pred_full_text = MLEvaluator.calculate_pred_full_text(self)

    #Prediction Rationales Only
    y_pred_rationales = torch.tensor(self.labels_prob)

    #Calculate Sufficiency per Label
    suff_per_label = list()

    #Identifying matches between labels and predicted labels
    matches = np.logical_and(self.labels, self.pred_labels)
    n_label = matches.sum(0)

    for i in range(matches.shape[1]):

      #Calculate Diff between y_red and y_pred_comp_rationales
      diff = y_pred_full_text[:, i] - y_pred_rationales[:, i]
      diff_matches = diff[matches[:, i]]

      #Calculate max(0, diff between probabilities)
      max_suff = torch.max(diff_matches, torch.zeros_like(diff_matches))

      #Calculate 1 - max(0, diff between probabilities)
      suff = torch.ones_like(max_suff) - max_suff

      #Calculate average comprehensiveness per label
      suff_label = round(torch.mean(suff).cpu().numpy().item(),4)

      #Checking whether there is no match and replace with zero
      if np.isnan(suff_label) == True:
        suff_per_label.append(0)
      else:
        suff_per_label.append(suff_label)

    wei_average_suff = round(np.dot(n_label, suff_per_label) / n_label.sum(),4)

    return wei_average_suff, suff_per_label

# Test Set Evaluation (Single Batch - Test Dataset)
test_sender_input, test_labels, test_receiver_input, test_aux_input = [], [], [], []
test_dataset_len = test_loader.dataset.__len__()

for z in range(test_dataset_len):
  sender_input, labels, receiver_input, _aux_input = test_loader.dataset[z]
  test_sender_input.append(sender_input)
  test_labels.append(labels)
  test_receiver_input.append(receiver_input)
  test_aux_input.append(_aux_input['attention'])

test_sender_input = torch.stack(test_sender_input)
test_labels = torch.stack(test_labels)
test_receiver_input = torch.stack(test_receiver_input)
test_aux_input = {'attention': torch.stack(test_aux_input)}

test_dataset = [[test_sender_input, test_labels, test_receiver_input, test_aux_input]]

# Dump Game Interactions
interaction = \
  core.dump_interactions(game, test_dataset, gs=False, variable_length=False)

#Data Transformation
input_ids = test_sender_input.numpy()
labels = test_labels.numpy()
human_rationales = test_aux_input['attention'].cpu().numpy()
machine_rationales = interaction.message.numpy()
labels_prob = interaction.receiver_output.numpy()

#Retrieving Model
model = game.receiver.agent

#Create Object
evaluator = MLEvaluator(input_ids=input_ids,
                        machine_rationales=machine_rationales,
                        labels_prob=labels_prob,
                        human_rationales=human_rationales,
                        labels=labels,
                        pred_threshold=0.25,
                        model=model)

#Evaluate Predictive Performance
metrics = evaluator.evaluate_multilabel_performance()
print(f'Predictive Performance Metrics:')
pprint(metrics)

#Evaluate Plausbility
rationale_matches = evaluator.evaluate_plausibility(iou_threshold=0.5)
f1_score_weighted, f1_score_label = MLEvaluator.calculate_plausibility_f1_score(matches_dict=rationale_matches)
print('F1 Score (Weighted):', round(f1_score_weighted, 3))
print('F1 Score (Labels):', f1_score_label)

#Calculate Predictions Full Text
y_pred_full_text = evaluator.calculate_pred_full_text()

#Evaluate Comprehensiveness
average_comp, comp_per_label = evaluator.calculate_comprehensiveness()
print('Overall Comprehensiveness:,', average_comp)
print('Comprehensiveness per Label:,', comp_per_label)

#Evaluate Sufficiency
average_suff, suff_per_label = evaluator.calculate_sufficiency()
print('Overall Sufficiency:,', average_suff)
print('Sufficiency per Label:,', suff_per_label)