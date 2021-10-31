import torch
import torch.nn as nn
from transformers import BertPreTrainedModel, BertModel, AlbertPreTrainedModel, AlbertModel, DistilBertPreTrainedModel, DistilBertModel

class BertForSentimentClassification(BertPreTrainedModel):
	def __init__(self, config):
		super().__init__(config)
		self.bert = BertModel(config)
		# self.dropout = nn.Dropout(config.hidden_dropout_prob)
		self.cls_layer = nn.Linear(768, 1)

	def forward(self, input_ids, attention_mask):
		'''
		Inputs:
			-input_ids : Tensor of shape [B, T] containing token ids of sequences
			-attention_mask : Tensor of shape [B, T] containing attention masks to be used to avoid contibution of PAD tokens
			(where B is the batch size and T is the input length)
		'''
		#Feed the input to Bert model to obtain contextualized representations
		reps, _ = self.bert(input_ids=input_ids, attention_mask=attention_mask)
		# Obtain the representations of [CLS] heads
		# cls_reps = self.dropout(cls_reps)
		cls_reps = reps[:, 0]
		logits = self.cls_layer(cls_reps)
		return logits

class BertLSTMForSentimentClassification(BertPreTrainedModel):
	def __init__(self, config):
		super().__init__(config)
		self.bert = BertModel(config)
		self.bilstm1 = nn.LSTM(input_size=config.hidden_size,bidirectional=True,hidden_size=256)
		self.cls_layer = nn.Linear(512, 1)

	def forward(self, input_ids, attention_mask):

		reps, _ = self.bert(input_ids=input_ids, attention_mask=attention_mask)
		cls_reps,_ = self.bilstm1(reps)
		cls_reps = cls_reps[:, 0]
		logits = self.cls_layer(cls_reps)
		return logits
class BertCNNForSentimentClassification(BertPreTrainedModel):
	def __init__(self, config):
		super().__init__(config)
		self.bert = BertModel(config)
		self.conv1=nn.Conv2d(1,32,3,stride=3,padding=0)
		self.pool1=nn.MaxPool2d(kernel_size=2,stride=1)
		self.conv2=nn.Conv2d(32,config.hidden_size,3,stride=3,padding=0)
		self.pool2=nn.MaxPool2d(kernel_size=2,stride=1)
		self.cls_layer = nn.Linear(12*84, 1)

	def forward(self, input_ids, attention_mask):
		reps, _ = self.bert(input_ids=input_ids, attention_mask=attention_mask)
		cls_reps = reps.unsqueeze(1)
		cls_reps = self.conv1(cls_reps)
		cls_reps = self.pool1(cls_reps)
		cls_reps = self.conv2(cls_reps)
		cls_reps = self.pool2(cls_reps)
		cls_reps = cls_reps[:, 0]
		cls_reps = torch.flatten(cls_reps,start_dim=1)
		logits = self.cls_layer(cls_reps)
		return logits

class AlbertForSentimentClassification(AlbertPreTrainedModel):
	def __init__(self, config):
		super().__init__(config)
		self.albert = AlbertModel(config)
		#The classification layer that takes the [CLS] representation and outputs the logit
		self.cls_layer = nn.Linear(768, 1)

	def forward(self, input_ids, attention_mask):

		reps, _ = self.albert(input_ids=input_ids, attention_mask=attention_mask)
		cls_reps = reps[:, 0]
		logits = self.cls_layer(cls_reps)
		return logits

class DistilBertForSentimentClassification(DistilBertPreTrainedModel):
	def __init__(self, config):
		super().__init__(config)
		self.distilbert = DistilBertModel(config)
		self.cls_layer = nn.Linear(768, 1)

	def forward(self, input_ids, attention_mask):
		reps, = self.distilbert(input_ids=input_ids, attention_mask=attention_mask)
		cls_reps = reps[:, 0]
		logits = self.cls_layer(cls_reps)
		return logits