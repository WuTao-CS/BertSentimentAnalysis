import torch
from transformers import AutoTokenizer, AutoConfig,BertConfig,BertTokenizer
from modeling import BertForSentimentClassification, AlbertForSentimentClassification, DistilBertForSentimentClassification
from arguments import args

def classify_sentiment(sentence,model,tokenizer):
	with torch.no_grad():
		tokens = tokenizer.tokenize(sentence)
		tokens = ['[CLS]'] + tokens + ['[SEP]']
		input_ids = tokenizer.convert_tokens_to_ids(tokens)
		input_ids = torch.tensor(input_ids)
		input_ids = input_ids.unsqueeze(0)
		attention_mask = (input_ids != 0).long()
		logit = model(input_ids=input_ids, attention_mask=attention_mask)
		prob = torch.sigmoid(logit.unsqueeze(-1))
		prob = prob.item()
		soft_prob = prob > 0.5
		if soft_prob == 1:
			print('正向情绪，置信度为 {}%.'.format(int(prob*100)))
			result='正向情绪，置信度为 {}%.'.format(int(prob*100))
			return True,result
		else:
			print('负面情绪，置信度为 {}%.'.format(int(100-prob*100)))
			result='负面情绪，置信度为 {}%.'.format(int((1-prob)*100))
			return False,result
		

if __name__ == "__main__":

	if args.model_name_or_path is None:
		args.model_name_or_path = './models/64model/'

	#Configuration for the desired transformer model
	config = AutoConfig.from_pretrained(args.model_name_or_path)

	print('Please wait while the analyser is being prepared.')

	#Create the model with the desired transformer model
	if config.model_type == 'bert':
		model = BertForSentimentClassification.from_pretrained(args.model_name_or_path,config=config)
	elif config.model_type == 'albert':
		model = AlbertForSentimentClassification.from_pretrained(args.model_name_or_path)
	elif config.model_type == 'distilbert':
		model = DistilBertForSentimentClassification.from_pretrained(args.model_name_or_path)
	else:
		raise ValueError('This transformer model is not supported yet.')

	#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	#model = model.to(device)
	
	model.eval()

	#Initialize the tokenizer for the desired transformer model
	tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
	
	sentence = input('Input sentiment to analyze: ')
	while sentence:
		classify_sentiment(sentence)
		sentence = input('Input sentiment to analyze: ')