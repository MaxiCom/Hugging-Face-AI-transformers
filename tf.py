from transformers import BertTokenizer, TFBertModel

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = TFBertModel.from_pretrained('bert-base-uncased')

text = 'MaxiCom is a great company.'

encoded_input = tokenizer(text, return_tensors='tf')
output = model(encoded_input)
