from transformers import BertTokenizer, BertModel

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

text = 'MaxiCom is a great company.'

encoded_input = tokenizer(text, return_tensors='pt')
output = model(**encoded_input)


print("Keys:", output.keys())
print("Last hidden state shape:", output['last_hidden_state'].shape)
print("Pooler output shape:", output['pooler_output'].shape)
print("First token embedding:", output['last_hidden_state'][0, 0])  # e.g., [CLS] token
