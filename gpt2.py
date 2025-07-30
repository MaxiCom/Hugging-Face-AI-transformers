from transformers import pipeline, set_seed

set_seed(42)

generator = pipeline('text-generation', model='gpt2')
output = generator('Hello, I\'m a language model,', max_length=30, num_return_sequences=5);

for line in output:
    print(line)
