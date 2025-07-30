from transformers import pipeline

unmasker = pipeline('fill-mask', model='bert-base-uncased')
results = unmasker("Hello I'm a [MASK] model.")

for r in results:
    print(r)
