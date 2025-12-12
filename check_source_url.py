import json

# Check for invalid source_url fields
with open(r'c:\Users\pashi\Downloads\llm\Json\sample_context_vectors-01.json') as f:
    data = json.load(f)

items = data['data']['vector_data']

for i, item in enumerate(items):
    if not item.get('text'):
        continue
    source_url = item.get('source_url')
    if not isinstance(source_url, str):
        print(f'Item {i}: source_url type = {type(source_url)}, value = {source_url}')
