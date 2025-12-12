import json

# Check for missing text fields
with open(r'c:\Users\pashi\Downloads\llm\Json\sample_context_vectors-01.json') as f:
    data = json.load(f)

items = data['data']['vector_data']
missing = [i for i, item in enumerate(items) if 'text' not in item or not item.get('text')]

print(f'Total items: {len(items)}')
print(f'Items missing/empty text: {len(missing)}')
if missing:
    print(f'Indices: {missing[:10]}')
    for idx in missing[:3]:
        print(f'\nItem {idx}: {items[idx]}')
