# Quick Start Guide

## 1. Initial Setup

```bash
# On Linux/Mac
chmod +x setup.sh
./setup.sh

# On Windows
powershell -ExecutionPolicy Bypass -File setup.ps1
```

## 2. Run with Sample Files

The project can use the sample JSON files from the `../Json` folder:

```bash
# Activate virtual environment first
# On Linux/Mac:
source venv/bin/activate
# On Windows:
venv\Scripts\activate

# Run evaluation on sample 1
python -m src.cli evaluate \
  -c ../Json/sample-chat-conversation-01.json \
  -x ../Json/sample_context_vectors-01.json \
  -o output/results-01.json \
  -v

# Or use the helper script
python run_sample_evaluation.py
```

## 3. View Results

```bash
# Check the output file
cat output/results-01.json

# View database statistics
python -m src.cli stats
```

## 4. Run Example Script

```bash
python examples/run_example.py
```

## 5. Run Tests

```bash
pytest tests/ -v
```

## Configuration

Edit `.env` file to configure:
- LLM provider (mock or openai)
- API keys
- Evaluation thresholds
- Cost parameters

## Docker

```bash
# Build
docker build -t llm-eval .

# Run
docker run -v $(pwd)/../Json:/input -v $(pwd)/output:/output \
  llm-eval evaluate \
  -c /input/sample-chat-conversation-01.json \
  -x /input/sample_context_vectors-01.json \
  -o /output/results.json
```

## Troubleshooting

### Module not found errors
```bash
# Ensure you're in the project root and venv is activated
cd llm-eval-pipeline
source venv/bin/activate  # or venv\Scripts\activate on Windows
```

### OpenAI API errors
```bash
# Make sure API key is set
export OPENAI_API_KEY=sk-...
# Or edit .env file
```

### Import errors in tests
```bash
# Install in editable mode
pip install -e .
```

## Next Steps

1. ✅ Run with provided samples
2. 📝 Try with your own conversation/context JSONs
3. 🔧 Adjust thresholds in `.env`
4. 📊 Analyze results in database
5. 🚀 Deploy with Docker

For detailed documentation, see [README.md](README.md)
