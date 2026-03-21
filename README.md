# Resume Tailor

Local AI-powered resume tailoring with Ollama + Flask.

## Setup

```bash
# 1. Install Python deps
pip install -r requirements.txt

# 2. Make sure Ollama is running
ollama serve

# 3. Pull a model if you haven't (pick one)
ollama pull llama3.2        # fast, good quality
ollama pull llama3.3        # better quality, needs more RAM
ollama pull mistral         # solid alternative

# 4. Set your default model in app.py (line: DEFAULT_MODEL = "llama3.2")

# 5. Run the app
python app.py
```

Open http://localhost:5050 in your browser.

## Usage

1. Paste your full `.tex` resume in the left panel
2. Paste the job description
3. Add any specific skills you want injected
4. Hit **Generate** (or Cmd/Ctrl+Enter)
5. The output streams in real-time — modified lines are highlighted in green
6. Download the `.tex` file when done

## Tips

- The prompt tells the model to add `% MODIFIED` on changed lines — use `diff` or your editor to review changes
- Bigger models (llama3.3, mixtral) give better results but are slower
- If Ollama isn't detected (red dot top-right), run `ollama serve` first
