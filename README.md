### Create venv
```bibtex
python -m venv .env
source .env/bin/activate          # Linux(bash/zsh)/MacOS
.env\Scripts\activate.bat  # Windows(cmd.exe) 
```

### Install
```bibtex
pip install fastapi uvicorn transformers torch sentencepiece langdetect
pip install -r requirements.txt
```

### Run
```bibtex
uvicorn app:app --reload --port 8000
python -m uvicorn app:app --reload --port 8000  # Windows
```
