---
title: Script Chunker
emoji: 🤖
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 6.5.1
app_file: app.py
pinned: false
---

# Script Chunker

Split scripts into `[SCENE XXX]` chunks for storyboard generation.

## Features

- Paste script text or upload a `.txt` file
- Adjustable words-per-scene slider
- Split mode toggle: `Basic` or `Context-Aware`
- Context-Aware mode uses spaCy sentence segmentation
- Copy output to clipboard
- Download chunked output as a text file

## Local Run

```bash
pip install -r requirements.txt
python app.py
```

## Hugging Face Spaces Deployment

1. Create a new Space on Hugging Face and choose **Gradio** SDK.
2. Upload these files to the Space root:
   - `app.py`
   - `script_chunker.py`
   - `requirements.txt`
   - `README.md`
3. The Space will build automatically and publish a public URL.
