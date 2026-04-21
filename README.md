# mBART-51

A speech-to-speech Spanish → English translator built for **code-switching and Spanglish**, the kind of natural bilingual speech that off-the-shelf models tend to mangle. The pipeline transcribes audio with a Spanglish-tuned Whisper, translates with a fine-tuned mBART-50, and resynthesizes the translation with Kokoro TTS — all wrapped in a small Streamlit app.

This repository is open source and contributions are welcome. The notes below are aimed at first-time contributors.

## Pipeline

```
audio in  →  Whisper Small (Spanglish-tuned)  →  mBART-50 (fine-tuned)  →  Kokoro TTS  →  audio out
```

Hosted model artifacts on Hugging Face:

- ASR: [`drewoodward/whisper-small-spanglish`](https://huggingface.co/drewoodward/whisper-small-spanglish)
- MT:  [`drewoodward/mBART-51`](https://huggingface.co/drewoodward/mBART-51)
- TTS: `kokoro` (`KPipeline`, `lang_code="a"`)

## Repository layout

| Path | What it is |
| --- | --- |
| [src/app.py](src/app.py) | Streamlit UI tying ASR → MT → TTS together |
| [src/cha_to_vtt.py](src/cha_to_vtt.py) | Converts CHAT (`.cha`) transcripts from the Bangor Miami corpus into Whisper-ready `.vtt` |
| [src/merge_corpus.py](src/merge_corpus.py) | Concatenates per-conversation `.mp3` + `.vtt` pairs into one merged training file (uses `ffmpeg`/`ffprobe`) |
| [src/fine_tune_whisper.ipynb](src/fine_tune_whisper.ipynb) | Notebook used to fine-tune Whisper Small on the merged Spanglish corpus |
| [src/requirements.txt](src/requirements.txt) | Python dependencies for the app |
| [miamiCorpus/](miamiCorpus/) | Per-conversation `.mp3` + `.vtt` pairs (Bangor Miami subset, after CHAT cleaning) |
| [content/](content/) | Sample audio + transcript for quick local testing |

`checkpoint-*/` and `model.safetensors` are gitignored — pull weights from the Hugging Face repos above instead of committing them.

## Getting started

Requires Python 3.10+ and `ffmpeg` on your `PATH` (used by Whisper and by `merge_corpus.py`).

```bash
git clone <your-fork-url>
cd mBART-51

python -m venv venv
source venv/bin/activate
pip install -r src/requirements.txt

streamlit run src/app.py
```

The first run downloads the Whisper, mBART, and Kokoro weights from Hugging Face — expect a few GB on disk and a slow first launch. CUDA is used automatically if available; otherwise everything falls back to CPU.

## Working with the corpus

Training data comes from the **Bangor Miami Corpus** of Spanish–English bilingual conversation. The raw CHAT (`.cha`) files are not redistributed here; the `.vtt` files in [miamiCorpus/](miamiCorpus/) are the output of running [src/cha_to_vtt.py](src/cha_to_vtt.py) on them.

Typical workflow when adding a new conversation:

1. Drop the original `.cha` and `.mp3` somewhere on disk.
2. Edit the file paths at the bottom of [src/cha_to_vtt.py](src/cha_to_vtt.py) and run it to produce a cleaned `.vtt`.
3. Place the `.mp3` + `.vtt` pair in a new subfolder under [miamiCorpus/](miamiCorpus/) (one folder per conversation, matching names).
4. Run [src/merge_corpus.py](src/merge_corpus.py) to regenerate `miamiCorpus_merged.mp3` / `.vtt` with cumulative timestamp offsets.
5. Re-run the relevant cells in [src/fine_tune_whisper.ipynb](src/fine_tune_whisper.ipynb).

The hardcoded paths in `cha_to_vtt.py` are a known rough edge — see *Good first issues* below.

## Contributing

1. Fork the repo and create a feature branch off `master`.
2. Keep changes focused — one PR per logical change.
3. If you touch the model loading or audio pipeline in [src/app.py](src/app.py), smoke-test by running the Streamlit app on the sample file in [content/herring3.mp3](content/herring3.mp3) end-to-end.
4. Open a PR describing **what** changed and **why**; link any relevant corpus or model changes.

### Good first issues

- Replace the hardcoded paths at the bottom of [src/cha_to_vtt.py](src/cha_to_vtt.py) with a small CLI (`argparse`).
- Pin versions in [src/requirements.txt](src/requirements.txt) and split out a `requirements-train.txt` for the notebook-only deps.
- Add a `--device` / model-id override to [src/app.py](src/app.py) so contributors without GPUs can swap in `whisper-tiny` for faster iteration.
- Convert [src/fine_tune_whisper.ipynb](src/fine_tune_whisper.ipynb) into a reproducible script.
- Add evaluation metrics (WER for ASR, BLEU/chrF for MT) on a held-out slice of the Miami corpus.

### Code style

- Standard PEP 8; small, readable functions over clever ones.
- No new top-level dependencies without a note in the PR explaining why.
- Don't commit model weights, audio over a few MB, or anything under `checkpoint-*/`.

## License & data

Code in this repository is released for open contribution. The Bangor Miami Corpus has its own license terms — please consult the [TalkBank/Bangor Miami documentation](https://biling.talkbank.org/access/Bangor/Miami.html) before redistributing the underlying transcripts or audio.
