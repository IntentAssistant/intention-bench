# IntentionBench Code Repository

This repository hosts the scripts used to collect, synthesize, and evaluate the **IntentionBench** dataset introduced in the paper  
_[“State Your Intention to Steer Your Attention: An AI Assistant for Intentional Digital Living.”](https://arxiv.org/abs/2510.14513)_

The dataset itself is published on Hugging Face. Clone this repo for the tooling, then download the data from Hugging Face and place it under `intention_bench/dataset/` following the structure described below.

> **Dataset on Hugging Face**  
> https://huggingface.co/datasets/juheonch/intention_bench

---

## Repository Structure

```
intention_bench/
├── code/
│   ├── analysis.py              # end-to-end evaluation pipeline
│   ├── bulid_mixed_sessions.py    # synthesizes mixed sessions
│   ├── screen_capture_tool.py     # GUI for recording focused sessions
│   ├── utils/                     # shared helpers (dataset loading, metrics, logging)
│   └── api_config.example.json    # template for Gemini API credentials
└── dataset/
    ├── README.md                  # dataset card text (English)
    └── annotations/metadata/...   # intentions.csv, clarify_stated_intentions.json, manifest
```

The raw images and synthesized JSON files are **not** tracked in GitHub. After downloading them from Hugging Face, place them in:

- `intention_bench/dataset/images/` (created by unzipping `images.zip`)
- `intention_bench/dataset/annotations/mixed_sessions/raw_jsons/`

See `dataset/README.md` for detailed documentation of the file layout.

---

## Setup

1. **Clone this repository**
   ```bash
   git clone git@github.com:IntentAssistant/intention-bench.git
   cd intention-bench
   ```

2. **Create a virtual environment and install dependencies**
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```
   (Provide `requirements.txt` according to your environment.)

3. **Configure API access**
   - Copy `intention-bench/code/api_config.example.json` to `api_config.json`.
   - Fill in your Gemini API key, GCP project ID, and preferred location.
   ```bash
   cp code/api_config.example.json code/api_config.json
   ```

4. **Download the dataset from Hugging Face**
   ```bash
   hf download juheonch/intention_bench \
     --repo-type dataset \
     --local-dir ./dataset
   unzip dataset/images.zip -d dataset
   ```
   After extraction, the directory layout matches the paths referenced by the mixed-session JSON files.

---

## Running the End-to-End Evaluation

```bash
python intention_bench/code/analysis.py \
  --api_config intention_bench/code/api_config.json \
  --data_dir intention_bench/dataset/annotations/mixed_sessions/raw_jsons \
  --dataset_config intention_bench/dataset/annotations/metadata/config/intentions.csv \
  --clarify_intentions intention_bench/dataset/annotations/metadata/config/clarify_stated_intentions.json \
  --model_name gemini-2.5-flash-lite \
  --max_workers 36 \
  --sample_rate 100 \
  --experiment_name demo_run
```

Results and logs are written to `intention_bench/results/` and `intention_bench/logs/`.

---

## Re-synthesizing Mixed Sessions

If you want to generate new mixed sessions from the focused session archives:

```bash
python intention_bench/code/bulid_mixed_sessions.py \
  --trajectory_dir intention_bench/dataset/images \
  --output_dir intention_bench/dataset/annotations/mixed_sessions/raw_jsons \
  --data_mode all \
  --num_samples 50 \
  --seed 42
```

---

## Screen Capture Tool

`screen_capture_tool.py` provides a PyQt GUI for recording new focused sessions.  
Captured frames and metadata are stored under `intention_bench/dataset/images/<TASK_ID>/`.

---

## License & Citation

Please consult the dataset card on Hugging Face for licensing terms.  
When using IntentionBench, cite the original paper:

```
State Your Intention to Steer Your Attention: An AI Assistant for Intentional Digital Living
```

---

## Responsible Use

The focused sessions were created by the authors to mimic realistic workflows.  
Do not attempt to reconstruct sensitive personal information.  
Ensure compliance with the Gemini API terms of service and applicable privacy regulations when extending this dataset.
