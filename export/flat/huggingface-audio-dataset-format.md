# Hugging Face Audio Dataset Format: The Standard for ASR Fine-Tuning

## Question
What is the standard audio dataset format used on Hugging Face (where transcriptions are in JSON metadata rather than separate text files)? What's it called, and where can you find the exact definition?

## Answer

Great observation! You've discovered the **Hugging Face Datasets format**, specifically the **Audio Dataset format** (also called **`datasets` Audio feature type**). Let's explain the structure and where to find the official spec.

---

## The Standard: Hugging Face `datasets` Audio Format

### **What It Is**

Hugging Face has standardized a dataset format for ML/AI that's become the de facto standard for ASR (and other) datasets. The key insight:

**Instead of:**
```
/dataset
  /audio
    file1.wav
    file2.wav
  /text
    file1.txt
    file2.txt
```

**The standard uses:**
```
/dataset
  /audio
    file1.wav
    file2.wav
  metadata.jsonl  (or metadata.csv, or data.arrow)
```

Where `metadata.jsonl` contains:
```jsonl
{"audio": "audio/file1.wav", "text": "This is the transcription", "speaker_id": 1}
{"audio": "audio/file2.wav", "text": "Another transcription", "speaker_id": 2}
```

**Or using Hugging Face's `datasets` library directly (recommended):**
```python
from datasets import Dataset, Audio

dataset = Dataset.from_dict({
    "audio": ["audio/file1.wav", "audio/file2.wav"],
    "text": ["This is the transcription", "Another transcription"],
})

# Cast audio column to Audio feature
dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))
```

---

## Why This Format?

### **Benefits Over Separate Text Files:**

1. **Single Source of Truth**: All metadata in one place (JSON/CSV/Arrow)
2. **Easier Iteration**: Load with one command, no manual file matching
3. **Atomic**: Audio + transcription + metadata together (can't get out of sync)
4. **Lazy Loading**: Datasets library loads audio on-demand (memory efficient)
5. **Streaming**: Can stream from remote (no need to download entire dataset)
6. **Standardization**: Works across Hugging Face ecosystem (Transformers, Datasets, Hub)

### **Traditional Separate Files:**
```python
# Manual matching required
audio_files = glob("audio/*.wav")
text_files = glob("text/*.txt")

# Hope they match!
for audio, text in zip(sorted(audio_files), sorted(text_files)):
    # ... load and process
```

**Error-prone**: Easy to get mismatched files if one is missing or renamed.

### **Hugging Face Format:**
```python
from datasets import load_dataset

dataset = load_dataset("audiofolder", data_dir="path/to/dataset")

# Everything is aligned automatically
for example in dataset:
    audio = example["audio"]["array"]  # numpy array
    text = example["text"]  # transcription
```

**Safe**: Audio-text pairs guaranteed to match.

---

## The Format Details

### **Option 1: `audiofolder` Format (Simplest)**

This is the most common for local datasets:

**Directory Structure:**
```
my_dataset/
├── metadata.csv  (or metadata.jsonl)
└── audio/
    ├── file1.wav
    ├── file2.wav
    └── ...
```

**metadata.csv:**
```csv
file_name,text
audio/file1.wav,This is the transcription for file one
audio/file2.wav,This is the transcription for file two
```

**Or metadata.jsonl (JSON Lines):**
```jsonl
{"file_name": "audio/file1.wav", "text": "This is the transcription for file one"}
{"file_name": "audio/file2.wav", "text": "This is the transcription for file two"}
```

**Loading:**
```python
from datasets import load_dataset

dataset = load_dataset("audiofolder", data_dir="my_dataset")

print(dataset)
# DatasetDict({
#     train: Dataset({
#         features: ['audio', 'text'],
#         num_rows: 2
#     })
# })
```

**Key Details:**
- Column `file_name` (or `audio`) points to audio files
- Column `text` contains transcriptions
- Additional columns allowed (speaker_id, duration, etc.)
- Audio automatically loaded as `Audio` feature type

---

### **Option 2: Hugging Face Hub Format (For Uploading)**

When uploading to Hugging Face Hub, use this structure:

**Directory Structure:**
```
my_asr_dataset/
├── README.md  (dataset card)
├── data/
│   ├── train/
│   │   ├── metadata.csv
│   │   └── audio/
│   │       ├── file1.wav
│   │       └── ...
│   ├── validation/
│   │   ├── metadata.csv
│   │   └── audio/
│   └── test/
│       ├── metadata.csv
│       └── audio/
```

**Or using Arrow files (more efficient):**
```
my_asr_dataset/
├── README.md
├── train.arrow
├── validation.arrow
└── test.arrow
```

**Loading from Hub:**
```python
dataset = load_dataset("your-username/my_asr_dataset")
```

---

### **Option 3: Direct Arrow Format (Most Efficient)**

For large datasets, Hugging Face uses **Apache Arrow**:

```python
from datasets import Dataset, Audio

# Create dataset
dataset = Dataset.from_dict({
    "audio": ["file1.wav", "file2.wav"],
    "text": ["transcription 1", "transcription 2"],
})

# Cast audio column
dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))

# Save as Arrow
dataset.save_to_disk("dataset.arrow")

# Load
dataset = Dataset.load_from_disk("dataset.arrow")
```

**Benefits:**
- Fast loading (mmap-based)
- Memory efficient
- No CSV/JSON parsing overhead

---

## The "Audio" Feature Type

**The key to the format is the `Audio` feature**:

### **What It Does:**

When you load a dataset with an `Audio` column:

```python
example = dataset[0]

# Audio is automatically loaded and decoded
example["audio"]
# {
#     'path': 'audio/file1.wav',
#     'array': array([0.1, 0.2, ...]),  # numpy array
#     'sampling_rate': 16000
# }
```

**Under the hood:**
- Stores path to audio file
- Lazy-loads audio (only loads when accessed)
- Automatically decodes (WAV, MP3, FLAC, etc.)
- Resamples to target sampling rate if needed

**This is why transcriptions go in metadata**: The audio files are referenced, not embedded.

---

## Official Documentation

### **Where to Find the Exact Definition:**

#### **1. Hugging Face Datasets Documentation**

**Main page:**
[https://huggingface.co/docs/datasets](https://huggingface.co/docs/datasets)

**Audio-specific docs:**
[https://huggingface.co/docs/datasets/audio_dataset](https://huggingface.co/docs/datasets/audio_dataset)

**Audio feature docs:**
[https://huggingface.co/docs/datasets/package_reference/main_classes#datasets.Audio](https://huggingface.co/docs/datasets/package_reference/main_classes#datasets.Audio)

**`audiofolder` format:**
[https://huggingface.co/docs/datasets/audio_load#audiofolder](https://huggingface.co/docs/datasets/audio_load#audiofolder)

#### **2. Example Datasets (Reference Implementations)**

**Common Voice (Mozilla):**
[https://huggingface.co/datasets/mozilla-foundation/common_voice_17_0](https://huggingface.co/datasets/mozilla-foundation/common_voice_17_0)

**LibriSpeech:**
[https://huggingface.co/datasets/librispeech_asr](https://huggingface.co/datasets/librispeech_asr)

**GigaSpeech:**
[https://huggingface.co/datasets/speechcolab/gigaspeech](https://huggingface.co/datasets/speechcolab/gigaspeech)

Browse these datasets' file structures on the "Files and versions" tab.

#### **3. Dataset Card Template**

Hugging Face provides a template:
[https://github.com/huggingface/datasets/blob/main/templates/README.md](https://github.com/huggingface/datasets/blob/main/templates/README.md)

#### **4. GitHub Repos**

**Datasets library source code:**
[https://github.com/huggingface/datasets](https://github.com/huggingface/datasets)

**Audio feature implementation:**
[https://github.com/huggingface/datasets/blob/main/src/datasets/features/audio.py](https://github.com/huggingface/datasets/blob/main/src/datasets/features/audio.py)

---

## Creating Your Own Dataset (Practical Guide)

### **Step 1: Organize Audio Files**

```bash
my_dataset/
└── audio/
    ├── speaker1_utterance1.wav
    ├── speaker1_utterance2.wav
    └── ...
```

### **Step 2: Create metadata.csv**

```python
import pandas as pd

data = {
    "file_name": [
        "audio/speaker1_utterance1.wav",
        "audio/speaker1_utterance2.wav",
    ],
    "text": [
        "This is the first transcription",
        "This is the second transcription",
    ],
    "speaker_id": ["speaker1", "speaker1"],  # Optional metadata
    "duration": [3.2, 4.1],  # Optional metadata
}

df = pd.DataFrame(data)
df.to_csv("my_dataset/metadata.csv", index=False)
```

### **Step 3: Load as Hugging Face Dataset**

```python
from datasets import load_dataset

dataset = load_dataset("audiofolder", data_dir="my_dataset", split="train")

# Verify
print(dataset[0])
# {
#     'audio': {
#         'path': 'my_dataset/audio/speaker1_utterance1.wav',
#         'array': array([...]),
#         'sampling_rate': 16000
#     },
#     'text': 'This is the first transcription',
#     'speaker_id': 'speaker1',
#     'duration': 3.2
# }
```

### **Step 4: (Optional) Upload to Hugging Face Hub**

```python
from huggingface_hub import create_repo, upload_folder

# Create repo
create_repo("your-username/my-asr-dataset", repo_type="dataset")

# Upload
dataset.push_to_hub("your-username/my-asr-dataset")
```

---

## Schema Definition (The "Exact Specification")

**There's no single RFC-style spec document**, but the format is defined by:

### **Minimum Required Schema (audiofolder):**

```python
{
    "audio": Audio(sampling_rate=16000),  # or other rates
    "text": Value("string"),
}
```

**Extended Schema (Common):**

```python
{
    "audio": Audio(sampling_rate=16000),
    "text": Value("string"),
    "speaker_id": Value("string"),  # Optional
    "chapter_id": Value("int64"),    # Optional
    "id": Value("string"),           # Optional
    "duration": Value("float32"),    # Optional
    # ... any other metadata
}
```

**The only hard requirements:**
1. A column with audio file paths (typically `audio` or `file_name`)
2. That column cast to `Audio()` feature type
3. (For ASR) A column with transcriptions (typically `text` or `transcription`)

**Everything else is flexible.**

---

## Common Variations

### **For Multi-Split Datasets (train/val/test):**

**Option A: Separate directories**
```
dataset/
├── train/
│   ├── metadata.csv
│   └── audio/
├── validation/
│   ├── metadata.csv
│   └── audio/
└── test/
    ├── metadata.csv
    └── audio/
```

**Load:**
```python
dataset = load_dataset("audiofolder", data_dir="dataset")
# Automatically detects splits
```

**Option B: Single metadata with split column**
```csv
file_name,text,split
audio/file1.wav,transcription 1,train
audio/file2.wav,transcription 2,train
audio/file3.wav,transcription 3,validation
```

**Load:**
```python
from datasets import load_dataset

dataset = load_dataset("csv", data_files="dataset/metadata.csv")
dataset = dataset.train_test_split(test_size=0.1)  # Manual split
```

---

## Why JSON/CSV Instead of Separate Text Files?

**You asked about the shift from individual text files:**

### **Separate Text Files (Old Approach):**
```
dataset/
├── audio/
│   ├── file1.wav
│   └── file2.wav
└── text/
    ├── file1.txt
    └── file2.txt
```

**Problems:**
1. **Manual matching**: Need code to pair files correctly
2. **Fragility**: Renaming/deleting one file breaks dataset
3. **No atomic operations**: Can't update transcription + metadata together
4. **Poor performance**: Reading thousands of small text files is slow
5. **No schema validation**: Each text file is independent (no structure)

### **Metadata-Based (New Approach):**
```
dataset/
├── metadata.csv
└── audio/
    ├── file1.wav
    └── file2.wav
```

**Benefits:**
1. **Automatic pairing**: Column-based, no manual matching
2. **Atomic**: All metadata in one file
3. **Fast**: Single file read (or Arrow mmap)
4. **Schema**: CSV/JSON enforces structure
5. **Extensible**: Easy to add columns (speaker_id, duration, etc.)

**The shift happened because datasets grew from dozens to millions of examples.**

---

## Practical Tips

### **1. Always Use `audiofolder` for Local Datasets**

Unless you have specific needs, `audiofolder` + `metadata.csv` is the easiest.

### **2. Use Arrow for Large Datasets (>10k examples)**

```python
dataset.save_to_disk("dataset.arrow")  # Fast, memory-efficient
```

### **3. Validate Your Dataset**

```python
from datasets import load_dataset

dataset = load_dataset("audiofolder", data_dir="my_dataset")

# Check schema
print(dataset.features)

# Verify first example loads
print(dataset[0])

# Check for missing audio files
for i, example in enumerate(dataset):
    try:
        _ = example["audio"]["array"]
    except Exception as e:
        print(f"Error at index {i}: {e}")
```

### **4. Add a `README.md` (Dataset Card)**

Even for local datasets, document:
- Audio format (WAV, MP3, sample rate, bit depth)
- Transcription conventions (capitalization, punctuation)
- Metadata columns explanation
- Licensing (if applicable)

---

## Summary

| Question | Answer |
|----------|--------|
| **Format name** | Hugging Face `datasets` Audio format (often via `audiofolder` loader) |
| **Why metadata in JSON/CSV?** | Single source of truth, atomic operations, fast loading, extensibility |
| **Official docs** | [https://huggingface.co/docs/datasets/audio_dataset](https://huggingface.co/docs/datasets/audio_dataset) |
| **Minimum schema** | `audio` (Audio feature) + `text` (string) |
| **Best for local** | `audiofolder` + `metadata.csv` |
| **Best for large** | Arrow format (`.save_to_disk()`) |

**The "standard" is the Hugging Face `datasets` library's Audio feature type**, which has become the de facto format for ASR datasets across the ecosystem. It's not a formal spec like JSON Schema, but a well-documented convention.

**For reference implementations, browse popular ASR datasets on Hugging Face Hub and examine their structure.**

---

**Note**: This guide was generated by Claude Code (claude-sonnet-4-5) for Daniel Rosehill's STT Fine-Tuning Notebook. The Hugging Face Datasets format continues to evolve—always check the official documentation for the latest features. For production datasets, consider using Arrow format for performance and validate your dataset structure before fine-tuning to catch errors early.
