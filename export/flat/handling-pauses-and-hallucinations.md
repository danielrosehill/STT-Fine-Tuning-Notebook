# Handling Pauses in Dictation: VAD, Hallucinations, and Solutions

## The Problem: When Silence Causes Hallucinations

If you've used Whisper-based transcription tools while dictating notes or blog outlines, you've likely encountered an annoying phenomenon: when you pause to think (10-20 seconds), the model sometimes "hallucinates" and inserts phantom text that you never spoke.

**Common hallucinations during silence:**
- Repeated phrases ("Thank you for watching. Thank you for watching.")
- Background music descriptions ("♪ music playing ♪")
- Generic filler text ("Please subscribe to my channel")
- Foreign language phrases
- Made-up words or nonsense

This document explains why this happens and how Voice Activity Detection (VAD) provides a practical solution—without requiring always-on listening or wake word detection.

## Why Whisper Hallucinates During Long Pauses

### The Root Cause: Attention Mechanism Behavior

Whisper (and similar ASR models) uses a transformer architecture with an attention mechanism. When given long segments of silence:

1. **The model expects speech:** Whisper is trained on audio with speech, not extended silence
2. **Attention seeks patterns:** The attention mechanism looks for *something* to focus on
3. **Noise becomes signal:** Background noise, breathing, ambient sounds get over-interpreted
4. **Decoder generates "plausible" text:** To fulfill its objective, the model generates text that "could" be there

### Why Long Pauses Are Worse

**Short pauses (1-3 seconds):** Generally handled well—model recognizes natural speech gaps

**Medium pauses (5-10 seconds):** Risk zone—model starts searching for signal in noise

**Long pauses (15-30+ seconds):** High hallucination risk—model "invents" content

**The trigger:** It's not the pause itself, but the length of silence fed to the model. Whisper processes audio in ~30-second chunks, so a 20-second pause in a 30-second window means 66% silence—enough to confuse the model.

### Common Hallucination Patterns

**1. Training Data Artifacts**
```
"Thank you for watching"
"Please subscribe"
"Don't forget to like and comment"
```
*Why:* Whisper was trained on YouTube videos—these phrases are common in that dataset.

**2. Music/Audio Descriptions**
```
"♪ instrumental music ♪"
"[music playing]"
"(upbeat music)"
```
*Why:* Training data included audio with music; model tries to describe what it "hears" in noise.

**3. Repeated Phrases**
```
"The project timeline. The project timeline. The project timeline."
```
*Why:* Attention mechanism gets stuck in a loop when there's no new information.

**4. Foreign Language Snippets**
```
"Gracias" (Spanish)
"Merci" (French)
```
*Why:* Multi-lingual training—model sometimes switches languages to "explain" ambiguous audio.

## Enter VAD: Voice Activity Detection

### What VAD Actually Does

**Core Function:** VAD detects when speech is present in audio and when it's absent.

**Key Clarification:** VAD is NOT the same as:
- **Always-on listening** (VAD can be used in push-to-record apps)
- **Wake word detection** (VAD doesn't trigger on keywords)

### How VAD Solves the Pause Problem

**Without VAD (Your Current Experience):**
```
You hit "Record"
    ↓
Audio buffer captures everything (speech + pauses + noise)
    ↓
You hit "Stop"
    ↓
Entire audio (including 20-second pauses) sent to Whisper
    ↓
Whisper tries to transcribe silence → hallucinations
```

**With VAD (Improved Workflow):**
```
You hit "Record"
    ↓
Audio buffer captures everything
    ↓
VAD analyzes audio in real-time or post-recording
    ↓
VAD marks segments: [speech] [silence] [speech] [silence] [speech]
    ↓
Only [speech] segments sent to Whisper
    ↓
Silence is completely removed from what Whisper sees
    ↓
No silence = no hallucinations
```

### VAD in Push-to-Record Applications

You don't need always-on listening to benefit from VAD. Here's how it works in a typical dictation app:

**Use Case 1: Post-Recording VAD Filtering**
```python
# User records audio (with pauses)
audio = record_audio()  # Contains speech + 20-second pauses

# Apply VAD after recording
vad = load_vad_model()
speech_segments = vad.get_speech_timestamps(audio)

# Extract only speech
speech_only_audio = extract_segments(audio, speech_segments)

# Transcribe speech-only audio
transcript = whisper_model.transcribe(speech_only_audio)

# Result: No hallucinations from pauses
```

**Use Case 2: Real-time VAD During Recording (Streaming)**
```python
# User hits "Record"
audio_buffer = []

for audio_chunk in audio_stream:
    # VAD checks each chunk
    if vad.is_speech(audio_chunk):
        audio_buffer.append(audio_chunk)
    else:
        # Silence detected - ignore this chunk
        pass

# User hits "Stop"
# audio_buffer contains only speech
transcript = whisper_model.transcribe(audio_buffer)
```

**Key Point:** In both cases, you still manually control when recording starts and stops. VAD simply filters out the silent parts *within* your recording session.

## Practical Implementation

### Solution 1: Silero VAD (Recommended)

**Why Silero VAD?**
- Lightweight (1.5 MB model)
- Fast (< 5ms per audio chunk)
- Highly accurate (< 1% false positive rate)
- Easy to integrate

**Installation:**
```bash
pip install torch torchaudio
```

**Implementation:**
```python
import torch
import torchaudio

# Load Silero VAD
model, utils = torch.hub.load(
    repo_or_dir='snakers4/silero-vad',
    model='silero_vad',
    force_reload=False
)

(get_speech_timestamps, _, read_audio, _, _) = utils

# Load your recorded audio
audio = read_audio('your_recording.wav', sampling_rate=16000)

# Get speech timestamps
speech_timestamps = get_speech_timestamps(
    audio,
    model,
    threshold=0.5,        # Confidence threshold (0.3-0.7 typical)
    sampling_rate=16000,
    min_speech_duration_ms=250,  # Ignore very short speech segments
    min_silence_duration_ms=500  # Minimum silence to trigger segmentation
)

# Extract speech-only audio
speech_segments = []
for timestamp in speech_timestamps:
    start = timestamp['start']
    end = timestamp['end']
    speech_segments.append(audio[start:end])

# Concatenate all speech segments
speech_only = torch.cat(speech_segments)

# Save for transcription
torchaudio.save('speech_only.wav', speech_only.unsqueeze(0), 16000)

# Now transcribe with Whisper
import whisper
model = whisper.load_model("base")
result = model.transcribe("speech_only.wav")
print(result["text"])
```

**Result:** Your 20-second pauses are completely removed; Whisper only sees actual speech.

### Solution 2: Whisper with VAD Pre-filtering (whisper-ctranslate2)

Some Whisper implementations have VAD built-in:

**Installation:**
```bash
pip install whisper-ctranslate2
```

**Usage:**
```python
from whisper_ctranslate2 import Transcribe

# Initialize with VAD enabled
transcriber = Transcribe(
    model_path="base",
    device="cpu",
    compute_type="int8",
    vad_filter=True,  # Enable VAD filtering
    vad_parameters={
        "threshold": 0.5,
        "min_speech_duration_ms": 250,
        "min_silence_duration_ms": 2000  # 2 seconds of silence = segment boundary
    }
)

# Transcribe with automatic VAD filtering
result = transcriber.transcribe("your_recording.wav")
print(result["text"])
```

**Advantage:** Single-step process—VAD and transcription combined.

### Solution 3: Faster-Whisper with VAD

**Installation:**
```bash
pip install faster-whisper
pip install silero-vad
```

**Implementation:**
```python
from faster_whisper import WhisperModel
import torch

# Load VAD
vad_model, utils = torch.hub.load(
    repo_or_dir='snakers4/silero-vad',
    model='silero_vad'
)

get_speech_timestamps = utils[0]
read_audio = utils[2]

# Load audio
audio = read_audio('your_recording.wav', sampling_rate=16000)

# Get speech timestamps
speech_timestamps = get_speech_timestamps(
    audio,
    vad_model,
    threshold=0.5
)

# Load Faster-Whisper
whisper_model = WhisperModel("base", device="cpu", compute_type="int8")

# Transcribe only speech segments
full_transcript = []
for timestamp in speech_timestamps:
    start_sample = timestamp['start']
    end_sample = timestamp['end']

    # Convert samples to time (for faster-whisper)
    start_time = start_sample / 16000
    end_time = end_sample / 16000

    # Transcribe segment (using seek parameter)
    segments, info = whisper_model.transcribe(
        'your_recording.wav',
        word_timestamps=False,
        vad_filter=False  # We already applied VAD
    )

    for segment in segments:
        if start_time <= segment.start <= end_time:
            full_transcript.append(segment.text)

print(" ".join(full_transcript))
```

## Configuration: Tuning VAD for Dictation

### Key Parameters

**1. Threshold (0.0 - 1.0)**
- **Lower (0.3-0.4):** More sensitive—catches quiet speech, but may include noise
- **Higher (0.6-0.7):** Less sensitive—only clear speech, but may miss soft speech
- **Recommended for dictation:** 0.5 (balanced)

**2. Min Speech Duration (ms)**
- **Purpose:** Ignore very short bursts (likely noise)
- **Too low (< 100ms):** Noise/clicks detected as speech
- **Too high (> 500ms):** Short words/syllables missed
- **Recommended for dictation:** 250ms

**3. Min Silence Duration (ms)**
- **Purpose:** Define when a pause is "silence" vs. natural speech gap
- **Lower (100-300ms):** Aggressive segmentation—splits on brief pauses
- **Higher (1000-2000ms):** Allows longer pauses within same segment
- **Recommended for dictation:** 500-1000ms

**For your use case (thinking pauses):**
```python
speech_timestamps = get_speech_timestamps(
    audio,
    vad_model,
    threshold=0.5,
    min_speech_duration_ms=250,
    min_silence_duration_ms=1000  # 1 second allows natural pauses
    # But 10-20 second thinking pauses will be filtered out
)
```

### Testing Your Configuration

**Validation Script:**
```python
import torch
import torchaudio
from pprint import pprint

# Load VAD
model, utils = torch.hub.load('snakers4/silero-vad', model='silero_vad')
get_speech_timestamps = utils[0]
read_audio = utils[2]

# Load audio
audio = read_audio('test_recording.wav', sampling_rate=16000)

# Test different configurations
configs = [
    {"threshold": 0.4, "min_silence_duration_ms": 500},
    {"threshold": 0.5, "min_silence_duration_ms": 1000},
    {"threshold": 0.6, "min_silence_duration_ms": 1500},
]

for config in configs:
    print(f"\nTesting: {config}")
    speech_timestamps = get_speech_timestamps(
        audio,
        model,
        min_speech_duration_ms=250,
        **config
    )

    # Analyze results
    total_speech_time = sum(
        (ts['end'] - ts['start']) / 16000 for ts in speech_timestamps
    )
    num_segments = len(speech_timestamps)

    print(f"  Segments detected: {num_segments}")
    print(f"  Total speech time: {total_speech_time:.2f}s")
    print(f"  First 3 segments:")
    pprint(speech_timestamps[:3])
```

**Run this on a test recording with known pauses to find your ideal settings.**

## Applications Beyond Always-On Listening

You mentioned associating VAD with always-on listening—here's the full range of VAD use cases to clarify:

### 1. Push-to-Record Dictation (Your Use Case)
- **You control:** When recording starts/stops
- **VAD controls:** Which parts of your recording get transcribed
- **Benefit:** Hallucination-free transcripts despite thinking pauses

### 2. Always-On Listening (Virtual Assistants)
- **VAD controls:** When recording starts (speech detected)
- **VAD controls:** When recording stops (silence detected)
- **You don't manually trigger anything**

### 3. Meeting/Podcast Transcription
- **You control:** Load audio file
- **VAD controls:** Segments sent to ASR (ignores silence between speakers)
- **Benefit:** Faster transcription, lower costs

### 4. Real-time Streaming (Live Captions)
- **Audio continuously captured**
- **VAD controls:** When to send chunks to ASR
- **Benefit:** Lower latency, reduced compute

**Key Distinction:** VAD is a *tool* that can be used in any of these scenarios. It's not inherently tied to always-on listening.

## Alternative Approaches (Without VAD)

If you can't or don't want to use VAD, here are workarounds:

### 1. Prompt Engineering (Limited Effectiveness)

**Whisper's `initial_prompt` parameter:**
```python
result = model.transcribe(
    "recording.wav",
    initial_prompt="This is a dictation with natural pauses. Do not add filler text."
)
```

**Reality:** This helps slightly but doesn't eliminate hallucinations during long silence.

### 2. Temperature Reduction

**Lower temperature = less creative (fewer hallucinations):**
```python
result = model.transcribe(
    "recording.wav",
    temperature=0.0  # Default is 0.0-1.0
)
```

**Limitation:** Also makes the model less flexible with accents/vocabulary.

### 3. Shorter Recording Sessions

**Workaround:** Don't let pauses sit in the recording buffer.
- Manually pause/resume recording during thinking breaks
- Record in shorter bursts (30-60 seconds)
- Stitch transcripts together post-processing

**Downside:** Interrupts your workflow; requires manual management.

### 4. Post-Processing Cleanup

**Filter hallucinations with keyword detection:**
```python
hallucination_phrases = [
    "thank you for watching",
    "please subscribe",
    "♪",
    "[music",
]

transcript = result["text"]
for phrase in hallucination_phrases:
    transcript = transcript.replace(phrase, "")

print(transcript)
```

**Limitation:** Only catches known hallucinations; won't catch all.

## Recommended Setup for Dictation

**For your specific workflow (blog outlines with thinking pauses):**

### Option A: Silero VAD + Whisper (Most Control)

**Pros:**
- Complete control over VAD parameters
- Works with any Whisper backend (faster-whisper, whisper.cpp, etc.)
- Transparent—you can inspect speech segments before transcription

**Cons:**
- Requires two-step process (VAD → transcribe)
- Slightly more code

### Option B: Whisper-CTranslate2 with Built-in VAD (Easiest)

**Pros:**
- Single command
- VAD automatically applied
- Good defaults for dictation

**Cons:**
- Less control over VAD parameters
- CTranslate2 dependency

### Option C: Faster-Whisper + External VAD (Best Performance)

**Pros:**
- Fastest inference (2-4x faster than OpenAI Whisper)
- High-quality VAD with Silero
- Good for large volumes of dictation

**Cons:**
- More complex setup
- GPU recommended for best speed

**Recommendation:**
Start with **Option B** (whisper-ctranslate2) for simplicity. If you need more control, switch to **Option A** (Silero + Whisper).

## Real-World Example: Before and After VAD

### Before VAD (With Hallucinations)

**Your dictation:**
> "I want to outline a blog post about AI transcription tools. (20-second pause thinking) The first section should cover accuracy metrics."

**Whisper's transcript (with hallucinations):**
> "I want to outline a blog post about AI transcription tools. Thank you for watching. Thank you for watching. Please subscribe. The first section should cover accuracy metrics."

### After VAD (Clean)

**VAD detects:**
- Speech: 0-5s ("I want to outline...")
- Silence: 5-25s (pause)
- Speech: 25-30s ("The first section...")

**VAD sends to Whisper:**
- Segment 1: "I want to outline..."
- Segment 2: "The first section..."

**Whisper's transcript (no hallucinations):**
> "I want to outline a blog post about AI transcription tools. The first section should cover accuracy metrics."

## Performance Impact

**Overhead of VAD:**
- Silero VAD: ~1-5ms per 100ms audio chunk
- For 60 seconds of audio: ~100ms total VAD processing
- **Negligible impact** compared to ASR (which takes seconds)

**Benefit:**
- Reduced ASR processing time (only transcribing speech)
- No manual cleanup of hallucinations
- Improved accuracy

**Net result:** Faster overall workflow despite extra VAD step.

## Conclusion

**The short answer to your question:** Yes, VAD absolutely solves your pause problem, and no, it doesn't require always-on listening.

**What VAD does:**
- Detects when you're speaking vs. pausing
- Filters out silent segments before they reach Whisper
- Prevents hallucinations caused by long thinking pauses

**How to use it:**
1. Record your dictation as usual (pauses and all)
2. Apply VAD post-recording to extract speech-only segments
3. Transcribe speech-only audio with Whisper
4. Get clean transcripts without phantom text

**Recommended starting point:**
```bash
pip install whisper-ctranslate2
```

```python
from whisper_ctranslate2 import Transcribe

transcriber = Transcribe(
    model_path="base",
    vad_filter=True,
    vad_parameters={"min_silence_duration_ms": 1000}
)

result = transcriber.transcribe("your_recording.wav")
print(result["text"])
```

**Result:** No more "Thank you for watching" hallucinations during your coffee-free morning thought pauses.

---

*This document was generated by Claude Code as part of Daniel Rosehill's STT Fine-Tuning Notebook. VAD technology continues to improve; consult current documentation for the latest models and parameters.*
