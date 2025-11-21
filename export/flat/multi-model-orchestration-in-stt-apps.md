# Multi-Model Orchestration in Speech-to-Text Applications

## Overview

Modern speech-to-text (STT) applications are far more complex than they initially appear. What seems like a simple "record and transcribe" app actually orchestrates multiple AI models working in harmony. This document explains how these models interact, the sequence of operations, and the architectural patterns that make it all work seamlessly.

## The Multi-Model Architecture

### Core Components

A typical modern STT application combines 4-6 different models:

1. **Voice Activity Detection (VAD)** - Detects when speech is present
2. **Wake Word Detection (WWD)** - (Optional) Triggers on specific phrases
3. **Automatic Speech Recognition (ASR)** - Core transcription model
4. **Punctuation Restoration** - Adds punctuation to raw transcripts
5. **Diarization** - (Optional) Identifies different speakers
6. **Language Identification** - (Optional) Detects spoken language

### Size and Resource Distribution

**Typical Model Sizes:**
- **VAD:** 1-5 MB (e.g., Silero VAD: 1.5 MB)
- **Wake Word:** 1-10 MB (e.g., Porcupine: 1-3 MB per keyword)
- **ASR Model:** 70 MB - 3 GB (e.g., Whisper tiny: 75 MB, large-v3: 3 GB)
- **Punctuation:** 50-500 MB (e.g., FullStop: 300 MB)
- **Diarization:** 100-500 MB (e.g., pyannote diarization: 300 MB)

The ASR model dominates resource usage (compute, memory, latency), while supporting models are lightweight and fast.

## The Processing Pipeline: From Recording to Text

### Phase 1: Pre-Processing (During Recording)

#### 1.1 Audio Capture
```
User hits "Record"
    ↓
Audio Device Initialization
    ↓
Audio Buffer Stream (typically 16kHz or 44.1kHz)
```

**What happens:**
- Audio driver opens input device
- Circular buffer created (typically 1-10 seconds)
- Audio chunks streamed at fixed intervals (e.g., 100ms frames)

#### 1.2 Voice Activity Detection (VAD) - Real-time

**Purpose:** Filter out silence and non-speech audio

**How it works:**
```
Audio Chunk (100ms)
    ↓
VAD Model (lightweight CNN/RNN)
    ↓
Speech Probability (0.0 - 1.0)
    ↓
Threshold Check (e.g., > 0.5 = speech)
    ↓
Decision: Keep or Discard
```

**Benefits:**
- Reduces data sent to ASR (saves compute)
- Eliminates silent segments
- Lowers transcription latency
- Reduces API costs (for cloud services)

**Real-world Example:**
```python
# Silero VAD (popular lightweight VAD)
import torch

vad_model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad',
                                    model='silero_vad')

get_speech_timestamps = utils[0]

# Process audio chunks in real-time
speech_timestamps = get_speech_timestamps(
    audio_chunk,
    vad_model,
    threshold=0.5,
    sampling_rate=16000
)
```

**Timing:** 1-5ms per 100ms audio chunk (real-time capable)

#### 1.3 Wake Word Detection (If Enabled)

**Purpose:** Trigger recording only on specific phrases ("Hey Siri", "Alexa", etc.)

**How it works:**
```
Continuous Audio Stream
    ↓
WWD Model (small neural network)
    ↓
Keyword Match Score
    ↓
Threshold Check (e.g., > 0.8 = keyword detected)
    ↓
Trigger: Start ASR Pipeline
```

**Architecture:**
- Always-on listening mode
- Ultra-low power consumption critical
- Edge deployment (on-device, not cloud)
- False positive rate < 1 per hour

**Popular Solutions:**
- Porcupine (Picovoice)
- Snowboy (deprecated but still used)
- Custom models (openWakeWord)

**Timing:** 1-3ms per audio frame (must be faster than real-time)

### Phase 2: Primary Transcription

#### 2.1 Audio Buffering

**Buffering Strategy:**

**A. Streaming Mode (Real-time)**
```
VAD Active
    ↓
Buffer audio in chunks (e.g., 5-30 second segments)
    ↓
Send to ASR when:
    - Buffer reaches max duration
    - VAD detects end of speech (silence > threshold)
    - User manually stops
```

**B. Batch Mode (Post-recording)**
```
User hits "Stop Recording"
    ↓
All audio collected
    ↓
Single file/buffer ready for processing
```

#### 2.2 ASR Model Inference

**How it works:**
```
Audio Segment (5-30 seconds)
    ↓
Preprocessing:
    - Resample to model's expected rate (often 16kHz)
    - Convert to mel spectrogram
    - Normalize audio levels
    ↓
ASR Model (e.g., Whisper, Wav2Vec2)
    ↓
Raw Transcription (no punctuation, lowercase)
    ↓
Confidence Scores (optional)
```

**Key Considerations:**

**Chunking for Long Audio:**
For audio > 30 seconds, apps typically use one of two strategies:

**Strategy A: Sequential Chunking**
```python
# Pseudo-code
chunks = split_audio(audio, chunk_duration=30)
transcripts = []

for chunk in chunks:
    transcript = asr_model.transcribe(chunk)
    transcripts.append(transcript)

full_transcript = merge_with_overlap_handling(transcripts)
```

**Strategy B: Sliding Window with Overlap**
```python
# Better approach for continuity
chunks = split_audio_with_overlap(audio, chunk=30, overlap=5)
transcripts = []

for chunk in chunks:
    transcript = asr_model.transcribe(chunk)
    transcripts.append(transcript)

# Merge using overlap to resolve chunk boundaries
full_transcript = merge_overlapping_chunks(transcripts)
```

**Timing:**
- Depends on model size and hardware
- **Real-time factor (RTF):**
  - RTF = 0.5 means 10 seconds of audio transcribed in 5 seconds
  - Whisper large-v3 on RTX 4090: RTF ≈ 0.1 (very fast)
  - Whisper large-v3 on CPU: RTF ≈ 1.5-3.0 (slower than real-time)

#### 2.3 Parallel Processing (Optional)

Some apps process VAD and ASR in parallel:

```
Audio Stream
    ├─→ VAD (continuous, filters silence)
    └─→ ASR (processes VAD-approved segments)
```

**Why parallel?**
- VAD filters unnecessary audio before ASR
- ASR only sees speech, improving accuracy and speed
- Reduces compute costs

### Phase 3: Post-Processing

#### 3.1 Punctuation Restoration

**Purpose:** Add punctuation and capitalization to raw ASR output

**Input:**
```
"hey how are you doing today i wanted to ask you about the project timeline"
```

**Output:**
```
"Hey, how are you doing today? I wanted to ask you about the project timeline."
```

**How it works:**
```
Raw ASR Transcript
    ↓
Punctuation Model (BERT-based, T5, or custom RNN)
    ↓
    - Detects sentence boundaries
    - Inserts periods, commas, question marks
    - Capitalizes proper nouns and sentence starts
    ↓
Punctuated Transcript
```

**Popular Models:**
- FullStop (Hugging Face)
- DeepPunctuation
- recasepunc (Nvidia NeMo)

**Architecture:**
- Usually transformer-based (BERT, RoBERTa)
- Input: raw text + optional audio features
- Output: text with punctuation tokens

**Example Implementation:**
```python
from transformers import pipeline

# Load punctuation restoration model
punctuator = pipeline(
    "token-classification",
    model="oliverguhr/fullstop-punctuation-multilang-large"
)

raw_text = "hey how are you doing today"
punctuated = punctuator(raw_text)

# Result: "Hey, how are you doing today?"
```

**Timing:** 50-500ms for typical paragraphs

#### 3.2 Speaker Diarization (Optional)

**Purpose:** Identify "who spoke when"

**Output Format:**
```
[00:00 - 00:15] Speaker 1: "Hey, how are you doing today?"
[00:15 - 00:30] Speaker 2: "I'm doing great, thanks for asking!"
[00:30 - 00:45] Speaker 1: "That's wonderful to hear."
```

**How it works:**
```
Audio File + Transcript
    ↓
Extract Speaker Embeddings (every few seconds)
    ↓
Clustering Algorithm (group similar embeddings)
    ↓
Assign Speaker Labels to Transcript Segments
```

**Popular Solutions:**
- pyannote.audio (state-of-the-art)
- NVIDIA NeMo
- Kaldi-based systems

**Timing:** 0.5-2x real-time (depends on audio duration)

#### 3.3 Language Identification (Optional)

**Purpose:** Detect spoken language before transcription

**Use Cases:**
- Multi-lingual apps
- Automatic model selection
- Translation triggers

**How it works:**
```
Initial Audio Segment (1-5 seconds)
    ↓
Language ID Model (CNN or Whisper's built-in LID)
    ↓
Language Code (e.g., "en", "es", "fr")
    ↓
Select appropriate ASR model or configure decoder
```

**Whisper's Approach:**
- Built-in language detection
- First 30 seconds used for detection
- 97 languages supported

## Orchestration Patterns: How It All Works Together

### Pattern 1: Sequential Pipeline (Most Common)

**Architecture:**
```
User Hits Record
    ↓
[VAD continuously filters audio]
    ↓
User Hits Stop
    ↓
[ASR processes VAD-approved audio]
    ↓
[Punctuation restoration on transcript]
    ↓
[Optional: Diarization]
    ↓
Display final transcript
```

**Advantages:**
- Simple to implement
- Easy to debug
- Clear error boundaries

**Disadvantages:**
- Higher latency (sequential processing)
- No partial results during recording

### Pattern 2: Streaming Pipeline with Partial Results

**Architecture:**
```
User Hits Record
    ↓
Continuous Processing Loop:
    ├─→ [VAD filters audio chunk]
    ├─→ [ASR transcribes chunk (streaming mode)]
    ├─→ [Display partial transcript]
    └─→ [Next chunk]
    ↓
User Hits Stop
    ↓
[Final punctuation restoration on full transcript]
    ↓
Display final polished transcript
```

**Advantages:**
- Low latency
- User sees progress
- Better UX for long recordings

**Disadvantages:**
- More complex implementation
- Requires streaming-capable ASR model
- Potential for interim transcript changes

**Example: Whisper Streaming**
```python
from whisper_streaming import WhisperStreamingTranscriber

transcriber = WhisperStreamingTranscriber()

# Stream audio chunks
for audio_chunk in audio_stream:
    partial_transcript = transcriber.process_chunk(audio_chunk)
    display_to_user(partial_transcript)  # Update UI in real-time

final_transcript = transcriber.finalize()
```

### Pattern 3: Parallel Processing with Async Queue

**Architecture:**
```
                    User Hits Record
                            ↓
                 [Audio Input Thread]
                            ↓
                    [Queue: audio_queue]
                    /                  \
                   /                    \
    [Thread 1: VAD]              [Thread 2: ASR]
           ↓                              ↓
    Filters audio                Transcribes segments
    Feeds to ASR queue           Sends to punctuation queue
                    \                   /
                     \                 /
                    [Thread 3: Punctuation]
                            ↓
                    [Output Queue]
                            ↓
                    Display to User
```

**Advantages:**
- Maximum performance (utilizes multiple cores)
- Lower latency
- Efficient resource usage

**Disadvantages:**
- Complex to implement
- Requires thread-safe queue management
- Harder to debug

**Implementation Example:**
```python
import queue
import threading

# Queues for each stage
audio_queue = queue.Queue()
vad_queue = queue.Queue()
asr_queue = queue.Queue()
punctuation_queue = queue.Queue()

def audio_capture_thread():
    """Capture audio and feed to VAD"""
    while recording:
        chunk = capture_audio()
        audio_queue.put(chunk)

def vad_thread():
    """Filter silence from audio"""
    while True:
        chunk = audio_queue.get()
        if vad_model.is_speech(chunk):
            vad_queue.put(chunk)

def asr_thread():
    """Transcribe speech segments"""
    buffer = []
    while True:
        chunk = vad_queue.get()
        buffer.append(chunk)

        if len(buffer) >= TARGET_LENGTH:
            transcript = asr_model.transcribe(buffer)
            asr_queue.put(transcript)
            buffer = []

def punctuation_thread():
    """Add punctuation to raw transcripts"""
    while True:
        raw_text = asr_queue.get()
        punctuated = punctuation_model.restore(raw_text)
        punctuation_queue.put(punctuated)

# Start all threads
threads = [
    threading.Thread(target=audio_capture_thread),
    threading.Thread(target=vad_thread),
    threading.Thread(target=asr_thread),
    threading.Thread(target=punctuation_thread)
]

for t in threads:
    t.start()
```

## Preventing Model Collisions

### Problem: Model Interference

**Issue:**
Multiple models competing for:
- GPU memory
- CPU cores
- Disk I/O
- Memory bandwidth

**Solutions:**

### 1. Resource Isolation

**GPU Memory Management:**
```python
# Explicitly allocate GPU memory per model
import torch

# Load VAD on GPU with limited memory
vad_model = load_vad()
torch.cuda.set_per_process_memory_fraction(0.1)  # 10% GPU memory

# Load ASR on GPU with remaining memory
asr_model = load_whisper()
torch.cuda.set_per_process_memory_fraction(0.8)  # 80% GPU memory
```

**CPU Core Affinity:**
```python
import os

# Pin VAD to specific CPU cores
os.sched_setaffinity(0, {0, 1})  # Cores 0-1 for VAD

# ASR can use remaining cores
os.sched_setaffinity(0, {2, 3, 4, 5})  # Cores 2-5 for ASR
```

### 2. Sequential Execution with Clear Dependencies

**Dependency Graph:**
```
VAD (required before ASR)
    ↓
ASR (required before punctuation)
    ↓
Punctuation (final step)
```

**Implementation:**
```python
def process_audio(audio):
    # Step 1: VAD (filters audio)
    speech_segments = vad_model.detect_speech(audio)

    # Step 2: ASR (only on speech segments)
    raw_transcripts = []
    for segment in speech_segments:
        transcript = asr_model.transcribe(segment)
        raw_transcripts.append(transcript)

    # Step 3: Punctuation
    full_transcript = " ".join(raw_transcripts)
    final_transcript = punctuation_model.restore(full_transcript)

    return final_transcript
```

### 3. Model Warm-up and Caching

**Problem:** First inference slow due to model initialization

**Solution:**
```python
class STTOrchestrator:
    def __init__(self):
        # Pre-load all models during app startup
        print("Loading models...")
        self.vad = load_vad_model()
        self.asr = load_asr_model()
        self.punctuation = load_punctuation_model()

        # Warm-up inference (compile kernels, allocate buffers)
        dummy_audio = generate_dummy_audio()
        _ = self.vad(dummy_audio)
        _ = self.asr(dummy_audio)
        _ = self.punctuation("test text")
        print("Models ready!")

    def transcribe(self, audio):
        # Now inference is fast
        speech = self.vad(audio)
        transcript = self.asr(speech)
        final = self.punctuation(transcript)
        return final
```

## Real-World Examples

### Example 1: Otter.ai (Commercial App)

**Architecture:**
```
[Real-time Audio Stream]
        ↓
[Client-side VAD] (lightweight)
        ↓
[Send to cloud only when speech detected]
        ↓
[Cloud ASR] (Whisper or similar)
        ↓
[Punctuation + Diarization] (parallel)
        ↓
[Return to client with formatting]
```

**Key Features:**
- Hybrid client/cloud architecture
- VAD on-device (saves bandwidth and costs)
- Heavy ASR in cloud (better accuracy, GPU acceleration)
- Streaming results (partial transcripts)

### Example 2: Whisper Desktop Apps (e.g., MacWhisper)

**Architecture:**
```
[Record audio to file]
        ↓
[User hits "Transcribe"]
        ↓
[Load audio file]
        ↓
[VAD preprocessing] (optional, reduces compute)
        ↓
[Whisper ASR] (on-device, uses GPU if available)
        ↓
[Display transcript]
        ↓
[User can manually edit]
```

**Key Features:**
- Fully on-device (privacy)
- Batch processing (not real-time)
- Utilizes Metal (macOS) or CUDA/ROCm for GPU acceleration

### Example 3: Real-time Meeting Transcription (e.g., Google Meet captions)

**Architecture:**
```
[Audio from meeting]
        ↓
[Acoustic Echo Cancellation] (filter out speakers)
        ↓
[VAD] (per participant if multi-source)
        ↓
[Streaming ASR] (processes ~3 second chunks)
        ↓
[Display partial results immediately]
        ↓
[Punctuation applied in real-time]
        ↓
[Speaker diarization] (if enabled)
        ↓
[Final transcript saved]
```

**Key Features:**
- Ultra-low latency (< 2 seconds)
- Streaming architecture
- Multi-speaker handling
- Noise suppression

## Timing and Latency Breakdown

**Typical Latency for a 30-second Recording:**

```
Component                    Time        Cumulative
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Audio Capture               30.0s            30.0s
VAD Processing               0.5s            30.5s
ASR Inference (GPU)          3.0s            33.5s
Punctuation Restoration      0.3s            33.8s
Diarization (optional)       15.0s           48.8s
Display to User              0.1s            48.9s
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Total: ~49 seconds (1.6x real-time)
```

**For Streaming (Real-time) Mode:**

```
Component                           Latency     Update Frequency
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Audio Buffer                         1-3s       Continuous
VAD Processing                      10-50ms     Per chunk (100ms)
ASR Streaming Inference             500-1000ms  Every 3-5 seconds
Punctuation (partial)               100ms       Every new segment
Display Update                      10-30ms     Per transcript update
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Perceived Latency: 1-3 seconds behind real-time
```

## Error Handling and Fault Tolerance

### Common Failure Modes

1. **VAD False Negatives:** Speech detected as silence
   - **Solution:** Adjust VAD threshold, use multiple VAD models

2. **ASR Inference Timeout:** Model takes too long
   - **Solution:** Fallback to smaller model, chunk audio more aggressively

3. **GPU Out of Memory:** Models too large for VRAM
   - **Solution:** Sequential model unloading, model quantization

4. **Audio Buffer Overflow:** Recording too long
   - **Solution:** Automatic chunking, progressive processing

### Graceful Degradation

**Priority Hierarchy:**
```
Critical:     ASR transcription
High:         VAD (improves speed, not accuracy)
Medium:       Punctuation (improves readability)
Low:          Diarization (nice to have)
```

**Fallback Strategy:**
```python
def robust_transcribe(audio):
    try:
        # Try full pipeline
        speech = vad(audio)
        transcript = asr(speech)
        punctuated = punctuation(transcript)
        diarized = diarization(audio, punctuated)
        return diarized
    except OutOfMemoryError:
        # Disable diarization
        speech = vad(audio)
        transcript = asr(speech)
        punctuated = punctuation(transcript)
        return punctuated
    except Exception as e:
        # Minimal pipeline: ASR only
        transcript = asr(audio)
        return transcript
```

## Optimization Strategies

### 1. Model Quantization
- Convert FP32 models to INT8 or FP16
- 2-4x speedup with minimal accuracy loss
- Essential for edge deployment

### 2. Model Pruning
- Remove unnecessary weights from models
- Reduces model size and inference time
- Particularly effective for VAD and punctuation models

### 3. Batch Processing
- Process multiple audio segments simultaneously
- Better GPU utilization
- Only applicable for post-recording processing

### 4. Caching and Memoization
- Cache VAD results for repeated audio
- Store ASR outputs for common phrases
- Useful for limited domain applications

## Future Trends

### 1. End-to-End Models
Unified models handling multiple tasks:
- Whisper already includes language detection
- Next-gen models may include punctuation, diarization
- Simpler architecture, but less flexible

### 2. On-Device Everything
- Smaller, more efficient models (e.g., Whisper tiny, Distil-Whisper)
- Privacy-focused (no cloud processing)
- Lower latency

### 3. Multimodal Integration
- Video + audio for better context
- Visual cues for speaker diarization
- Gesture recognition for control

## Conclusion

Modern STT applications are sophisticated orchestrations of multiple AI models, each serving a specific purpose:

1. **VAD** filters silence (reduces compute)
2. **Wake Word** triggers recording (optional)
3. **ASR** performs core transcription (the heavy lifter)
4. **Punctuation** improves readability
5. **Diarization** identifies speakers (optional)

The "magic" behind the scenes involves:
- **Careful sequencing** of model execution
- **Resource isolation** to prevent collisions
- **Queuing and threading** for parallel processing
- **Error handling** for graceful degradation
- **Optimization techniques** for real-time performance

Apps use various orchestration patterns—sequential, streaming, or parallel—depending on latency requirements, hardware constraints, and user experience goals.

The result is a seamless experience where the user presses "Record," speaks, hits "Stop," and receives a fully punctuated, formatted transcript seconds later—all powered by a symphony of AI models working in perfect harmony.

---

*This document was generated by Claude Code as part of Daniel Rosehill's STT Fine-Tuning Notebook. For technical accuracy verification and the latest developments in multi-model STT architectures, consult current research and documentation from model providers.*
