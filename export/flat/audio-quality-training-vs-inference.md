# Audio Quality in Training Data: Clean Studio vs Real-World Conditions

## The Question

When recording training data for ASR fine-tuning, should you:

**Option A:** Record in optimal conditions (quiet room, quality microphone, clean audio)?

**Option B:** Record in real-world conditions (phone mic, background noise, realistic environments)?

Since you'll be using the model primarily in noisy, real-world conditions, wouldn't training on similar conditions produce better results?

## Short Answer

**You should primarily record clean, high-quality training data, then add controlled noise augmentation.**

This approach gives you:

1. Clean signal for the model to learn your voice and vocabulary
2. Controlled noise addition that teaches robustness
3. Flexibility to adapt to different noise conditions
4. Better training efficiency and convergence

Recording natively in noisy conditions sounds intuitive but actually produces worse results for fine-tuning.

## Why Clean Data + Augmentation Beats Noisy Recording

### The Core Principle: Learn Signal, Then Noise

ASR models learn two things:

1. **Signal:** Your voice characteristics, pronunciation, vocabulary
2. **Noise robustness:** How to extract signal from noise

**Optimal learning:** Teach these separately, then combine

**Suboptimal learning:** Try to learn both simultaneously from noisy data

### Problem 1: Noise Variability

When you record natively in real-world conditions:

```
Recording 1: Your voice + office air conditioning hum + keyboard typing
Recording 2: Your voice + street traffic + wind on mic
Recording 3: Your voice + café chatter + coffee machine
Recording 4: Your voice + home (different) + dog barking
```

**Issues:**

- Every recording has **different noise**
- Model must learn: "Ignore air conditioning AND traffic AND café noise AND..."
- Model has only ~10 hours of data to learn all these noise patterns
- Inefficient learning: splitting attention between voice and dozens of noise types

### Problem 2: Signal Masking

Noise obscures the very features you want the model to learn:

```
Clean recording:
  "Mekolet" pronunciation clearly captured
  Phonemes: [me-ko-let] with clear formants

Noisy recording (street):
  "M-k-l-t" (traffic masked vowels)
  Phonemes partially obscured by noise
```

**Result:** Model learns degraded representation of your voice, not the clean acoustic patterns

### Problem 3: Inconsistent Quality

Real-world recording produces inconsistent quality:

- Some samples loud, some quiet
- Some samples mostly clean, some very noisy
- Some samples have one noise type, others have different noise

**Training issue:** Model gets confused by inconsistency, learns poorly

### The Better Approach: Clean Data + Augmentation

```python
# Training pipeline
clean_audio = record_in_quiet_room_with_quality_mic()

# Add controlled augmentation
augmented_data = [
    clean_audio,                          # 40% clean
    clean_audio + cafe_noise,             # 15% café noise
    clean_audio + traffic_noise,          # 15% traffic noise
    clean_audio + office_noise,           # 15% office noise
    clean_audio + phone_mic_simulation,   # 15% phone simulation
]

# Train on augmented dataset
model.finetune(augmented_data)
```

**Advantages:**

1. **Clean signal learning:** Model learns your voice without interference
2. **Controlled noise diversity:** You choose which noise types to include
3. **Adjustable noise levels:** You control signal-to-noise ratio (SNR)
4. **Reproducible:** Same clean base can be augmented differently for experiments
5. **Efficient:** 1 clean recording → 5+ augmented versions

## The Science: Domain Adaptation vs Domain Mismatch

### Scenario A: Train clean, test noisy (with augmentation)

```
Training: Clean + augmented noise
Testing: Real-world noise
Result: ✓ Good performance
```

**Why it works:**

- Model learns clean acoustic patterns
- Augmentation teaches: "noise can appear in many forms"
- Model generalizes noise robustness from augmented examples
- Base acoustic model remains clean and accurate

### Scenario B: Train noisy, test noisy

```
Training: Native noisy recordings
Testing: Real-world noise
Result: ✗ Poor performance
```

**Why it fails:**

- Model learns degraded acoustic patterns
- Noise in training ≠ noise in testing (different types)
- Model overfits to specific training noise
- Base acoustic model is compromised

### Scenario C: Train clean, test noisy (no augmentation)

```
Training: Clean only
Testing: Real-world noise
Result: △ Moderate performance
```

**Why it's suboptimal:**

- Model learns clean patterns well
- No noise robustness training
- Some transfer to noise (Whisper pre-training helps)
- Performance degrades in very noisy conditions

### Scenario D: Train clean + augmented, test clean

```
Training: Clean + augmented noise
Testing: Clean conditions
Result: ✓ Best performance
```

**Why it's optimal:**

- Model learned from clean signal
- Augmentation doesn't hurt clean performance
- Model can perform well in both clean and noisy conditions

## Practical Guidelines

### Recording Setup: Optimal Approach

**Primary data collection (80% of recordings):**

- **Location:** Quiet room (not silent booth, just quiet)
- **Microphone:** Decent USB mic or quality headset
  - Samson Q2U
  - Blue Yeti
  - Rode NT-USB Mini
  - Even a good gaming headset like HyperX Cloud
- **Distance:** 6-12 inches from mic
- **Settings:** 16kHz or 48kHz sample rate, 16-bit or higher
- **Format:** WAV or FLAC (lossless)

**Supplementary real-world data (20% of recordings):**

- Record some sessions on your phone in typical conditions
- Use these to teach model phone mic characteristics
- Still try to minimize extreme noise

### Audio Quality Targets

**Goal:** Clean, clear speech with minimal but natural noise

**Good SNR (Signal-to-Noise Ratio):**

- Optimal: 30-40 dB SNR (very quiet background)
- Acceptable: 20-30 dB SNR (normal quiet room)
- Borderline: 15-20 dB SNR (noticeable background)
- Avoid: <15 dB SNR (loud background competing with voice)

**Check your recording:**

```bash
# Use ffmpeg to check audio levels
ffmpeg -i recording.wav -af "volumedetect" -f null /dev/null

# Look for:
# mean_volume: Should be around -20 dB to -30 dB
# max_volume: Should not be 0 dB (clipping)
```

### Data Augmentation Strategy

After recording clean data, augment programmatically:

#### 1. **Noise Addition**

```python
# Add realistic noise types
augmentations = [
    add_noise(audio, noise_type="cafe", snr=15),
    add_noise(audio, noise_type="traffic", snr=10),
    add_noise(audio, noise_type="office", snr=20),
    add_noise(audio, noise_type="home", snr=25),
]
```

**Noise sources:**

- Environmental noise datasets (AudioSet, FreeSound)
- Your own noise recordings (record 30s of each environment without speaking)
- Synthetic noise (white, pink, brown noise)

#### 2. **Microphone Simulation**

```python
# Simulate phone mic characteristics
phone_audio = apply_phone_mic_response(clean_audio)
```

**Techniques:**

- Frequency response curve (phone mics roll off bass/treble)
- Dynamic range compression
- Subtle distortion/clipping

#### 3. **Room Acoustics**

```python
# Add realistic reverb
reverb_audio = add_reverb(
    audio,
    room_size="small",    # or "medium", "large"
    decay_time=0.3        # seconds
)
```

#### 4. **Speed/Pitch Perturbation**

```python
# Slight variations to improve generalization
augmented = [
    change_speed(audio, factor=0.95),  # 5% slower
    change_speed(audio, factor=1.05),  # 5% faster
    change_pitch(audio, semitones=-1), # Slight pitch down
    change_pitch(audio, semitones=+1), # Slight pitch up
]
```

#### 5. **Volume Variation**

```python
# Simulate different recording distances
augmented = [
    change_volume(audio, factor=0.7),  # Quieter (further away)
    change_volume(audio, factor=1.3),  # Louder (closer)
]
```

### Recommended Mix for Training

From 10 hours of clean recordings, create:

- **40% original clean recordings** (4 hours)
- **30% with noise augmentation** (3 hours equivalent)
- **15% with mic simulation** (1.5 hours equivalent)
- **10% with reverb** (1 hour equivalent)
- **5% with speed/pitch perturbation** (0.5 hours equivalent)

**Total effective training data:** ~10 hours original → 15-20 hours augmented

## Tools for Data Augmentation

### Python Libraries

#### **audiomentations**

```python
from audiomentations import Compose, AddGaussianNoise, TimeStretch, PitchShift

augment = Compose([
    AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.5),
    TimeStretch(min_rate=0.8, max_rate=1.25, p=0.5),
    PitchShift(min_semitones=-4, max_semitones=4, p=0.5),
])

augmented_audio = augment(samples=audio, sample_rate=16000)
```

#### **torch-audiomentations**

```python
from torch_audiomentations import Compose, AddBackgroundNoise, ApplyImpulseResponse

augment = Compose([
    AddBackgroundNoise(
        background_paths="/path/to/noise/files",
        min_snr_in_db=10.0,
        max_snr_in_db=25.0,
        p=0.5
    ),
    ApplyImpulseResponse(
        ir_paths="/path/to/room/impulses",
        p=0.3
    )
])
```

#### **nlpaug**

```python
import nlpaug.augmenter.audio as naa

# Add noise
aug = naa.NoiseAug()
augmented = aug.augment(audio)

# Speed tuning
aug = naa.SpeedAug()
augmented = aug.augment(audio)
```

### Pre-built Noise Datasets

1. **MUSAN** (Music, Speech, and Noise corpus)
   - 900+ hours of noise, music, speech
   - Free download

2. **AudioSet**
   - Google's 2M+ audio clips
   - 600+ sound categories

3. **FreeSound**
   - Community-contributed sound effects
   - CC-licensed

4. **RIR (Room Impulse Response) databases**
   - Realistic room acoustics
   - Apply via convolution

## The Phone Mic Question

Since you mentioned using a phone as your primary inference device:

### Should you record ANY data on your phone?

**Yes, but as supplementary data:**

**Primary recordings:** Quality mic in quiet environment (80%)

**Phone recordings:** Actual phone in typical conditions (20%)

**Why this ratio:**

1. **Clean data teaches voice patterns:** 80% on quality mic ensures model learns your voice clearly
2. **Phone data teaches transfer:** 20% on phone teaches model to handle phone mic characteristics
3. **Augmentation fills gaps:** Noise augmentation covers various real-world scenarios

### Phone Recording Tips

When recording supplementary phone data:

1. **Consistent phone position:** Hold phone same way each time (e.g., 6 inches from mouth)
2. **Don't deliberately add extreme noise:** Normal environment is fine
3. **Use phone's best mic:** If phone has multiple mics (bottom, top), use the primary voice mic
4. **Avoid wind:** Even light wind creates massive artifacts on phone mics
5. **Monitor levels:** Don't shout (clipping) or whisper (too quiet)

## Real-World Testing Strategy

After training, test in progressive noise conditions:

### Test Set 1: Clean audio

- Similar to training conditions
- Expected: Best performance
- Baseline for comparison

### Test Set 2: Mild noise (20-30 dB SNR)

- Office, quiet café, home
- Expected: Slight degradation (5-15% WER increase)

### Test Set 3: Moderate noise (10-20 dB SNR)

- Busy café, car with windows up, urban street
- Expected: Noticeable degradation (15-30% WER increase)

### Test Set 4: Heavy noise (<10 dB SNR)

- Loud street, car with windows down, construction
- Expected: Significant degradation (30-50%+ WER increase)

**Augmentation effectiveness check:**

- If heavy noise has >80% WER: Need more aggressive noise augmentation
- If mild noise has >20% WER: Possible overfitting to clean data
- If clean audio performance is poor: Problem with base model training

## Exception: Training for Extreme Noise

If you ONLY use your model in extremely noisy conditions:

**Example:** Factory floor, construction site, loud machinery

**Then:** You might record more real-world data with that specific noise

**But still:**

1. Record some clean data (30-40%)
2. Record in-situ with real noise (60-70%)
3. Be aware: Model will specialize to this noise type, potentially at cost of clean performance

## Common Mistakes

### Mistake 1: Recording in silent booth

**Problem:** Too clean—doesn't match ANY real-world use

**Better:** Quiet room with natural ambient sound (computer fan, air conditioning—subtle background)

### Mistake 2: Recording with highly variable noise

**Problem:** Inconsistent training signal

**Better:** Consistent quiet environment, augment programmatically

### Mistake 3: Using low-quality mic to "match phone"

**Problem:** Captures poor voice representation

**Better:** Quality mic, then simulate phone response via augmentation

### Mistake 4: No augmentation

**Problem:** Model is brittle to noise

**Better:** Even simple Gaussian noise addition helps significantly

### Mistake 5: Over-augmentation

**Problem:** So much augmentation that original voice patterns are obscured

**Better:** Keep 30-50% clean data in final training set

## Conclusion

**Optimal strategy for ASR fine-tuning:**

1. **Record 80% in clean conditions with quality mic**
   - Quiet room (not silent)
   - Decent USB mic or headset
   - 16kHz+, lossless format

2. **Record 20% supplementary data on target device**
   - Phone recordings in typical use conditions
   - Don't seek out extreme noise

3. **Apply controlled augmentation**
   - Noise addition (various types, controlled SNR)
   - Microphone simulation
   - Room acoustics
   - Subtle speed/pitch variations

4. **Create balanced training set**
   - 40% clean
   - 40% augmented with noise
   - 20% real device recordings

5. **Test progressively**
   - Clean → Mild noise → Moderate noise → Heavy noise
   - Adjust augmentation based on results

**Why this works:**

- Clean data lets model learn your voice characteristics clearly
- Augmentation teaches noise robustness with controlled variety
- Real device data handles device-specific quirks
- Combined approach generalizes better than native noisy recording

Recording in deliberately noisy conditions seems logical but actually degrades the training signal you need. Let the model learn your voice clearly first, then teach it robustness through systematic augmentation.

---

*Note: This document was generated by Claude Code, an AI assistant. Please validate technical details and test recommendations in your specific environment before implementing.*
