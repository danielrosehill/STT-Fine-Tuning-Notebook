# GPU and VRAM Requirements for Local Whisper Inference

## The Observation

Running Whisper Medium on an AMD Radeon RX 7700 XT (8GB VRAM) via whisper.cpp shows:

- GPU usage jumping to **100% during inference**
- Inference happens every few seconds during streaming transcription
- Surprising that Medium model maxes out the GPU

**Questions:**

1. Does 100% GPU usage mean the model is too large?
2. How much VRAM/GPU power do you really need for quality local STT?
3. Is hitting 90-100% GPU utilization during inference problematic?

## Short Answer

**100% GPU usage during inference is completely normal and expected—it's actually ideal!** This means:

- You're using your GPU efficiently
- The model is running at full speed
- This is NOT a problem or bottleneck
- You do NOT need a bigger GPU

**The concern about "maxing out" the GPU is based on a misconception:** Unlike gaming where 100% GPU means you're bottlenecked, in AI inference, 100% utilization during processing is the *goal*. Between inference bursts, GPU usage drops back down—this is normal streaming behavior.

## Understanding GPU Utilization in AI Inference

### Gaming/Graphics Workload (Continuous)

```
Timeline: [████████████████████████████] 100% sustained
Meaning:  GPU struggling to keep up with frame rate demands
Problem:  You need a better GPU or lower settings
```

In gaming, continuous 100% means bottleneck.

### AI Inference Workload (Bursty)

```
Timeline: [████____████____████____████] Bursts to 100%
Meaning:  GPU efficiently processing, then idle, then processing
Normal:   This is optimal behavior
```

In AI inference, bursts to 100% mean efficient utilization.

### Your Whisper.cpp Streaming Case

```
Every 3 seconds:
  [Recording audio]     GPU: 0-5%   ← Waiting for audio
  [Inference begins]    GPU: 100%   ← Processing audio
  [Inference complete]  GPU: 0-5%   ← Done, waiting
```

**This pattern is perfect.** You want GPU to spike to 100% during the brief inference, then return to idle.

## VRAM Requirements vs GPU Compute

Two separate concerns:

### 1. VRAM Capacity (Memory Size)

**What it determines:** Which model size you can load

**Whisper model VRAM requirements:**

| Model | Parameters | VRAM (FP16) | VRAM (INT8) | VRAM (Q5) | VRAM (Q4) |
|-------|-----------|-------------|-------------|-----------|-----------|
| Tiny | 39M | ~150 MB | ~80 MB | ~50 MB | ~40 MB |
| Base | 74M | ~290 MB | ~150 MB | ~100 MB | ~80 MB |
| Small | 244M | ~950 MB | ~480 MB | ~350 MB | ~280 MB |
| Medium | 769M | ~3.0 GB | ~1.5 GB | ~1.1 GB | ~900 MB |
| Large-v2 | 1550M | ~6.0 GB | ~3.0 GB | ~2.2 GB | ~1.8 GB |
| Large-v3 | 1550M | ~6.0 GB | ~3.0 GB | ~2.2 GB | ~1.8 GB |
| Large-v3-turbo | 809M | ~3.1 GB | ~1.6 GB | ~1.2 GB | ~1.0 GB |

**Your AMD RX 7700 XT (8GB VRAM) can handle:**

- ✓ Medium (FP16, INT8, all quantizations) with room to spare
- ✓ Large-v3-turbo (FP16, INT8, quantized)
- ✓ Large-v2/v3 (INT8 and quantized versions)
- ✗ Large-v2/v3 (FP16) - would use ~6GB, leaving only 2GB for system

**Whisper.cpp default:** Usually uses Q5 or Q4 quantization, so your 8GB is plenty even for Large models

### 2. GPU Compute Power (Processing Speed)

**What it determines:** How *fast* inference runs

**AMD RX 7700 XT specs:**

- Compute Units: 54
- Peak FP16 performance: ~35 TFLOPS
- Memory bandwidth: 432 GB/s
- Architecture: RDNA 3 (Navi 32)

**This is a mid-to-upper-tier GPU**—very capable for local AI.

## Decoding the "100% GPU Usage"

### What's Actually Happening

When whisper.cpp processes audio:

```python
# Simplified inference flow
audio_chunk = capture_audio(3_seconds)  # GPU: 0%

# Load audio into GPU memory
gpu_buffer = transfer_to_gpu(audio_chunk)  # GPU: 5-10%

# Run inference (THE BIG COMPUTATION)
transcription = model.forward(gpu_buffer)   # GPU: 100%
                                            # Duration: 0.5-2 seconds

# Return result
print(transcription)  # GPU: 0%

# Wait for next chunk
time.sleep(1)  # GPU: 0%
```

**Your observation:** GPU hits 100% during `model.forward()`

**This is correct and optimal!** You *want* the GPU to work at full capacity during inference.

### Why This Isn't a Problem

**1. Inference is short:** Even at 100%, each inference burst lasts only 0.5-2 seconds

**2. Duty cycle is low:** If inference takes 1 second every 3 seconds, that's only 33% average utilization

**3. Temperature managed:** AMD GPUs throttle if they overheat—100% for 1 second won't cause thermal issues

**4. No frame drops:** Unlike gaming, there's no frame rate to drop. Either inference finishes or it doesn't—and yours is finishing successfully.

### The Display/System Concern

**Your question:** "Doesn't GPU also need to run displays?"

**Answer:** GPU time-shares, and display composition uses negligible compute:

```
GPU time allocation (simplified):
[Inference: 0.8s] [Display: 0.01s] [Idle: 1.99s] [Inference: 0.8s] ...
```

**Display needs:** ~5-20ms per frame at 60 FPS = ~0.3-1% of GPU time

Even at 100% inference utilization, there's enough GPU time between frames for display updates. You'd notice display issues (stuttering, lag) if this were a problem—and you haven't mentioned any.

## Real-World Performance Expectations

### Inference Speed (Real-Time Factor)

**Real-Time Factor (RTF):** How long to transcribe vs audio duration

```
RTF = inference_time / audio_duration

RTF = 1.0 → Real-time (1 second to process 1 second of audio)
RTF = 0.5 → 2× real-time (0.5 seconds to process 1 second of audio)
RTF = 2.0 → 0.5× real-time (2 seconds to process 1 second of audio)
```

**Your AMD RX 7700 XT expected performance (whisper.cpp with ROCm):**

| Model | RTF (approx) | Meaning |
|-------|-------------|---------|
| Tiny | 0.05-0.1 | 10-20× real-time |
| Base | 0.1-0.15 | 6-10× real-time |
| Small | 0.2-0.3 | 3-5× real-time |
| Medium | 0.4-0.6 | 1.6-2.5× real-time |
| Large-v3 | 0.7-1.0 | 1-1.4× real-time |
| Large-v3-turbo | 0.5-0.7 | 1.4-2× real-time |

**Your Medium at ~100% GPU usage likely achieving RTF ≈ 0.5**, meaning it's processing 2× faster than real-time—which is *excellent* for streaming transcription.

### What "Quality Results" Requires

**Myth:** High GPU utilization = poor quality results

**Reality:** Quality depends on:

1. **Model accuracy** (Medium is highly accurate)
2. **Successful completion** (your transcriptions are working)
3. **Reasonable latency** (you're getting results every few seconds)

**GPU utilization percentage is irrelevant to output quality.** As long as inference completes successfully (which it is), you're getting full-quality results.

## When Would You Actually Need More GPU?

You'd need a bigger GPU if:

### 1. Real-Time Factor Too Slow

```
Your audio: 3 seconds
Inference time: 4+ seconds
Result: Transcription falls behind
```

**Your case:** Not happening—Medium is processing faster than real-time

### 2. Running Multiple Models Simultaneously

```
Whisper + Stable Diffusion + LLM inference
Result: Out of VRAM or extreme slowdown
```

**Your case:** Only running Whisper

### 3. Batch Processing Many Files

```
Processing 100 audio files
Want: 5× faster throughput
Result: Larger GPU would help batch processing
```

**Your case:** Streaming transcription—batch speed less relevant

### 4. Using Unquantized Large Models

```
Loading Large-v3 in FP16: 6GB VRAM
Remaining: 2GB for system
Result: Might struggle with very large models in full precision
```

**Your case:** whisper.cpp uses quantization—you're fine

## Optimizing Your Current Setup

You don't need a new GPU, but you can optimize:

### 1. Ensure ROCm is Properly Configured

```bash
# Check ROCm installation
rocm-smi

# Should show your RX 7700 XT
# If not detected, ROCm might not be working
```

**If whisper.cpp falls back to CPU:** Performance would be much worse, but wouldn't show 100% GPU usage

### 2. Try Large-v3-Turbo

```bash
# Download turbo model
whisper.cpp --model large-v3-turbo input.wav

# Benefits:
# - Better accuracy than Medium
# - Only slightly more VRAM (~1GB vs ~900MB in Q4)
# - Faster inference than Large-v3
# - Should still run well on your GPU
```

**Expected:** GPU still hits 100% during inference (which is fine), but possibly slightly longer bursts

### 3. Check Thermal Throttling

```bash
# Monitor GPU temperature
watch -n 1 rocm-smi

# Look for:
# Temperature: Should stay under 85°C
# Clock speed: Should stay at boost clocks during inference
```

**If throttling:** GPU automatically reduces clock speed when hot—this *would* hurt performance, but 100% utilization doesn't necessarily mean throttling

### 4. Monitor VRAM Usage, Not Just Utilization

```bash
# Check actual VRAM usage
rocm-smi | grep "Memory"

# For Medium model:
# Should see ~1-1.5GB used (plenty of headroom)
```

**If VRAM is nearly full (>7GB):** Then you're at the limit

**If VRAM usage is low (~1-2GB):** You have lots of headroom

## Model Selection Guide for Your GPU

**Your AMD RX 7700 XT (8GB) can comfortably run:**

### Recommended for Quality + Speed Balance:

**1. Large-v3-Turbo (best choice)**

- Accuracy: 90-95% of Large-v3
- Speed: ~1.4-2× real-time on your GPU
- VRAM: ~1GB (Q4 quantization)
- **Best overall option**

**2. Medium (what you're using)**

- Accuracy: Excellent for most use cases
- Speed: ~2-2.5× real-time on your GPU
- VRAM: ~900MB (Q4 quantization)
- **Very solid choice, no need to change unless you want better accuracy**

### If You Want Maximum Accuracy:

**3. Large-v3 (quantized)**

- Accuracy: Best available
- Speed: ~1-1.4× real-time on your GPU
- VRAM: ~2GB (Q4 quantization)
- **Slight latency increase, but still real-time capable**

### If You Want Maximum Speed:

**4. Small**

- Accuracy: Good for clean audio
- Speed: ~3-5× real-time on your GPU
- VRAM: ~300MB
- **Fast, but noticeably less accurate than Medium**

## Comparing Your GPU to Others

**Your AMD RX 7700 XT ranks:**

| GPU Class | Example | Whisper Medium RTF | Can Handle Large? |
|-----------|---------|-------------------|-------------------|
| **Entry-level** | GTX 1650, RX 6500 XT | 0.8-1.2 | Barely |
| **Mid-range** | RTX 3060, RX 6700 XT | 0.5-0.7 | Yes (quantized) |
| **Your tier** | RX 7700 XT, RTX 3070 | 0.4-0.6 | Yes, easily |
| **High-end** | RTX 4070 Ti, RX 7900 XT | 0.3-0.4 | Yes, very fast |
| **Flagship** | RTX 4090, RX 7900 XTX | 0.15-0.25 | Yes, blazing |

**You're in a very good tier for local STT.** A 4090 would be ~2× faster, but you're already faster than real-time, so it wouldn't meaningfully improve user experience.

## The Psychology of 100%

**Why 100% *feels* wrong:**

- Gaming culture: 100% GPU = "maxed out", need upgrade
- CPU usage: 100% CPU often means system is struggling
- Temperature concerns: High utilization = heat

**Why 100% is actually *right* for AI inference:**

- You're paying for compute—use it!
- Burst workload: 100% for 1 second every 3 seconds ≠ sustained load
- Efficient resource usage: Idle GPU is wasted GPU during inference
- No quality impact: Model runs full computation regardless

**Better metrics to watch:**

- ✓ Inference speed (faster than real-time?)
- ✓ VRAM usage (under 7GB?)
- ✓ Temperature (under 85°C?)
- ✓ Transcription latency (acceptable?)
- ✗ GPU utilization percentage (irrelevant for quality)

## Recommendations

### What You Should Do

**1. Keep using Medium—it's working great!**

- Your GPU is handling it well
- 100% utilization during inference is optimal
- Results are good quality

**2. Optionally try Large-v3-Turbo**

```bash
# Better accuracy with acceptable speed
whisper.cpp --model large-v3-turbo
```

- Test if accuracy improvement is worth slight latency increase
- Your GPU can handle it

**3. Monitor VRAM and temperature, not utilization**

```bash
# Useful monitoring
watch -n 1 'rocm-smi | grep -E "Temperature|Memory"'
```

- VRAM <7GB? ✓ You're fine
- Temperature <85°C? ✓ You're fine
- Utilization 100%? ✓ This is correct!

### What You Should NOT Do

**✗ Don't upgrade GPU based on 100% utilization**

- You're not bottlenecked
- Inference is faster than real-time
- Quality is excellent

**✗ Don't drop to Small/Tiny to "reduce GPU load"**

- You'd lose accuracy for no benefit
- GPU sitting at 50% instead of 100% doesn't help anything
- Use the compute power you have!

**✗ Don't worry about "maxing out" GPU**

- This isn't gaming
- Burst loads are normal and healthy
- Your GPU is designed for this

## Ballpark VRAM Requirements

**For quality local STT with Whisper:**

| Use Case | Recommended | VRAM Needed | GPU Example |
|----------|-------------|-------------|-------------|
| **Minimum viable** | Small | 2GB | GTX 1050 Ti |
| **Good experience** | Medium | 4GB | GTX 1660 |
| **Excellent** | Medium/Large-Turbo | 6-8GB | RX 7700 XT, RTX 3070 |
| **Best** | Large-v3 | 8-12GB | RTX 3080, RX 7900 XT |
| **Overkill** | Large + fine-tunes | 16GB+ | RTX 4090 |

**You're in the "Excellent" tier.** You don't need more VRAM for quality local STT—you already have it.

## Conclusion

**Your AMD RX 7700 XT (8GB VRAM) is more than sufficient for quality local STT.**

**100% GPU utilization during inference is:**

- ✓ Normal and expected
- ✓ Sign of efficient resource usage
- ✓ Not a problem or bottleneck
- ✓ Not affecting quality

**You can comfortably run:**

- Medium (what you're using)—excellent choice
- Large-v3-Turbo—better accuracy, still good speed
- Large-v3 (quantized)—best accuracy, acceptable speed

**You should NOT:**

- Worry about 100% GPU spikes
- Think you need to upgrade
- Drop to smaller models to "reduce load"

**The display/system concern is unfounded:** Display composition takes <1% of GPU time and doesn't compete meaningfully with inference bursts.

**Your whisper.cpp setup is working optimally.** Enjoy your fast, accurate local transcription!

---

*Note: This document was generated by Claude Code, an AI assistant. Please validate technical details and test recommendations in your specific environment before implementing.*
