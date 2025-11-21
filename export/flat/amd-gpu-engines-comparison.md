# AMD GPU Engines for Speech-to-Text: A Comprehensive Comparison

## Question
With an AMD GPU (rather than NVIDIA), what are the best inference engines for ASR that have good AMD/ROCm support? And when converting models from safe-tensors format after fine-tuning, which formats should you target to work well with AMD GPUs?

## Answer

Running speech-to-text (STT) workloads locally on AMD GPUs presents unique challenges compared to NVIDIA's more mature CUDA ecosystem. This guide provides a comprehensive comparison of the most reliable engines for AMD GPU-accelerated STT inference.

## Current State of AMD GPU Support for STT

### The Challenge
AMD GPU support for AI workloads relies primarily on ROCm (Radeon Open Compute), which has historically lagged behind NVIDIA's CUDA in terms of software support and ecosystem maturity. Many popular inference engines were built with CUDA as the primary target, with AMD support added later or not at all.

## Engine Comparison

### 1. **Faster-Whisper** (Recommended)

**Status:** Most reliable option for AMD GPU acceleration

**Key Details:**
- Uses CTranslate2 backend, which has ROCm support
- Supports AMD GPUs through ROCm (tested with gfx1100, gfx1101, gfx1030, and other RDNA architectures)
- Offers 3-4x faster inference than OpenAI's Whisper while maintaining accuracy
- Lower VRAM requirements than original Whisper

**Installation:**
```bash
pip install faster-whisper
```

**ROCm Requirements:**
- ROCm 5.4+ recommended
- Proper `HSA_OVERRIDE_GFX_VERSION` may be needed for some cards
- For RDNA 3 (RX 7000 series): `HSA_OVERRIDE_GFX_VERSION=11.0.0` or `11.0.1`

**Verification of GPU Usage:**
```bash
# Monitor AMD GPU usage
watch -n 1 rocm-smi

# Or with more detail
watch -n 1 'rocm-smi --showuse --showmeminfo vram'
```

**Pros:**
- Best balance of speed, accuracy, and AMD GPU support
- Active development and community
- Good documentation for ROCm setup

**Cons:**
- Initial setup can be tricky
- ROCm version compatibility issues

### 2. **whisper.cpp**

**Status:** Mixed AMD GPU support - unreliable for production

**Key Details:**
- Primarily CPU-optimized (uses SIMD, AVX, etc.)
- HIP (ROCm) support exists but is experimental
- Must be compiled from source with specific flags for AMD GPU support
- GPU acceleration often doesn't engage properly

**Why Your GPU Monitoring Showed No Activity:**
The app you're using likely:
1. Uses a pre-compiled binary without ROCm support
2. Falls back to CPU when GPU initialization fails
3. Doesn't have proper ROCm runtime environment configured

**When to Use:**
- CPU-only inference (where it excels)
- Embedded/edge devices
- When you need minimal dependencies

**Pros:**
- Excellent CPU performance
- Low memory footprint
- Fast for CPU-only workloads

**Cons:**
- AMD GPU support is experimental and unreliable
- Requires manual compilation with HIP support
- Often falls back to CPU silently

### 3. **OpenAI Whisper (Original)**

**Status:** No direct AMD GPU support through PyTorch

**Key Details:**
- Built on PyTorch with CUDA backend
- PyTorch has experimental ROCm support (separate installation)
- Slower than optimized alternatives
- Higher VRAM requirements

**ROCm PyTorch Installation:**
```bash
# ROCm PyTorch (example for ROCm 6.0)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.0
```

**Pros:**
- Reference implementation
- Most accurate (though Faster-Whisper matches it)
- Good for testing/validation

**Cons:**
- Slowest inference
- ROCm PyTorch support is hit-or-miss
- High VRAM usage
- Not optimized for inference

### 4. **Whisper-JAX**

**Status:** Limited AMD support through experimental ROCm JAX builds

**Key Details:**
- Built on JAX framework
- JAX has experimental ROCm support but very limited
- Primarily designed for TPU/CUDA

**When to Use:**
- You have specific JAX ROCm expertise
- Research/experimental workloads

**Recommendation:** Avoid for AMD GPU production use

### 5. **Whisper.onnx / ONNX Runtime**

**Status:** Growing AMD GPU support through DirectML and ROCm

**Key Details:**
- ONNX Runtime has ROCm execution provider
- Can convert Whisper models to ONNX format
- DirectML provider works on Windows with AMD GPUs

**Installation:**
```bash
# For ROCm
pip install onnxruntime-rocm

# Model conversion needed
python -m optimum.exporters.onnx --model openai/whisper-base whisper-onnx/
```

**Pros:**
- Cross-platform
- Good optimization potential
- Growing ecosystem

**Cons:**
- Requires model conversion
- ROCm provider less mature than CUDA
- More complex setup

## Ranking for AMD GPU Users

### Tier 1: Production-Ready
1. **Faster-Whisper** - Best overall choice for AMD GPUs
   - Reliability: ⭐⭐⭐⭐⭐
   - Performance: ⭐⭐⭐⭐⭐
   - Ease of Setup: ⭐⭐⭐⭐

### Tier 2: Workable with Caveats
2. **OpenAI Whisper + ROCm PyTorch** - Reference implementation
   - Reliability: ⭐⭐⭐
   - Performance: ⭐⭐⭐
   - Ease of Setup: ⭐⭐⭐

3. **ONNX Runtime (ROCm provider)** - For specific use cases
   - Reliability: ⭐⭐⭐
   - Performance: ⭐⭐⭐⭐
   - Ease of Setup: ⭐⭐

### Tier 3: Not Recommended for AMD GPU
4. **whisper.cpp** - CPU-focused, unreliable GPU support
   - Reliability (GPU): ⭐⭐
   - Performance (CPU): ⭐⭐⭐⭐⭐
   - Ease of Setup (GPU): ⭐

5. **Whisper-JAX** - Limited ROCm support
   - Reliability: ⭐
   - Performance: N/A
   - Ease of Setup: ⭐

## Practical Recommendations

### For Your Use Case

Given that you're using an app with whisper.cpp and not seeing GPU activity, here's what's likely happening:

1. **The app is using CPU-only whisper.cpp** - Most pre-packaged apps don't include ROCm-compiled versions
2. **GPU support is claimed but not functional** - The app may have been tested only with NVIDIA GPUs
3. **Silent fallback to CPU** - whisper.cpp will use CPU if GPU initialization fails

### Action Plan

**Option A: Switch to Faster-Whisper (Recommended)**
```bash
# Install ROCm if not already installed
# (instructions vary by distro)

# Install faster-whisper
pip install faster-whisper

# Test script
python << EOF
from faster_whisper import WhisperModel

model = WhisperModel("base", device="cuda", compute_type="float16")
segments, info = model.transcribe("audio.wav")

for segment in segments:
    print(f"[{segment.start:.2f}s -> {segment.end:.2f}s] {segment.text}")
EOF

# Monitor GPU while running
watch -n 1 rocm-smi
```

**Option B: Verify whisper.cpp ROCm Support**
If you want to stick with your current app:
1. Check if the app supports custom whisper.cpp builds
2. Compile whisper.cpp with HIP support:
```bash
git clone https://github.com/ggerganov/whisper.cpp
cd whisper.cpp
mkdir build && cd build
cmake .. -DWHISPER_HIPBLAS=ON
make
```
3. Replace the app's whisper.cpp binary with your ROCm-enabled build

**Option C: Use PyTorch ROCm + Original Whisper**
For research/development:
```bash
# Install ROCm PyTorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.0

# Install Whisper
pip install -U openai-whisper

# Verify GPU detection
python -c "import torch; print(torch.cuda.is_available())"
```

## Verifying GPU Usage on AMD

### ROCm System Management Interface
```bash
# Basic monitoring
rocm-smi

# Detailed monitoring
rocm-smi --showuse --showmeminfo vram --showtemp

# Continuous monitoring
watch -n 1 rocm-smi
```

### Process-Specific GPU Usage
```bash
# Install radeontop for detailed GPU monitoring
sudo apt install radeontop
radeontop

# Or use rocm-smi with PID
rocm-smi --showpids
```

### PyTorch ROCm Verification
```python
import torch
print(f"ROCm available: {torch.cuda.is_available()}")
print(f"ROCm version: {torch.version.hip}")
print(f"Device name: {torch.cuda.get_device_name(0)}")
```

## Common Pitfalls

### 1. Silent CPU Fallback
Many inference engines will silently fall back to CPU if GPU initialization fails. Always verify GPU usage with monitoring tools.

### 2. HSA_OVERRIDE_GFX_VERSION
RDNA 2/3 GPUs often need:
```bash
export HSA_OVERRIDE_GFX_VERSION=11.0.0  # For gfx1100 (RX 7900 XTX)
export HSA_OVERRIDE_GFX_VERSION=11.0.1  # For gfx1101 (RX 7800 XT)
export HSA_OVERRIDE_GFX_VERSION=10.3.0  # For gfx1030 (RX 6000 series)
```

### 3. ROCm Version Compatibility
Different inference engines support different ROCm versions. Check compatibility before installation.

### 4. Pre-compiled Binaries
Most pre-compiled applications and Python wheels are CUDA-only. AMD GPU support often requires:
- Custom compilation
- Specific ROCm wheels
- Environment configuration

## Model Recommendations for AMD GPUs

When using Faster-Whisper or other ROCm-enabled engines:

### VRAM Considerations
- **4-6GB VRAM:** `tiny`, `base`, `small` models
- **8-12GB VRAM:** `medium` model
- **16GB+ VRAM:** `large-v2`, `large-v3` models

### Quantization
Faster-Whisper supports various quantization levels:
- `float16` - Best quality, 2x memory reduction
- `int8` - Good quality, 4x memory reduction
- `int8_float16` - Hybrid approach (recommended)

```python
model = WhisperModel("large-v3", device="cuda", compute_type="int8_float16")
```

## Future Outlook

### Improving AMD Support
- ROCm 6.x+ shows significant improvements
- More inference frameworks adding ROCm support
- Community-driven optimizations growing

### Alternative Approaches
- **Containers:** Pre-configured ROCm containers simplify setup
- **Cloud inference:** Consider cloud GPUs for production if local AMD support remains problematic
- **Hybrid approach:** Use AMD GPU for development, NVIDIA for production

## Conclusion

For AMD GPU users running local STT workloads, **Faster-Whisper** is currently the most reliable choice. It offers:
- Proven ROCm support
- Excellent performance
- Active development
- Straightforward setup

The whisper.cpp issues you're experiencing are common - the engine excels at CPU inference but has unreliable AMD GPU support. Unless your application specifically requires whisper.cpp, switching to Faster-Whisper will provide better AMD GPU utilization and performance.

---

*This document was generated by Claude Code as part of Daniel Rosehill's STT Fine-Tuning Notebook. While comprehensive, please verify specific technical details and version compatibility for your use case. AMD GPU support in AI workloads evolves rapidly.*
