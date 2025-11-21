# AMD GPU + ROCm: Optimal Inference Strategies for ASR

## Question
With an AMD GPU and ROCm, what inference engines have the best support for ASR, and what model formats should you target when converting from safe-tensors after fine-tuning?

## Answer

### The AMD/ROCm Reality Check

You're right—NVIDIA's CUDA ecosystem dominates AI, and AMD/ROCm support is more limited. However, for ASR specifically (and Whisper in particular), there are well-supported pathways that work excellently on AMD.

### Best-Supported Inference Engines for AMD + ROCm

#### 1. **CTranslate2 (Recommended - Best Performance)**

**Why CTranslate2:**
- Purpose-built for transformer inference optimization
- Excellent ROCm support (officially supported)
- Used by Faster-Whisper (the fastest Whisper implementation)
- 4x faster than PyTorch, significantly lower memory usage
- Supports quantization (INT8, FP16)

**ROCm Compatibility:**
- Works with ROCm 5.0+
- Your GPU (gfx1101 - Navi 32) is well-supported
- Requires `HSA_OVERRIDE_GFX_VERSION=11.0.1` (which you're already using)

**Target Format:**
```
Safetensors/PyTorch → CTranslate2 format (.ctranslate2/)
```

**Conversion Process:**
```bash
# Install CTranslate2 with ROCm support
pip install ctranslate2

# Convert your fine-tuned model
ct2-transformers-converter --model /path/to/finetuned-whisper \
    --output_dir /path/to/ctranslate2-model \
    --quantization float16  # or int8 for faster inference
```

**Why This Works for AMD:**
- CTranslate2 uses optimized ROCm kernels
- Well-maintained AMD support
- Active community using it on AMD GPUs

#### 2. **ONNX Runtime with ROCm Execution Provider**

**Why ONNX Runtime:**
- Open standard (ONNX format)
- Microsoft-backed with official ROCm support
- Good performance (though not as fast as CTranslate2 for Whisper)
- Wide compatibility across frameworks

**ROCm Compatibility:**
- ONNXRuntime 1.14+ has ROCmExecutionProvider
- Works on gfx1101 with ROCm 5.4+

**Target Format:**
```
Safetensors/PyTorch → ONNX (.onnx)
```

**Conversion Process:**
```python
from optimum.onnxruntime import ORTModelForSpeechSeq2Seq
from transformers import WhisperProcessor

# Load your fine-tuned model
model = ORTModelForSpeechSeq2Seq.from_pretrained(
    "path/to/finetuned-whisper",
    export=True,
    provider="ROCMExecutionProvider"
)

# Save in ONNX format
model.save_pretrained("path/to/onnx-model")
```

**Optimization:**
```bash
# Quantization for faster inference
python -m onnxruntime.quantization.preprocess \
    --input model.onnx \
    --output model-quantized.onnx
```

#### 3. **PyTorch with ROCm Backend (Fallback Option)**

**Why PyTorch:**
- Native format (no conversion needed)
- Most flexible for experimentation
- Good ROCm support (AMD maintains torch-rocm)
- Easier debugging

**ROCm Compatibility:**
- PyTorch 2.0+ has solid ROCm support
- Works directly with safetensors/PyTorch checkpoints

**Target Format:**
```
Safetensors/PyTorch (native) - no conversion needed
```

**Usage:**
```python
import torch
from transformers import WhisperForConditionalGeneration

# Load directly (ROCm will be used if available)
model = WhisperForConditionalGeneration.from_pretrained(
    "path/to/finetuned-whisper"
).to("cuda")  # "cuda" works with ROCm

# Use torch.compile for optimization (PyTorch 2.0+)
model = torch.compile(model)
```

**Performance:**
- Slower than CTranslate2 or ONNX
- Higher memory usage
- But most straightforward for debugging

### Comparison Table

| Engine | Performance | ROCm Support | Conversion Complexity | Best Use Case |
|--------|-------------|--------------|----------------------|---------------|
| **CTranslate2** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | Medium | Production inference |
| **ONNX Runtime** | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | Medium | Cross-platform deployment |
| **PyTorch** | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | None | Development/debugging |

### Recommended Workflow for Your AMD Setup

#### **Primary Path: CTranslate2 (Faster-Whisper)**

This is the optimal choice for your AMD GPU:

```bash
# 1. Fine-tune in PyTorch (standard process)
# Your model is saved as safetensors/pytorch_model.bin

# 2. Convert to CTranslate2
ct2-transformers-converter \
    --model ./finetuned-whisper-medium \
    --output_dir ./finetuned-whisper-medium-ct2 \
    --quantization float16

# 3. Use with faster-whisper
pip install faster-whisper
```

```python
from faster_whisper import WhisperModel

# Load your fine-tuned CTranslate2 model
model = WhisperModel(
    "path/to/finetuned-whisper-medium-ct2",
    device="cuda",  # Works with ROCm
    compute_type="float16"
)

# Inference
segments, info = model.transcribe("audio.wav")
for segment in segments:
    print(f"[{segment.start:.2f}s -> {segment.end:.2f}s] {segment.text}")
```

#### **Why This Works Well on AMD:**

1. **Optimized Kernels**: CTranslate2 uses ROCm-optimized kernels
2. **Lower Memory**: Your 7700 XT/7800 XT has less VRAM than NVIDIA equivalents—CTranslate2's efficiency helps
3. **Proven Track Record**: Many AMD users successfully run faster-whisper
4. **Active Maintenance**: CTranslate2 team actively supports ROCm

### Format Conversion Summary

```
Post Fine-Tuning Workflow:

1. Training Output:
   ├── safetensors (raw weights)
   ├── pytorch_model.bin
   └── config.json

2. Convert to Target Format:
   ├── CTranslate2 (RECOMMENDED for AMD)
   │   └── Use ct2-transformers-converter
   │
   ├── ONNX (Good alternative)
   │   └── Use optimum.onnxruntime
   │
   └── Keep PyTorch (Development only)
       └── No conversion needed
```

### AMD-Specific Optimizations

**Environment Variables (You're Likely Already Using):**
```bash
export HSA_OVERRIDE_GFX_VERSION=11.0.1  # For gfx1101
export ROCM_PATH=/opt/rocm
export ROC_ENABLE_PRE_VEGA=0
```

**Quantization Strategy:**
- **FP16**: Best balance (2x faster, minimal quality loss)
- **INT8**: 4x faster, slight quality degradation
- **FP32**: Slowest, unnecessary for inference

**Batch Size Tuning:**
Your 7700 XT/7800 XT has 12GB VRAM:
- Whisper tiny: batch size 16-32
- Whisper small: batch size 8-16
- Whisper medium: batch size 4-8
- Whisper large: batch size 1-2 (or use quantization)

### What NOT to Use on AMD

❌ **TensorRT**: NVIDIA-only, won't work
❌ **CUDA-specific libraries**: FlashAttention, etc.
❌ **Some quantization tools**: GPTQ, AWQ (CUDA-focused)

### Bottom Line Recommendation

**For your AMD GPU (gfx1101) + ROCm setup:**

1. **Best Performance**: Fine-tune in PyTorch → Convert to CTranslate2 → Use faster-whisper
2. **Best Compatibility**: ONNX Runtime with ROCm execution provider
3. **Easiest Debugging**: Stay in PyTorch

**The conversion command you'll use most:**
```bash
ct2-transformers-converter \
    --model /path/to/your-finetuned-whisper \
    --output_dir /path/to/optimized-model-ct2 \
    --quantization float16
```

This gives you near-NVIDIA performance on AMD hardware for ASR inference.

---

**Note**: This guidance was generated by Claude Code (claude-sonnet-4-5) for Daniel Rosehill's STT Fine-Tuning Notebook. ROCm support varies by version—always verify compatibility with your specific ROCm version (`rocm-smi --showdriverversion`). For production deployments, test inference performance with your specific audio data.
