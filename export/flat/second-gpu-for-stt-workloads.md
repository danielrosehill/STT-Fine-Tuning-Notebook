# Second GPU for STT Workloads: Hardware Strategy

## Question Summary

Daniel is considering hardware upgrade strategies for his desktop setup. He currently uses an AMD GPU (chosen before getting into AI/STT work) and primarily uses speech-to-text throughout his workday. He's evaluating two options:
1. Replace the AMD GPU with an NVIDIA GPU
2. Keep the AMD GPU for display output and add a second GPU dedicated to STT processing

The question extends to whether a GPU, NPU, or TPU would be optimal for STT-only workloads, given that local LLM usage is minimal.

## Recommendation: Strategic Hardware Considerations

### Primary Recommendation: Single NVIDIA GPU (Next Build)

For your next build or major upgrade, I'd recommend **replacing your AMD GPU with a single NVIDIA GPU** rather than adding a second GPU. Here's why:

**Advantages:**
- **Simplified system management**: One GPU means less power consumption, heat, driver complexity, and PCIe lane allocation issues
- **CUDA ecosystem dominance**: The vast majority of AI/ML tools (including STT) have first-class CUDA support with better optimization
- **Flexibility**: A single NVIDIA GPU can handle both display and AI workloads efficiently
- **Better per-dollar performance**: You get more AI performance for your money with a single higher-tier NVIDIA card than splitting budget across two GPUs
- **Lower power draw**: Modern NVIDIA GPUs (especially 4000 series) are remarkably power-efficient for AI workloads

**Recommended GPU Tiers for STT + Light LLM:**

1. **Budget Option (~$500-600)**: NVIDIA RTX 4060 Ti 16GB
   - 16GB VRAM is crucial for larger Whisper models and future-proofing
   - Excellent for STT inference (Whisper large-v3 runs smoothly)
   - Can handle local LLMs up to 13B parameters reasonably well
   - Low power consumption (~160W TDP)

2. **Mid-Range Sweet Spot (~$800-1000)**: NVIDIA RTX 4070 Ti / 4070 Ti Super
   - 12GB VRAM (4070 Ti) or 16GB VRAM (4070 Ti Super)
   - Significantly faster inference for Whisper
   - Better headroom for local LLM experimentation
   - Still reasonable power draw (~285W TDP)

3. **High-End Option (~$1200-1500)**: NVIDIA RTX 4080 / 4080 Super
   - 16GB VRAM
   - Overkill for STT alone, but excellent for any AI workload you might explore
   - Near-workstation performance for AI tasks

### Why Not a Second GPU?

**Technical Drawbacks:**
- **PCIe lane limitations**: Most consumer motherboards don't have enough PCIe lanes to run two GPUs at full bandwidth, meaning you'd likely run both at x8 instead of x16
- **Power supply requirements**: You'd need a significantly larger PSU (likely 1000W+)
- **Heat and cooling**: Two GPUs generate substantial heat; your case might not have adequate cooling
- **Driver complexity**: Running AMD for display + NVIDIA for compute adds driver management overhead
- **ROCm limitations**: Your current AMD GPU already struggles with ROCm support for AI (as you've likely experienced), so keeping it doesn't provide much benefit

**Cost Consideration:**
A mid-range NVIDIA GPU (~$800) would likely provide better AI performance than your current AMD GPU + a budget NVIDIA card costing the same total amount.

### GPU vs NPU vs TPU for STT

**GPU (Recommended for STT):**
- ✅ Best option for STT workloads
- ✅ Whisper and similar models are heavily optimized for GPU
- ✅ Flexibility for other AI tasks (image generation, LLMs)
- ✅ Mature software ecosystem (PyTorch, ONNX, faster-whisper, CTranslate2)

**NPU (Neural Processing Unit):**
- ❌ Not recommended for desktop STT
- NPUs are designed for low-power inference on mobile/edge devices
- Poor software support for Whisper models on NPUs
- Would require significant model conversion/quantization work
- Performance would likely be worse than GPU for your use case
- Examples: Intel's AI Boost, Qualcomm's Hexagon NPU (laptop/mobile chips)

**TPU (Tensor Processing Unit):**
- ❌ Not practical for consumer desktop use
- TPUs are Google's proprietary accelerators (Cloud TPU or Google Edge TPU)
- Edge TPUs are underpowered for real-time STT of Whisper-scale models
- Cloud TPUs are rental-only and prohibitively expensive for continuous STT use
- Limited software compatibility with Whisper ecosystem

### Special Consideration: If You Must Keep Current AMD GPU

If you're not ready for a full build and want to add a second GPU with your current setup, here's what to consider:

**Prerequisites:**
- Verify your motherboard has a second PCIe x16 slot (or at least x8)
- Ensure adequate PCIe lane allocation from CPU
- Check power supply capacity (likely need 850W+ for dual-GPU)
- Verify case airflow can handle additional heat

**Budget Second GPU Options (~$300-400):**
- **NVIDIA RTX 3060 12GB** (used market): Good VRAM for STT, reasonable performance
- **NVIDIA RTX 4060 8GB** (new): Newer architecture but limited VRAM

**Setup Configuration:**
- AMD GPU: Primary display output
- NVIDIA GPU: Dedicated to CUDA compute (STT, AI workloads)
- Use `CUDA_VISIBLE_DEVICES` environment variable to explicitly route workloads to NVIDIA GPU
- Set display manager to use AMD GPU to avoid NVIDIA driver overhead on display tasks

### Practical Implementation for STT Workloads

Regardless of which option you choose, here's how to optimize for STT:

**Software Stack:**
1. **faster-whisper** (recommended): CTranslate2-based, highly optimized, low VRAM usage
   - large-v3 model runs well on 8GB VRAM
   - 2-3x faster than OpenAI's Whisper implementation
   - Significantly lower memory footprint

2. **whisper.cpp**: If you want CPU fallback option
   - Uses CUDA when available
   - Excellent quantized model support

3. **Hugging Face Transformers**: If you need fine-tuning capabilities
   - More VRAM intensive
   - Slower inference than faster-whisper

**VRAM Requirements by Whisper Model:**
| Model Size | Minimum VRAM (faster-whisper) | Recommended VRAM |
|------------|-------------------------------|------------------|
| tiny       | 1GB                           | 2GB              |
| base       | 1GB                           | 2GB              |
| small      | 2GB                           | 4GB              |
| medium     | 4GB                           | 6GB              |
| large-v2/v3| 6GB                           | 10GB             |

**Real-Time STT Performance Targets:**
- For real-time transcription (1x speed or faster), you want 4GB+ VRAM
- For comfortable headroom with large-v3 and parallel processing, 12GB+ VRAM is ideal

### Timeline Recommendation

**Immediate (if needed):**
- Continue using your AMD GPU with ROCm for STT
- Consider `whisper.cpp` with CPU offloading if ROCm is problematic

**Short-term (3-6 months):**
- If STT performance is blocking your workflow, consider a used RTX 3060 12GB as a second GPU stopgap
- Only if dual-GPU setup is viable on your current system

**Next build/major upgrade (12-24 months):**
- Replace with single NVIDIA RTX 4070 Ti Super 16GB or equivalent next-gen card
- This will serve you better than any dual-GPU configuration

### Additional Considerations

**Power Efficiency:**
Modern NVIDIA GPUs have excellent idle power management. If you're running STT intermittently throughout the day (not 24/7), the GPU will mostly idle at 10-30W, spiking only during active transcription.

**Future-Proofing:**
STT models are trending toward larger, more capable architectures (Whisper large-v3, Distil-Whisper, Canary). Having 16GB VRAM provides headroom for these developments.

**Local LLM Consideration:**
If you expand your local LLM usage, 16GB VRAM enables:
- 13B parameter models at good speed (Q4 quantization)
- 7B parameter models at full precision
- Simultaneous STT + small LLM workloads

## Summary

**Best Path Forward:**
1. **Next build**: Single NVIDIA RTX 4070 Ti Super 16GB (or equivalent)
2. **Current system**: Continue with AMD + ROCm or consider budget second NVIDIA GPU only if current performance is blocking work
3. **Hardware type**: GPU only—NPUs and TPUs are not suitable for desktop STT workloads

The single powerful NVIDIA GPU approach provides the best balance of performance, flexibility, power efficiency, and system simplicity for your STT-focused workload.

---

*Generated by Claude Code (Anthropic) - Please validate recommendations against your specific motherboard, PSU, and case specifications before purchasing.*
