# Deployment Options for Custom ASR Models: Serverless, Self-Hosted, and Cost Analysis

## Question Summary

Daniel is exploring deployment options for fine-tuned or custom ASR models, particularly for individual/solo users. He's found Replicate for serverless but is concerned about costs for 24/7 operation. He wants to understand the full spectrum of deployment options and cost implications for both serverless and always-on (local or cloud) deployments.

## Answer

You're right that this is somewhat niche territory for individual users, but it's increasingly relevant as more people fine-tune their own ASR models. Let me break down the deployment landscape comprehensively.

### Serverless Inference Options

**1. Replicate**
- **What you found:** Yes, Replicate is the most prominent serverless option
- **Pricing:** Pay-per-second of inference time
  - Typically $0.0005-0.0025 per second depending on hardware (CPU vs GPU)
  - For Whisper-sized models on GPU: ~$0.001/second
- **Cost Example:**
  - 1 hour of audio processing ≈ 6 minutes inference time (10x realtime)
  - Cost: ~$0.36 per hour of audio transcribed
  - For intermittent use (say, 5 hours of audio/month): ~$1.80/month
- **Pros:** Zero setup, scales automatically, no idle costs
- **Cons:** Cold start latency (2-15 seconds), per-request costs add up quickly for heavy use

**2. Hugging Face Inference Endpoints**
- **Overview:** Serverless inference for models hosted on HuggingFace
- **Pricing Tiers:**
  - Free tier: Limited requests, public models only
  - Paid: $0.06/hour (CPU) to $1.50/hour (GPU) when running
  - Auto-scales to zero when idle (no requests for 15 minutes)
- **Cost Example:**
  - If processing requests sporadically (active 2 hours/day): ~$90/month for GPU instance
  - Better than 24/7 ($1,080/month) but still pricey for continuous use
- **Pros:** Good HuggingFace integration, custom model support
- **Cons:** Not truly serverless (charges per hour active, not per request)

**3. Modal**
- **Overview:** Python-native serverless compute platform
- **Pricing:** Pay per GPU-second
  - A10G GPU: ~$0.0010/second
  - T4 GPU: ~$0.0005/second
- **Cost Example:**
  - Processing 10 hours of audio/day (realtime inference): ~$36/month on T4
- **Pros:** Excellent developer experience, true pay-per-use, fast cold starts
- **Cons:** Requires some Python infrastructure code setup

**4. Banana.dev (now Tonic.ai)**
- **Overview:** Serverless GPU inference platform
- **Pricing:** Similar to Replicate (~$0.0008/second for GPU)
- **Status:** Rebranded/transitioning, may be less stable option currently
- **Pros:** Previously popular for ASR deployments
- **Cons:** Platform uncertainty after rebrand

**5. Baseten**
- **Overview:** ML inference platform with serverless and dedicated options
- **Pricing:** Custom pricing, typically $0.0005-0.0015/second
- **Pros:** Good performance, handles custom models well
- **Cons:** Less transparent pricing, requires contact for details

**6. AWS Lambda + GPU (Emerging)**
- **Overview:** AWS is rolling out Lambda support for GPUs
- **Status:** Limited availability, not yet widely practical for ASR
- **Future Potential:** Could become very cost-effective for sporadic use

### 24/7 Self-Hosted Options

If you want always-available inference (locally or cloud), here are the realistic options:

#### Local Deployment (Home Server)

**Option A: Dedicated Machine**
- **Hardware Requirements for Whisper:**
  - CPU-only: Modern 8-core CPU (i7/Ryzen 7), 16GB RAM
  - GPU: RTX 3060 (12GB VRAM) or better for comfortable performance
  - Storage: 50-100GB SSD for models and OS

- **Costs:**
  - **Initial:** $800-1,500 for dedicated machine (or use existing hardware)
  - **Electricity:**
    - Idle GPU server: ~100-150W = ~$10-15/month (at $0.12/kWh)
    - Under load: ~250W = ~$25/month
    - Annual: ~$120-300/year in electricity

- **Networking:**
  - Port forwarding: Free (security risk - need VPN)
  - Cloudflare Tunnel: Free (recommended, secure)
  - Tailscale/ZeroTier: Free for personal use (private network)

**Option B: Your Existing Hardware**
- You have AMD RX 7700 XT with ROCm - excellent for ASR!
- **Costs:**
  - Electricity only (~$10-20/month if running 24/7)
  - Wear and tear on GPU (negligible for inference)
- **Pros:** No additional hardware cost, full control
- **Cons:** Home network dependency, potential security exposure

**Recommended Setup for Local 24/7:**
```bash
# Use Docker container with Faster-Whisper or whisper.cpp
# Expose via Cloudflare Tunnel (free, secure)
# Optionally: FastAPI wrapper for REST API
```

#### Cloud VPS Deployment

**Option 1: CPU-Only VPS (Budget)**
- **Providers:** Hetzner, OVH, DigitalOcean, Linode
- **Recommended Specs:** 8-core CPU, 16GB RAM
- **Costs:**
  - Hetzner CCX33: €32.69/month (~$35/month) - 8 vCores, 32GB RAM
  - DigitalOcean: $48/month - 8 vCPU, 16GB RAM
- **Performance:**
  - Realtime or slightly faster for Whisper-large
  - Acceptable for most use cases
- **Pros:** Predictable costs, reliable, no home network dependency
- **Cons:** Slower than GPU inference

**Option 2: GPU Cloud Instances**
- **RunPod:**
  - RTX A4000 (16GB): ~$0.34/hour = ~$245/month for 24/7
  - RTX 4090 (24GB): ~$0.69/hour = ~$497/month for 24/7
- **Vast.ai:**
  - RTX 3060 (12GB): ~$0.15/hour = ~$108/month for 24/7
  - Highly variable pricing (spot market)
- **Lambda Labs:**
  - A10 GPU: $0.60/hour = ~$432/month
- **Google Cloud / AWS / Azure:**
  - Much more expensive (~$0.70-2.00/hour for GPU instances)
  - GCP T4: ~$0.35/hour = ~$252/month

**Option 3: Hybrid Approach (Spot Instances)**
- **Vast.ai Spot Instances:**
  - Bid on idle GPU capacity
  - Can get RTX 3080 for ~$0.10/hour = ~$72/month
  - Risk: Instance can be reclaimed (need auto-restart logic)
- **AWS Spot / GCP Preemptible:**
  - 60-80% cheaper than on-demand
  - Requires interruption handling

### Cost Comparison Summary

| Deployment Option | Setup Cost | Monthly Cost (Light Use) | Monthly Cost (Heavy/24-7) |
|------------------|------------|-------------------------|--------------------------|
| **Replicate** | $0 | $5-20 | $300-1,000+ |
| **Modal** | $0 | $10-50 | $200-500 |
| **HF Inference Endpoints** | $0 | $30-100 | $1,080 (GPU always-on) |
| **Local (Existing HW)** | $0 | $10-20 | $15-30 |
| **Local (New Server)** | $800-1,500 | $10-20 | $15-30 |
| **CPU VPS (Hetzner)** | $0 | $35 | $35 |
| **GPU Cloud (Vast.ai)** | $0 | $108+ | $108-500 |
| **GPU Cloud (RunPod)** | $0 | $245+ | $245-500 |

### Recommendations Based on Use Cases

**Scenario 1: Occasional Personal Use (< 10 hours audio/month)**
- **Best Option:** Replicate or Modal
- **Reasoning:** Zero setup, only pay for what you use
- **Cost:** $5-20/month

**Scenario 2: Regular Personal Use (Daily, ~2-4 hours audio/day)**
- **Best Option:** Local deployment on your existing hardware
- **Reasoning:** Electricity costs less than serverless, full control
- **Cost:** ~$15-25/month (electricity only)
- **Setup:** Docker + Faster-Whisper + Cloudflare Tunnel

**Scenario 3: Service/App Development (Public API)**
- **Best Option:** CPU VPS (Hetzner) with queue system
- **Reasoning:** Predictable costs, good performance, professional reliability
- **Cost:** ~$35-50/month
- **Alternative:** Modal for burst capacity + CPU VPS for base load

**Scenario 4: High-Volume Production (100+ hours audio/day)**
- **Best Option:** Dedicated GPU cloud (RunPod/Vast.ai) or multiple CPU VPS
- **Reasoning:** Cost-effective at scale
- **Cost:** $250-500/month

### Your Specific Situation (Solo User, Custom Model)

Given your setup (AMD GPU with ROCm), here's what I'd recommend:

**Option A: Local 24/7 (Recommended)**
```bash
# Benefits:
- Zero additional hardware cost (you have RX 7700 XT)
- Whisper runs well on ROCm (HSA_OVERRIDE_GFX_VERSION=11.0.1)
- Can expose via Cloudflare Tunnel (free, secure, no port forwarding)
- Total cost: ~$15-20/month in electricity

# Setup:
1. Docker container with whisper.cpp or faster-whisper
2. FastAPI wrapper for REST API
3. Cloudflare Tunnel for secure external access
4. Optional: Nginx reverse proxy for API management
```

**Option B: Hybrid (Local + Serverless Fallback)**
```bash
# Use local inference as primary
# Fall back to Modal/Replicate when local is unavailable
# Best of both worlds: cheap + reliable
```

**Option C: CPU VPS (If You Don't Want Local Running 24/7)**
```bash
# Hetzner CCX33 (€32.69/month)
# Install faster-whisper with CPU optimization
# Performance: Near-realtime for Whisper-large
# No home network dependency
```

### Practical Cost Calculation Examples

**Scenario: Processing 5 hours of audio per day**

| Option | Daily Cost | Monthly Cost | Notes |
|--------|-----------|--------------|-------|
| Replicate (10x RT) | $1.80 | $54 | Quick bursts |
| Modal (realtime) | $1.20 | $36 | Python-friendly |
| Local (Your GPU) | $0.50 | $15 | Electricity only |
| Hetzner CPU VPS | $1.10 | $33 | Always available |
| Vast.ai GPU (spot) | $2.40 | $72 | Fast processing |

**Verdict for Solo User:** Local deployment on your existing hardware is by far the most cost-effective for 24/7 availability.

### Exposure/Security Considerations

If running locally and exposing to internet:

1. **Never expose ports directly** - major security risk
2. **Use Cloudflare Tunnel** (recommended):
   ```bash
   # Free, secure, no port forwarding needed
   cloudflared tunnel create my-asr
   # Creates encrypted tunnel from your server to Cloudflare edge
   ```
3. **Alternative: Tailscale** - Private mesh network (free for personal use)
4. **API Authentication:** Always implement API keys/tokens
5. **Rate Limiting:** Prevent abuse with request limits
6. **HTTPS Only:** Cloudflare provides this automatically

### Advanced Options for Solo Users

**Option: Fly.io**
- Deploy containers globally
- Pay per request (scales to zero)
- ~$0.0008/sec GPU or $0.00025/sec CPU
- Good middle ground between VPS and serverless

**Option: Railway.app**
- $5/month base + usage
- Good for hobby projects
- No GPU support (CPU only)

**Option: Self-hosted on Oracle Cloud Free Tier**
- 4 ARM cores, 24GB RAM - completely free forever
- Can run CPU inference
- Performance: Slower than x86, but usable for Whisper-base/small
- Great for experimentation

### Final Recommendation for You

Based on your setup and likely use pattern:

1. **Start with local deployment** on your RX 7700 XT
   - Use Docker + faster-whisper with ROCm
   - Expose via Cloudflare Tunnel
   - Cost: ~$15-20/month electricity
   - Benefit: Full control, lowest cost, instant inference

2. **Add Modal as backup** for when local is down
   - Minimal cost if rarely used
   - Python-friendly deployment
   - Automatic fallback logic in your client

3. **If you outgrow local:** Migrate to Hetzner CPU VPS
   - Still cheaper than GPU cloud options
   - Professional reliability
   - ~$35/month predictable cost

### Code Example: Local Deployment with Cloudflare Tunnel

```bash
# 1. Install Cloudflare Tunnel
wget -q https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64.deb
sudo dpkg -i cloudflared-linux-amd64.deb

# 2. Authenticate
cloudflared tunnel login

# 3. Create tunnel
cloudflared tunnel create my-asr-api

# 4. Configure tunnel (create config.yml)
cat > ~/.cloudflared/config.yml << EOF
tunnel: my-asr-api
credentials-file: /home/daniel/.cloudflared/<tunnel-id>.json

ingress:
  - hostname: asr.yourdomain.com
    service: http://localhost:8000
  - service: http_status:404
EOF

# 5. Run as service
sudo cloudflared service install
sudo systemctl start cloudflared

# Your ASR API is now accessible at https://asr.yourdomain.com
# Fully encrypted, no port forwarding, free!
```

### Conclusion

For a solo user with a custom ASR model:
- **Serverless options exist beyond Replicate** (Modal, HF Inference Endpoints, Baseten, Fly.io)
- **Local 24/7 deployment is surprisingly affordable** (~$15-30/month) using existing hardware
- **Cloud VPS CPU instances** are the sweet spot for "always-on" without managing local hardware (~$35/month)
- **GPU cloud is expensive for 24/7** ($250-500/month) but reasonable for burst usage
- **Your specific situation:** Local deployment on RX 7700 XT + Cloudflare Tunnel is the optimal approach

The niche you're in (solo user with custom ASR) actually has more options than you might think, and costs can be quite reasonable with the right architecture!

---

*Note: This response was generated by Claude Code as part of Daniel's STT Fine-Tuning Notebook project. Pricing information is approximate and based on 2025 rates. Always verify current pricing with service providers.*
