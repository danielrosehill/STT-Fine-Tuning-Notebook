# Audio-LLM Guided Transcription: Beyond Pure ASR

## Question
What is the technology called when you can provide audio + text prompts (like Gemini) for guided transcription (e.g., "diarize this as A and B")? Are there open-source, fine-tunable models that combine ASR with LLM-style prompt following for audio tasks?

## Answer

Excellent question! You've identified an emerging and powerful paradigm. Let's break this down.

---

## What Is This Technology Called?

**The general term is: "Audio-Language Models" (Audio-LLMs) or "Speech Large Language Models" (Speech-LLMs).**

More specifically, when combining:
- **Audio input** (speech, environmental sounds)
- **Text prompts** (instructions, context)
- **Text output** (transcription, diarization, summary, etc.)

...this is often called:

### **1. "Instruction-Following Speech Models"**
Models trained to follow text instructions about audio processing.

### **2. "Audio-Conditioned Language Models"**
LLMs that take audio as input alongside text prompts.

### **3. "Multimodal Audio-Text Models"**
Models that jointly understand audio and text modalities.

### **4. "Prompt-Guided Transcription"**
Transcription steered by natural language instructions (your use case).

**Gemini's audio capability is an example of #2: an audio-conditioned multimodal LLM.**

**There's no single universally-accepted name yet** (the field is young), but "Audio-Language Models" (Audio-LLMs) is gaining traction.

---

## How Gemini Works (vs. Whisper)

### **Whisper: Pure ASR**

**Architecture:**
```
Audio → Encoder → Decoder → Transcription
```

**Capabilities:**
- Transcribe audio to text
- Detect language
- Add timestamps
- (That's it—no customization beyond model parameters)

**Limitations:**
- Can't follow instructions
- Can't do speaker diarization
- Can't format output (e.g., "format as Q&A")
- Can't incorporate context (e.g., "this is a medical call")

---

### **Gemini (Audio-LLM): Multimodal Instruction-Following**

**Architecture:**
```
Audio → Audio Encoder → Multimodal Transformer (LLM) ← Text Prompt
                                  ↓
                            Text Output
```

**Capabilities:**
- Transcribe audio
- **Follow text instructions** ("diarize as A and B", "summarize this call")
- **Context-aware** ("this is a phone call between a doctor and patient")
- **Output formatting** ("format as JSON", "use markdown")
- **Reasoning** ("identify the main complaint", "what was decided?")

**Key Difference:**
Gemini treats audio as **another input modality to an LLM**, not as a standalone ASR task.

**What Enables This:**
1. **Audio encoder** converts audio → embeddings (like text tokens)
2. **LLM** processes both audio embeddings + text prompt together
3. **Decoder** generates text output following instructions

**Example:**
```
Input (Audio): [30s phone call recording]
Input (Text Prompt): "Transcribe this call. The participants are Alice (caller) and Bob (support agent). Format as Q&A."

Output:
Q (Alice): Hi, I'm having trouble with my account.
A (Bob): Sure, I can help with that. What's the issue?
Q (Alice): I can't log in.
...
```

**Whisper cannot do this** (it would just transcribe everything without structure or speaker labels).

---

## Open-Source Models with Audio-LLM Capabilities

**Good news: This field is exploding in 2023-2024.** Here are the major open-source options:

---

### **1. Qwen-Audio (Alibaba) ⭐ Recommended**

**What it is:**
- Large-scale audio-language pretrained model
- Understands 30+ audio tasks (ASR, diarization, audio captioning, etc.)
- Follows natural language instructions
- **Open-source and fine-tunable**

**Hugging Face:**
[https://huggingface.co/Qwen/Qwen-Audio](https://huggingface.co/Qwen/Qwen-Audio)

**Paper:**
"Qwen-Audio: Advancing Universal Audio Understanding via Unified Large-Scale Audio-Language Models" (Nov 2023)

**Capabilities:**
```
Prompt: "Transcribe this audio and identify the speakers."
Prompt: "Summarize the main points of this meeting."
Prompt: "What sounds do you hear in this audio?"
Prompt: "Translate this Spanish speech to English."
```

**Architecture:**
- Audio encoder (Whisper-like)
- Qwen LLM (7B or 13B parameters)
- Multimodal adapter

**Fine-tuning:**
```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-Audio")
# Fine-tune with your audio-text pairs + instructions
```

**Why this is good for you:**
- Open-source (Apache 2.0 license)
- Fine-tunable
- Supports custom instructions
- Active development

---

### **2. SpeechGPT (Fudan University)**

**What it is:**
- Enables LLMs to process speech directly
- Can follow instructions for transcription, diarization, etc.
- Uses discrete audio tokens

**Hugging Face:**
[https://huggingface.co/fnlp/SpeechGPT](https://huggingface.co/fnlp/SpeechGPT)

**Paper:**
"SpeechGPT: Empowering Large Language Models with Intrinsic Cross-Modal Conversational Abilities" (May 2023)

**Architecture:**
```
Audio → HuBERT encoder → Discrete tokens → LLM → Text output
```

**Use Case:**
- Conversational speech understanding
- Instruction-following transcription

**Limitation:**
- Smaller scale than Qwen-Audio
- Less mature ecosystem

---

### **3. Whisper + LLM Pipeline (DIY Approach)**

**What it is:**
- Combine Whisper (ASR) with an LLM (Llama, Mistral, etc.) in a pipeline
- Whisper transcribes, LLM processes instructions

**Architecture:**
```
Audio → Whisper → Raw transcription → LLM → Formatted output
```

**Example:**
```python
from faster_whisper import WhisperModel
from transformers import pipeline

# Step 1: Transcribe
whisper = WhisperModel("medium")
segments, info = whisper.transcribe("audio.wav")
raw_transcription = " ".join([seg.text for seg in segments])

# Step 2: Apply instructions via LLM
llm = pipeline("text-generation", model="meta-llama/Llama-2-7b-chat-hf")

prompt = f"""
You are a transcription assistant.

Audio transcription:
{raw_transcription}

Instructions: This is a phone call between Alice (caller) and Bob (agent).
Diarize the transcription and format as Q&A.

Output:
"""

result = llm(prompt, max_new_tokens=512)
print(result[0]["generated_text"])
```

**Pros:**
- ✅ Works today (no waiting for models)
- ✅ Highly customizable (swap components)
- ✅ Can use your fine-tuned Whisper

**Cons:**
- ❌ Two-stage (not end-to-end)
- ❌ Slower (two inference passes)
- ❌ Whisper doesn't "know" about instructions during transcription

**This is a practical workaround until unified models mature.**

---

### **4. LTU (Listening-and-Talking Understanding) Models**

**What it is:**
- Recent research on unified speech-text models
- Examples: SALMONN, LLaSM, etc.

**SALMONN (ByteDance):**
[https://github.com/bytedance/SALMONN](https://github.com/bytedance/SALMONN)

**Paper:**
"SALMONN: Towards Generic Hearing Abilities for Large Language Models" (Oct 2023)

**Capabilities:**
- Speech recognition
- Audio captioning (describe sounds)
- Speech emotion recognition
- Music understanding
- Instruction-following

**Status:**
- Research code (less production-ready than Qwen-Audio)
- Demonstrates feasibility of unified audio-LLMs

---

### **5. Gemini-Style Open Alternatives (Future)**

**What's coming:**
- **OpenAI Whisper v4** (rumored to have instruction-following)
- **Meta's SeamlessM4T v3** (multimodal, may add instructions)
- **Google's USM-v2** (Universal Speech Model, not yet released)

**Current state:** Gemini's audio capabilities are proprietary—no direct open-source equivalent yet.

---

## Comparison Table

| Model | Open-Source | Fine-Tunable | Instruction-Following | Maturity | Best For |
|-------|------------|--------------|----------------------|----------|----------|
| **Qwen-Audio** | ✅ | ✅ | ✅ | High | Production use, fine-tuning |
| **SpeechGPT** | ✅ | ✅ | ✅ | Medium | Research, experimentation |
| **Whisper + LLM** | ✅ | ✅ (separately) | ✅ | High | Immediate practical use |
| **SALMONN** | ✅ | ⚠️ (complex) | ✅ | Low | Research, demos |
| **Gemini** | ❌ | ❌ | ✅ | High | Production (if cost OK) |

---

## Fine-Tuning an Audio-LLM

### **Qwen-Audio Fine-Tuning Example**

**Goal:** Fine-tune for your specific use case (e.g., meeting transcription with diarization).

**Data Format:**
```json
[
  {
    "audio": "path/to/audio1.wav",
    "prompt": "Transcribe this meeting. Participants are Alice, Bob, and Charlie. Format with speaker labels.",
    "response": "Alice: Let's start with the budget.\nBob: I think we need to cut costs.\n..."
  },
  {
    "audio": "path/to/audio2.wav",
    "prompt": "Summarize the key decisions from this call.",
    "response": "1. Approved budget of $50k\n2. Next meeting on Friday\n..."
  }
]
```

**Fine-Tuning Code (Conceptual):**
```python
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-Audio")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen-Audio")

# Prepare dataset
# ... (load audio-prompt-response triples)

# Fine-tune
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
)

trainer.train()
```

**Challenges:**
- **Data collection**: Need audio + instruction + desired output triples
- **Compute**: Audio-LLMs are large (7B-13B params) → need GPUs
- **Labeling**: Creating instruction-following data is labor-intensive

---

## Practical Recommendations

### **Immediate Solution (Today):**

**Use Whisper + LLM Pipeline**

1. Fine-tune Whisper for your audio (if needed)
2. Use a local LLM (Llama 2, Mistral via Ollama) for post-processing
3. Prompt engineering for diarization/formatting

**Pros:**
- Works now
- Flexible
- Can run locally (privacy)

**Example:**
```python
# Transcribe
whisper_output = whisper.transcribe("call.wav")

# Process with LLM
llm_prompt = f"""
Transcription: {whisper_output}

Task: This is a support call. The caller is the customer, the agent is support.
Diarize and format as Q&A.
"""

formatted_output = llm(llm_prompt)
```

---

### **Short-Term (3-6 Months):**

**Experiment with Qwen-Audio**

1. Test Qwen-Audio on your audio samples
2. Evaluate instruction-following quality
3. If promising, fine-tune on your specific tasks

**Why:**
- Most mature open-source Audio-LLM
- Active development
- Fine-tunable

---

### **Long-Term (1-2 Years):**

**Wait for Specialized Models**

The field is moving fast. Within 1-2 years, expect:
- More open-source Audio-LLMs
- Better fine-tuning tools
- Purpose-built models for transcription + instructions

---

## Why Isn't This Standard Yet?

**Good question. Several reasons:**

### **1. Technical Complexity**

Combining ASR + LLM requires:
- Large-scale multimodal pretraining (expensive)
- Careful architecture design (modality fusion)
- Instruction-following data (labor-intensive)

### **2. Compute Requirements**

Audio-LLMs are **huge**:
- Qwen-Audio: 7B-13B parameters
- Gemini: Likely 100B+ parameters

**Training/fine-tuning needs serious compute.**

### **3. Data Scarcity**

Unlike text LLMs (trained on internet text), Audio-LLMs need:
- Audio recordings + transcriptions + instructions + desired outputs
- This data barely exists at scale

### **4. Commercial Incentives**

Google (Gemini), OpenAI (GPT-4 multimodal) have invested heavily but kept models proprietary.

Open-source is catching up, but slowly.

---

## Does It Have a Name? (Terminology Summary)

**The capability you're describing doesn't have ONE universally accepted name, but here are the terms used:**

| Term | Usage |
|------|-------|
| **Audio-Language Models (Audio-LLMs)** | Most common in research |
| **Speech Large Language Models (Speech-LLMs)** | Emphasizes speech focus |
| **Instruction-Following Transcription** | Task-specific description |
| **Multimodal Audio Understanding** | Broader term (includes non-speech audio) |
| **Prompt-Guided Speech Processing** | Emphasizes prompting aspect |

**If you need to search for papers/models, use "Audio-Language Models" or "Audio-LLM".**

---

## Future Outlook

**This is an active research area. Expect rapid progress:**

**2024:**
- More open-source Audio-LLMs (Qwen-Audio scale)
- Better fine-tuning recipes
- Specialized models (e.g., meeting transcription)

**2025:**
- On-device Audio-LLMs (mobile-optimized)
- Real-time instruction-following transcription
- Fine-tuning accessible to individuals (not just labs)

**2026:**
- Whisper-level ubiquity for Audio-LLMs
- Standardized fine-tuning pipelines
- Integration into consumer apps

**The trajectory is clear: Audio-LLMs will become standard.**

---

## Summary

| Question | Answer |
|----------|--------|
| **What's it called?** | Audio-Language Models (Audio-LLMs) or Instruction-Following Speech Models |
| **Open-source equivalent?** | **Qwen-Audio** (best option), SpeechGPT, or Whisper + LLM pipeline |
| **Fine-tunable?** | Yes—Qwen-Audio is fine-tunable, but requires substantial compute |
| **Why not standard?** | Technical complexity, compute cost, data scarcity, commercial incentives |
| **Recommended approach?** | **Near-term:** Whisper + LLM pipeline. **Future:** Fine-tune Qwen-Audio. |

**Gemini's audio capabilities represent the future of transcription—combining ASR with LLM reasoning. Open-source is catching up, with Qwen-Audio as the current leader.**

**For your use case (diarization, formatting, context-aware transcription), start with a Whisper + LLM pipeline today, and monitor Qwen-Audio for future fine-tuning.**

---

**Note**: This explanation was generated by Claude Code (claude-sonnet-4-5) for Daniel Rosehill's STT Fine-Tuning Notebook. Audio-Language Models are a rapidly evolving field—check Hugging Face's audio tasks page and arXiv for latest models. The Whisper + LLM pipeline is a practical interim solution until unified fine-tunable Audio-LLMs mature. For production use, evaluate Qwen-Audio's instruction-following quality on your specific audio domain before committing to fine-tuning.
