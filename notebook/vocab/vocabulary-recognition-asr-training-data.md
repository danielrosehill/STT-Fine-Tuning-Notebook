# Vocabulary Recognition in ASR: Why Some Words Work and Others Don't

## Question
Why does Whisper recognize some Hebrew words (like "Shabbat") but not others (like "Macolet")? Does ASR work at the word level or phonetic level, and how does training data determine which words make it into the model?

## Answer

Excellent question that touches on fundamental ASR mechanics. Let's break down how vocabulary recognition works in models like Whisper.

---

## How ASR Models Handle Vocabulary

### **The Short Answer:**

**ASR models work at the subword/phoneme level, not the word level.**

**Whether a word is recognized depends on:**
1. **Frequency in training data** (how often it appeared)
2. **Phonetic similarity to known words**
3. **Tokenization strategy** (how the model breaks down sounds)
4. **Language mode** (English vs. Hebrew vs. multilingual)

**Your observation is spot-on:** "Shabbat" is common enough in English-language audio (especially in North America/UK content) to be well-represented, while "Macolet" (מכולת) is Israeli-specific and rare in international English audio.

---

## How Whisper (and Modern ASR) Works

### **Phonetic Level → Subword Tokens → Words**

**Architecture Overview:**
```
Audio → Mel-spectrogram → Encoder → Decoder → Subword tokens → Words
```

**Key Insight: Whisper doesn't have a "vocabulary" like a dictionary.**

Instead:
1. **Audio encoding**: Convert sound waves → spectral features
2. **Sequence modeling**: Encoder learns phonetic patterns
3. **Token prediction**: Decoder predicts subword tokens (BPE - Byte-Pair Encoding)
4. **Token → Text**: Subword tokens combine into words

---

### **Byte-Pair Encoding (BPE) Tokenization**

**What is BPE?**
- Breaks words into frequent subword units
- Common subwords become single tokens
- Rare words are split into smaller pieces

**Example:**
```
Common word: "hello" → [hello]  (single token)
Rare word: "Macolet" → [Mac][ol][et]  (multiple tokens)
```

**Whisper's tokenizer has ~50,000 tokens**:
- Common English words: Single tokens
- Common names/terms: Single tokens
- Rare words: Split into subwords

**Why This Matters:**
If "Shabbat" appears frequently in training data, it becomes a **single token** in Whisper's vocabulary. If "Macolet" doesn't, it must be constructed from **phonetic subword tokens**—and this is where errors happen.

---

## Why "Shabbat" Works But "Macolet" Doesn't

### **Case Study: "Shabbat"**

**Frequency in Training Data:**
- Whisper trained on 680,000 hours of audio
- Sources include:
  - YouTube subtitles (religious/cultural content)
  - Podcasts (Jewish topics, interfaith discussions)
  - TV shows/movies (Jewish characters, cultural references)
  - News (stories about Israel, Judaism)

**"Shabbat" appears in:**
- Religious content (sermons, lectures)
- Cultural programming (food shows, travel vlogs)
- Mainstream media (discussions of Jewish holidays)

**Result:**
- **High frequency** → BPE tokenizer creates a token `[Shabbat]`
- Whisper learns acoustic patterns for "Shabbat"
- Decoder predicts `[Shabbat]` token confidently

**Transcription: ✅ "Shabbat"** (correct)

---

### **Case Study: "Macolet" (מכולת)**

**Frequency in Training Data:**
- "Macolet" (or "Makolet") is **Israeli-specific slang**
- Rarely used in English-language media
- Not commonly in international English audio
- Whisper's training data skews toward:
  - North American English
  - British English
  - International content (but not hyper-local terms)

**Result:**
- **Low/zero frequency** → No `[Macolet]` token
- Whisper must construct from phonetic subwords
- Decoder guesses: `[Mac][ol][et]` or similar
- Acoustically similar words interfere (e.g., "makeup lot", "mackerel", "macho let")

**Transcription: ❌ "Makeup lot" / "Maco late" / gibberish** (incorrect)

---

## The Phonetic Level: Why Errors Happen

### **How Whisper "Hears" Unknown Words**

When you say "Macolet" (`/ma-ko-let/`):

1. **Acoustic encoding**: Whisper converts sound → spectral features
   - Recognizes phonemes: `/m/`, `/a/`, `/k/`, `/o/`, `/l/`, `/e/`, `/t/`

2. **Decoder prediction**: Tries to match phonemes to known tokens
   - Searches for tokens that match `/ma-ko-let/` acoustically
   - Finds partial matches:
     - "Mac" (common prefix: Macintosh, McDonald's)
     - "lot" (common word)
     - "late" (common word)

3. **Decoder outputs best guess**:
   - "Mac lot" (if it parses as two words)
   - "Macolate" (if it tries to keep as one word)
   - "Macaulay" (if it finds a similar name)

**The problem:** Without seeing "Macolet" in training, Whisper has no prior to favor the correct spelling.

---

## Training Data Determines Recognition

### **The Rule:**

**If a word appears frequently enough in training data, it will be recognized reliably.**

**"Frequently enough" depends on:**
- **Raw count**: How many times it appears
- **Acoustic variability**: Different speakers, accents, contexts
- **Context**: Surrounding words that help disambiguation

**Thresholds (Rough Estimates):**
```
>10,000 occurrences: Very likely to be a single token → reliable recognition
1,000-10,000: May be a token or common subword sequence → good recognition
100-1,000: Likely subword split → moderate recognition (context-dependent)
<100: Definitely subword split → poor recognition (often fails)
```

**"Shabbat"**: Likely 10,000+ occurrences in Whisper's training data
**"Macolet"**: Likely <10 occurrences (if any)

---

## Language Mode and Code-Switching

### **Your Use Case: English + Hebrew Words**

**Whisper's multilingual model has language detection:**
```
Audio → Language detection → Decoder (language-specific mode)
```

**What happens when you speak English with Hebrew words:**

**Option 1: Whisper detects English**
- Decoder uses English tokens
- Hebrew words must map to English phonetics
- Result: Hebrew words often mis-transcribed

**Option 2: Whisper detects Hebrew**
- Decoder uses Hebrew tokens
- English words must map to Hebrew phonetics
- Result: English words may be transliterated incorrectly

**Option 3: Whisper code-switches (rare)**
- Decoder flips between English and Hebrew tokens
- Can work if the model learned this pattern
- But Whisper wasn't explicitly trained for code-switching

**Your experience:**
- When you say "I need to go to the Macolet," Whisper stays in English mode
- "Macolet" has no English token → phonetic guessing → error

---

## Fine-Tuning to Fix This

### **How Fine-Tuning Helps:**

**Your fine-tuning data:**
```
Audio: "I'm going to the Macolet to buy milk"
Text: "I'm going to the Macolet to buy milk"
```

**What the model learns:**
1. **Phonetic pattern**: `/ma-ko-let/` → "Macolet" (consistent mapping)
2. **Context**: "Macolet" appears after "the" (like "the store", "the shop")
3. **Frequency**: If you provide 50-100 examples, "Macolet" becomes a learned pattern

**Post-fine-tuning:**
- Whisper's decoder learns to output "Macolet" when it hears `/ma-ko-let/`
- Even if "Macolet" isn't a single token, the model learns the subword sequence
- Context helps (e.g., "going to the [Macolet]" vs. "Mac" + "lot")

**Result: ✅ Reliable transcription of "Macolet"**

---

## Vocabulary Expansion Strategies

### **1. Fine-Tuning (Your Best Option)**

**Data collection:**
- Record yourself using Hebrew words in English sentences
- Transcribe with the correct spelling (e.g., "Macolet")
- 2-5 hours of audio with these words

**Fine-tuning:**
- Train Whisper on your data
- Model learns your code-switching patterns
- Hebrew words become consistently transcribed

**Benefit:**
- Works for ALL your Hebrew words (Macolet, misrad, etc.)
- Learns your pronunciation patterns

---

### **2. Custom Tokenizer (Advanced, Not Recommended)**

**Concept:**
- Retrain Whisper's BPE tokenizer with your vocabulary
- Add "Macolet", "misrad", etc. as explicit tokens

**Problems:**
- Requires retraining the entire model (not just fine-tuning)
- Extremely compute-intensive
- Breaks compatibility with standard Whisper

**Not worth it** for your use case.

---

### **3. Post-Processing (Spelling Correction)**

**Concept:**
- Let Whisper transcribe ("Mac lot")
- Apply a spell-checker or LLM to fix known errors

**Implementation:**
```python
from faster_whisper import WhisperModel

# Transcribe
model = WhisperModel("medium")
segments, info = model.transcribe("audio.wav")
text = " ".join([seg.text for seg in segments])

# Post-process with corrections
corrections = {
    "Mac lot": "Macolet",
    "miss rod": "misrad",
    "to that say hoot": "te'udat zehut",
}

for wrong, right in corrections.items():
    text = text.replace(wrong, right)

print(text)
```

**Pros:**
- ✅ Works immediately (no training)
- ✅ Easy to implement

**Cons:**
- ❌ Manual dictionary maintenance
- ❌ Fragile (Whisper might transcribe "Mac lot" differently each time)
- ❌ Doesn't generalize (new words need new rules)

**Use case:** Temporary fix while preparing fine-tuning data.

---

### **4. Prompt/Injection (Whisper's Hidden Feature)**

**Whisper supports "initial prompt"** (hint to the decoder):

```python
result = model.transcribe(
    "audio.wav",
    initial_prompt="Common Hebrew words: Macolet, misrad, te'udat zehut, Shabbat"
)
```

**How it works:**
- Decoder sees these words as context
- Slightly biases output toward these spellings

**Effectiveness:**
- Modest improvement (not a silver bullet)
- Works best for words that are phonetically close to transcription errors
- Doesn't add new tokens, just biases existing ones

**Worth trying** as a quick test!

---

## Linguistic Origin vs. Training Data

### **Your Question: Does Linguistic Origin Matter?**

**Short answer: No, training data matters.**

**Examples:**

| Word | Origin | Whisper Recognition | Reason |
|------|--------|-------------------|--------|
| "Shabbat" | Hebrew | ✅ Good | High frequency in English audio |
| "Macolet" | Hebrew | ❌ Poor | Rare in English audio |
| "Schadenfreude" | German | ✅ Good | Common in English discourse |
| "Fernweh" | German | ❌ Poor | Rare in English discourse |
| "Sushi" | Japanese | ✅ Excellent | Ubiquitous in English |
| "Omakase" | Japanese | ⚠️ Mixed | Growing but not universal |

**What determines recognition:**
1. **Frequency** in English-language audio (not the word's origin)
2. **Cultural integration** (how much the word is used in English contexts)
3. **Media representation** (how often it appears in Whisper's training sources)

**Hebrew words in English:**
- "Shabbat", "kosher", "Hanukkah" → ✅ Well-known, high frequency
- "Macolet", "misrad", "te'udat zehut" → ❌ Israeli-specific, low frequency

---

## Summary: Why Variance Exists

**Your observation:**
> "I encounter variance in what I find [Whisper recognizing]"

**Explanation:**

| Factor | "Shabbat" (Works) | "Macolet" (Fails) |
|--------|------------------|------------------|
| **Training data frequency** | High (10k+ examples) | Low/Zero (<10 examples) |
| **BPE tokenization** | Single token `[Shabbat]` | Subword split `[Mac][ol][et]` |
| **Phonetic ambiguity** | Low (distinct sound) | High (sounds like "Mac lot") |
| **Cultural integration** | International Jewish culture | Israeli-specific slang |
| **Media representation** | YouTube, podcasts, TV | Rare outside Israel |

**The variance is entirely due to training data distribution, not linguistic origin.**

---

## Practical Recommendations for You

### **Option 1: Fine-Tune (Best Long-Term)**

Collect 2-5 hours of your speech with Hebrew words, transcribe carefully, fine-tune Whisper.

**Result:** All your Hebrew words (Macolet, misrad, etc.) recognized correctly.

### **Option 2: Initial Prompt (Quick Test)**

```python
result = model.transcribe(
    "audio.wav",
    initial_prompt="Hebrew words used: Macolet (convenience store), misrad (office), te'udat zehut (ID card)"
)
```

**Result:** Modest improvement (worth trying).

### **Option 3: Post-Processing (Interim Fix)**

Maintain a dictionary of corrections, apply after transcription.

**Result:** Works but fragile.

### **Recommended Path:**

1. **Now:** Use initial prompt + post-processing
2. **Short-term:** Collect audio data with Hebrew words
3. **Long-term:** Fine-tune Whisper (or wait for a Hebrew-English code-switching dataset to fine-tune on)

---

## Bottom Line

**ASR works at the phonetic/subword level, but vocabulary recognition is driven by training data frequency.**

- **"Shabbat" works**: High frequency in Whisper's training data (English-language audio with Jewish cultural content)
- **"Macolet" fails**: Low/zero frequency (Israeli-specific, rare outside Israel)

**Fine-tuning is the solution**: By providing examples of your Hebrew words in English contexts, you teach Whisper to recognize them reliably.

**This is exactly the use case where personal fine-tuning shines.**

---

**Note**: This explanation was generated by Claude Code (claude-sonnet-4-5) for Daniel Rosehill's STT Fine-Tuning Notebook. Whisper's vocabulary recognition is probabilistic and depends on training data distribution. For reliable transcription of code-switched speech (English + Hebrew), fine-tuning is the most effective solution. Consider creating a dataset of 2-5 hours with Hebrew words you use regularly, ensuring diverse contexts and pronunciations. Initial prompts can provide modest improvements as an interim measure.
