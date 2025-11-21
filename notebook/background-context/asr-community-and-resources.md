# ASR Community and Resources: Staying Current with Speech Recognition

## Question Summary

Daniel asks for recommendations on how to stay up-to-date with automatic speech recognition (ASR) developments. He mentions arXiv is well-known for LLMs and wonders what equivalents exist for ASR. He's looking for: individuals to follow, companies to watch, blogs, YouTube channels, conferences, and communities (physical/virtual) to connect with like-minded people exploring this field.

## Answer

Excellent question! The ASR community is vibrant but more specialized than the LLM community, which means finding the right resources requires knowing where to look. Let me provide a comprehensive guide to the ASR ecosystem.

### Academic & Research Publications

#### **arXiv (Yes, ASR is There!)**

```
arXiv Categories for ASR:

Primary:
- cs.SD (Sound) - Audio and speech processing
- cs.CL (Computation and Language) - Includes speech-to-text
- eess.AS (Audio and Speech Processing) - Signal processing

Search Terms:
- "automatic speech recognition"
- "ASR"
- "speech-to-text"
- "wav2vec"
- "whisper"
- "end-to-end speech recognition"

Frequency: 10-20 new ASR papers per week

Tip: Set up Google Scholar alerts for these terms
```

**How to Follow arXiv for ASR:**

```
Option 1: Daily arXiv Emails
- Subscribe to cs.SD and eess.AS
- Filter by keywords in your email

Option 2: arXiv Sanity (by Andrej Karpathy)
- http://www.arxiv-sanity.com/
- Better filtering and recommendations

Option 3: Papers with Code
- https://paperswithcode.com/task/speech-recognition
- Links papers with implementations
- Shows benchmarks and SOTA models
```

#### **Key Academic Conferences**

**Top-Tier Speech Conferences:**

**1. INTERSPEECH (Annual - September)**
```
- THE premier conference for speech research
- ~1,000-1,500 attendees
- Covers: ASR, TTS, speaker recognition, prosody
- Location: Rotates globally
- Papers: 500+ presentations
- Virtual attendance: Usually available

Why Follow:
- Cutting-edge research (6-12 months ahead of industry)
- Workshops on specific topics (low-resource ASR, streaming, etc.)
- Networking with researchers and engineers

How to Stay Updated:
- YouTube: ISCA (International Speech Communication Association)
- Papers: Available after conference
- Twitter/X: #INTERSPEECH hashtag
```

**2. ICASSP (IEEE International Conference on Acoustics, Speech, and Signal Processing)**
```
- Largest signal processing conference
- Broader than just ASR (audio, signal processing)
- ~3,000+ attendees
- Annual (usually April-May)

ASR Content:
- 100-200 ASR-specific papers
- Mixed with audio, music, signal processing

Access:
- IEEE Xplore (papers)
- YouTube (some talks)
- Very academic/technical
```

**3. ACL/EMNLP/NAACL (NLP Conferences with Speech Tracks)**
```
- Association for Computational Linguistics conferences
- Include speech-to-text, multimodal sessions
- More language-focused than acoustic-focused

Relevant for:
- Language modeling in ASR
- Cross-lingual speech recognition
- Speech translation
```

**4. NeurIPS/ICML (Machine Learning Conferences)**
```
- General ML conferences
- Include speech recognition papers
- More methodology-focused (new architectures, training techniques)

Example Topics:
- Self-supervised learning for speech (Wav2Vec papers)
- Efficient transformers for ASR
- Few-shot learning for low-resource languages
```

### Industry Blogs & Company Research

#### **Top Companies to Follow**

**1. OpenAI**
```
Website: openai.com/research
Blog: openai.com/blog
Twitter/X: @OpenAI

Contributions:
- Whisper (open source)
- Whisper API (closed source, likely v4)
- Multimodal models (GPT-4 with audio rumored)

Follow For:
- Whisper updates and improvements
- New model releases
- API enhancements
```

**2. Meta AI (Facebook AI Research)**
```
Website: ai.meta.com
Research: research.facebook.com
GitHub: github.com/facebookresearch

Major Contributions:
- Wav2Vec 2.0 (self-supervised learning)
- HuBERT (Hidden Unit BERT)
- MMS (Massively Multilingual Speech - 1,100+ languages)
- SeamlessM4T (speech translation)

Follow For:
- Open-source models
- Research on low-resource languages
- Self-supervised learning advances
```

**3. Google Research / Google AI**
```
Blog: ai.googleblog.com
Papers: research.google/pubs/ (filter by "speech")
YouTube: Google TechTalks

Major Contributions:
- USM (Universal Speech Model - 300+ languages)
- YouTube auto-captioning (drives Whisper training data!)
- Voice Search, Google Assistant
- Conformer architecture

Follow For:
- Multilingual ASR
- On-device models
- Production-scale systems
```

**4. NVIDIA**
```
Blog: developer.nvidia.com/blog
GitHub: github.com/NVIDIA
Developer: developer.nvidia.com/nemo

Major Contributions:
- NeMo Toolkit (ASR framework)
- Canary model (streaming ASR)
- Riva (deployment platform)

Follow For:
- Real-time streaming ASR
- GPU optimization techniques
- Enterprise deployment
```

**5. Microsoft Research**
```
Blog: www.microsoft.com/en-us/research/blog/
Research: microsoft.com/en-us/research/research-area/speech-language/

Contributions:
- Azure Speech Services
- Nuance acquisition (medical ASR)
- WavLM, UniSpeech models

Follow For:
- Enterprise ASR
- Azure API updates
- Medical transcription
```

**6. Hugging Face**
```
Blog: huggingface.co/blog
Models: huggingface.co/models?pipeline_tag=automatic-speech-recognition
Forum: discuss.huggingface.co

Why Follow:
- Community hub for ASR models
- Tutorials and guides
- Model comparisons and benchmarks
- Integration guides (Whisper, Wav2Vec, etc.)

Specific Follows:
- @patrickvonplaten (Hugging Face speech lead)
- Models: 1,000+ ASR models available
```

#### **Specialized ASR Companies**

**AssemblyAI**
```
Website: assemblyai.com
Blog: assemblyai.com/blog
Twitter: @AssemblyAI
YouTube: AssemblyAI

Why Follow:
- Excellent technical blog posts
- API-first ASR company
- Transparent about model development
- Real-world benchmarks
- Regular feature releases (LeMUR, speaker diarization, etc.)

Content Quality: Very high, developer-focused
```

**Deepgram**
```
Website: deepgram.com
Blog: deepgram.com/learn
Twitter: @DeepgramAI

Why Follow:
- Nova model (competitive with Whisper)
- Streaming ASR focus
- Developer tutorials
- Benchmarking studies
```

**Rev.ai**
```
Website: rev.ai
Blog: rev.ai/blog

Why Follow:
- Professional transcription perspective
- Human-ASR hybrid workflows
- Quality benchmarks
```

### Individual Researchers & Engineers to Follow

#### **Twitter/X Accounts**

**Academic Researchers:**

```
@awni00 - Awni Hannun
- Co-creator of Wav2Vec
- Meta AI researcher
- Deep learning for speech

@jacobandreas_ - Jacob Andreas
- MIT, NLP and speech
- Compositional learning

@alexeigz - Alexei Baevski
- Meta AI
- Wav2Vec 2.0, data2vec
- Self-supervised learning

@bhiksha - Bhiksha Raj
- CMU professor
- Speech processing research
```

**Industry Engineers:**

```
@sanchitgandhi99 - Sanchit Gandhi
- Hugging Face speech team
- Whisper expert
- Excellent tutorials

@patrickvonplaten - Patrick von Platen
- Hugging Face speech lead
- Transformers library maintainer

@jon_barker - Jon Barker
- Sheffield University
- CHiME challenges (noisy speech)

@shinji_watanabe - Shinji Watanabe
- Carnegie Mellon University
- ESPnet creator (ASR toolkit)
```

**Thought Leaders:**

```
@ylecun - Yann LeCun
- Meta Chief AI Scientist
- Occasionally discusses speech

@karpathy - Andrej Karpathy
- OpenAI (formerly)
- Occasionally covers multimodal (including speech)
```

### YouTube Channels

**Academic/Educational:**

**1. Yannic Kilcher**
```
Channel: youtube.com/@YannicKilcher
Focus: Paper reviews, including speech papers
Content: Deep dives into Wav2Vec, Whisper, etc.
Frequency: Weekly
Level: Advanced
```

**2. Two Minute Papers**
```
Channel: youtube.com/@TwoMinutePapers
Focus: General AI, occasional speech papers
Content: Accessible summaries
Frequency: Multiple per week
Level: Beginner-friendly
```

**3. Arxiv Insights**
```
Channel: youtube.com/@ArxivInsights
Focus: Research paper breakdowns
Content: Occasional ASR papers
Level: Intermediate
```

**Company/Product Channels:**

**4. AssemblyAI**
```
Channel: youtube.com/@AssemblyAI
Focus: ASR tutorials, demos, webinars
Content: Practical, developer-focused
Frequency: Monthly
Level: All levels
```

**5. Hugging Face**
```
Channel: youtube.com/@HuggingFace
Focus: Tutorials, model releases
Content: Code walkthroughs, demos
Frequency: Weekly
Level: Intermediate
```

**Conference Recordings:**

**6. INTERSPEECH YouTube**
```
Search: "INTERSPEECH [year]"
Content: Conference talks, tutorials
Level: Advanced
```

### Online Communities

#### **Reddit**

**r/speechrecognition**
```
URL: reddit.com/r/speechrecognition
Members: ~5,000
Activity: Moderate (5-10 posts/day)
Content:
- Troubleshooting ASR models
- New model discussions
- Project showcases
- Beginner questions

Best For: Practical implementation discussions
```

**r/MachineLearning**
```
URL: reddit.com/r/MachineLearning
Members: 2.8M+
Activity: Very high
ASR Content: Occasional (when major releases like Whisper v3)

Search: Filter by "speech" or "ASR" flair
```

**r/LanguageTechnology**
```
URL: reddit.com/r/LanguageTechnology
Members: 50K+
Activity: Moderate
Content: Speech-to-text, NLP overlap
```

#### **Discord Servers**

**Hugging Face Discord**
```
Invite: hf.co/join/discord
Channels: #audio, #speech
Members: 100K+
Activity: Very active

Best For:
- Getting help with Transformers library
- Model fine-tuning questions
- Community support
```

**EleutherAI Discord**
```
Focus: Open-source AI models
Channels: Occasional speech discussions
Members: 30K+

Best For: Technical discussions, research collaboration
```

**Laion Discord**
```
Focus: Open datasets, models
Channels: #audio, #speech-recognition
Members: 20K+

Best For: Dataset discussions, collaborative projects
```

#### **Forums & Discussion Boards**

**Hugging Face Forums**
```
URL: discuss.huggingface.co
Tags: #audio, #asr, #speech-recognition

Best For:
- Technical troubleshooting
- Model comparisons
- Fine-tuning guides
```

**Speech Recognition Discourse** (Less active)
```
Various university-hosted forums
Search: "[university] speech recognition forum"
```

### GitHub Repositories to Watch

**Frameworks & Toolkits:**

```
1. openai/whisper
   - Official Whisper repository
   - 60K+ stars
   - Watch for updates, issues

2. speechbrain/speechbrain
   - All-in-one speech toolkit
   - 8K+ stars
   - Comprehensive ASR, TTS, etc.

3. espnet/espnet
   - End-to-end speech processing
   - CMU/Johns Hopkins
   - Research-grade toolkit

4. NVIDIA/NeMo
   - NVIDIA's speech AI toolkit
   - Canary model, streaming ASR

5. huggingface/transformers
   - Whisper, Wav2Vec integrations
   - Production-ready implementations

6. m-bain/whisperX
   - Enhanced Whisper (better timestamps)
   - Active development

7. guillaumekln/faster-whisper
   - Optimized Whisper inference
   - 4-5x speedup
```

**"Awesome" Lists:**

```
awesome-speech-recognition
- Curated list of ASR resources
- Search GitHub: "awesome speech recognition"
```

### Blogs & Newsletters

**Technical Blogs:**

**1. AssemblyAI Blog**
```
URL: assemblyai.com/blog
Frequency: 2-3 posts/month
Quality: Excellent
Content:
- Deep dives into ASR architectures
- Benchmarking studies
- Tutorials and guides

Recommended Posts:
- "The Full Story of Large-Scale ASR"
- "Conformers for Speech Recognition"
- Speaker Diarization guides
```

**2. Deepgram Blog**
```
URL: deepgram.com/learn
Frequency: Monthly
Content: Developer-focused, practical guides
```

**3. Google AI Blog**
```
URL: ai.googleblog.com
Filter: Search "speech" or "ASR"
Frequency: Occasional speech posts
Content: High-level research summaries
```

**Newsletters:**

**1. The Batch (deeplearning.ai)**
```
URL: deeplearning.ai/the-batch
Editor: Andrew Ng
Frequency: Weekly
Content: General AI news, occasional ASR

ASR Coverage: ~1-2 times/month when major releases
```

**2. Import AI**
```
URL: importai.substack.com
Editor: Jack Clark
Frequency: Weekly
Content: AI research roundup, includes speech papers
```

**3. Papers with Code Newsletter**
```
URL: paperswithcode.com
Frequency: Weekly
Content: Latest SOTA results, includes ASR benchmarks
```

### Podcasts

**1. TWIML AI Podcast (This Week in Machine Learning & AI)**
```
Hosts: Occasional speech researchers
Frequency: Weekly (speech episodes ~monthly)
Episodes: Search "speech recognition" or "ASR"

Notable Episodes:
- Whisper release discussion
- Wav2Vec 2.0 deep dive
- Low-resource language ASR
```

**2. The AI Podcast (NVIDIA)**
```
Content: Occasional speech/audio episodes
Guest Quality: High (researchers, engineers)
```

**3. Practical AI**
```
Hosts: Changelog
Content: Practical ML, occasional ASR
Level: Intermediate
```

### Professional Organizations

**ISCA (International Speech Communication Association)**
```
Website: isca-speech.org
Benefits:
- Access to INTERSPEECH proceedings
- Student discounts
- Member events

Membership: ~$50-100/year
Worth It: Yes, if attending conferences
```

**IEEE Signal Processing Society**
```
Website: signalprocessingsociety.org
Benefits:
- ICASSP discounts
- IEEE Xplore access (papers)
- Webinars and events

Membership: ~$100-150/year
```

### Benchmarks & Leaderboards

**Track SOTA Models:**

**1. Papers with Code**
```
URL: paperswithcode.com/task/speech-recognition
Content:
- Current SOTA models
- Benchmark datasets (LibriSpeech, Common Voice, etc.)
- Historical WER trends

Updated: Real-time as papers released
```

**2. HuggingFace Leaderboards**
```
URL: huggingface.co/spaces (search "ASR leaderboard")
Content: Community-driven model comparisons
```

**3. ESB Benchmark (End-to-end Speech Benchmark)**
```
GitHub: speechbrain/benchmarks
Content: Comprehensive ASR benchmarking
Datasets: Multiple, diverse conditions
```

### Conferences (Beyond Academic)

**Industry Conferences:**

**1. Voice Summit / VOICE**
```
Focus: Voice AI, conversational AI, ASR
Attendees: ~2,000 (virtual + in-person)
Content: Industry trends, product demos
Frequency: Annual
```

**2. SpeechTEK**
```
Focus: Enterprise speech technology
Attendees: ~1,000
Content: Deployment, ROI, case studies
Audience: Business + technical
```

**3. AI Summit / RE•WORK**
```
Content: Broad AI, includes speech tracks
Format: Workshops + talks
Locations: Global (London, NYC, SF, etc.)
```

### Following Specific Use Cases

If you're interested in specific domains:

**Medical ASR:**
```
- Nuance Communications blog
- AMIA (American Medical Informatics Association)
- @NuanceMedical on Twitter
```

**Legal Transcription:**
```
- Verbit blog
- Court reporting associations
```

**Accessibility:**
```
- @AccessibleTech communities
- Caption accessibility forums
```

### How to Build Your Personal Feed

**Recommended Starter Pack:**

```
Twitter/X (Follow 5-10):
- @AssemblyAI
- @OpenAI
- @HuggingFace
- @sanchitgandhi99
- @patrickvonplaten

RSS/Newsletters (Subscribe to 2-3):
- AssemblyAI Blog RSS
- Papers with Code (ASR category)
- The Batch (deeplearning.ai)

YouTube (Subscribe):
- AssemblyAI
- Hugging Face
- Yannic Kilcher (for paper reviews)

GitHub (Watch):
- openai/whisper
- huggingface/transformers
- speechbrain/speechbrain

Reddit (Join):
- r/speechrecognition
- r/MachineLearning

Discord:
- Hugging Face Discord (#audio channel)

Conferences (Attend Virtual):
- INTERSPEECH (September, virtual option)
```

### Regional/Local Communities

**Look for:**
```
- University speech labs (if near major university)
  - CMU, MIT, Stanford, Johns Hopkins
- Meetup.com: Search "speech recognition" or "voice AI"
- Local AI/ML meetups (often include speech topics)
- Company-hosted events (Google, Meta, Microsoft research labs)
```

### Conclusion: Building Your ASR Ecosystem

**For Staying Current:**
1. **Academic:** arXiv (cs.SD, eess.AS) + INTERSPEECH
2. **Industry:** AssemblyAI blog, OpenAI updates, Hugging Face
3. **Community:** Reddit r/speechrecognition, Hugging Face Discord
4. **Code:** GitHub (Whisper, Transformers, SpeechBrain)

**For Networking:**
1. **Virtual:** Discord servers, Reddit communities
2. **Conferences:** INTERSPEECH (academic), Voice Summit (industry)
3. **Twitter/X:** Follow researchers and engineers

**For Hands-On Learning:**
1. **YouTube:** AssemblyAI, Hugging Face tutorials
2. **Blogs:** AssemblyAI deep dives
3. **GitHub:** Explore and star repositories

**Time Investment:**
- Casual: 1-2 hours/week (Twitter, Reddit, newsletter)
- Moderate: 3-5 hours/week (+ blog posts, YouTube)
- Deep: 10+ hours/week (+ papers, conferences, projects)

The ASR community is smaller than LLM but highly engaged. Start with the "starter pack" above and expand based on your specific interests (medical, multilingual, real-time, etc.). Welcome to the community!

---

*Note: This response was generated by Claude Code as part of Daniel's STT Fine-Tuning Notebook project. Links and resources are current as of 2025, but always verify availability.*
