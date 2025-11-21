# Audio Specifications for Whisper Fine-Tuning

## Overview

Proper audio specifications are critical for successful Whisper model fine-tuning. This guide covers the recommended bitrate settings and sample length requirements for preparing training data.

## Audio Format Requirements

### Sample Rate
- **Required**: 16 kHz (16,000 Hz)
- Whisper models are trained exclusively on 16 kHz audio
- Higher sample rates will be automatically downsampled
- Lower sample rates may result in quality degradation

### Bit Depth
- **Recommended**: 16-bit PCM
- 24-bit or 32-bit audio will be converted to 16-bit
- 8-bit audio is not recommended due to quality loss

### Bitrate
- **For 16 kHz, 16-bit mono**: ~256 kbps (uncompressed)
- **Compressed formats** (if using MP3/AAC):
  - Minimum: 128 kbps
  - Recommended: 192-256 kbps
  - Avoid: Below 128 kbps (artifacts may affect training)

### Channels
- **Required**: Mono (single channel)
- Stereo files will be converted to mono by averaging channels
- For stereo recordings, ensure important audio is not phase-cancelled

## Sample Length Guidelines

### Minimum Length
- **Absolute minimum**: 1 second
- **Practical minimum**: 2-3 seconds
- Very short samples may not provide enough context for effective learning

### Maximum Length
- **Hard limit**: 30 seconds
- Whisper processes audio in 30-second chunks
- Samples longer than 30 seconds will be truncated

### Optimal Length Range
- **Recommended**: 5-15 seconds per sample
- **Sweet spot**: 8-12 seconds
- This range provides:
  - Sufficient context for the model
  - Complete phrases or sentences
  - Efficient training batch processing
  - Good balance of data diversity

### Length Distribution
For best results, your dataset should have:
- **Varied lengths** within the 5-15 second range
- **Avoid**: All samples being exactly the same length
- **Include**: A mix of shorter phrases and longer utterances
- **Natural boundaries**: Cut at sentence or phrase boundaries when possible

## File Format Recommendations

### Best Formats
1. **WAV** (PCM, 16 kHz, 16-bit, mono)
   - Uncompressed, no quality loss
   - Larger file sizes
   - Industry standard for training data

2. **FLAC** (16 kHz, mono)
   - Lossless compression
   - Smaller than WAV
   - No quality degradation

### Acceptable Formats
3. **MP3** (192+ kbps, 16 kHz, mono)
   - Lossy compression
   - Use only if storage is critical
   - Ensure high bitrate (192 kbps minimum)

4. **OGG Vorbis** (192+ kbps, 16 kHz, mono)
   - Open-source alternative to MP3
   - Similar quality considerations

### Formats to Avoid
- Low-bitrate MP3 (<128 kbps)
- Highly compressed formats (AMR, SPEEX)
- Variable bitrate with very low minimum rates
- Formats with aggressive noise reduction applied

## Data Quality Considerations

### Signal-to-Noise Ratio
- **Minimum SNR**: 20 dB
- **Recommended SNR**: 30+ dB
- Clean audio produces better fine-tuning results

### Audio Preprocessing
- **Normalization**: Normalize audio to -3 dB to -1 dB peak
- **Silence trimming**: Remove long silences at start/end
- **Noise reduction**: Apply if needed, but avoid aggressive processing
- **Avoid**: Heavy compression, excessive EQ, artificial effects

### Recording Environment
- **Preferred**: Quiet indoor environment
- **Acceptable**: Controlled background noise
- **Avoid**: Highly reverberant spaces, loud background noise

## Batch Preparation Tips

### Converting Existing Audio

Convert to 16 kHz mono WAV:
```bash
ffmpeg -i input.mp3 -ar 16000 -ac 1 -c:a pcm_s16le output.wav
```

Batch conversion:
```bash
for file in *.mp3; do
    ffmpeg -i "$file" -ar 16000 -ac 1 -c:a pcm_s16le "${file%.mp3}.wav"
done
```

### Splitting Long Audio Files

Split into 30-second chunks:
```bash
ffmpeg -i input.wav -f segment -segment_time 30 -c copy output_%03d.wav
```

### Quality Check

Verify audio specifications:
```bash
ffprobe -v error -show_entries stream=sample_rate,channels,codec_name,bit_rate input.wav
```

## Dataset Size Recommendations

### Minimum Dataset
- **Audio duration**: 1 hour of transcribed audio
- **Number of samples**: Varies (120-720 samples depending on length)
- Sufficient for domain-specific adaptation

### Recommended Dataset
- **Audio duration**: 5-10 hours
- **Number of samples**: 1,000-5,000
- Provides robust fine-tuning results

### Large Dataset
- **Audio duration**: 20+ hours
- **Number of samples**: 10,000+
- For significant model adaptation or new languages

## Summary Table

| Parameter | Requirement | Recommended |
|-----------|-------------|-------------|
| Sample Rate | 16 kHz | 16 kHz |
| Bit Depth | 16-bit | 16-bit |
| Channels | Mono | Mono |
| Format | WAV/FLAC | WAV (PCM) |
| Bitrate (compressed) | 128+ kbps | 192-256 kbps |
| Min Length | 1 second | 5 seconds |
| Max Length | 30 seconds | 15 seconds |
| Optimal Range | 2-30 seconds | 8-12 seconds |
| Total Duration | 1+ hours | 5-10 hours |

## Common Issues and Solutions

### Issue: "Audio too short" errors
- **Solution**: Filter out samples under 2 seconds

### Issue: Poor training results
- **Solution**: Check SNR, ensure consistent audio quality, verify transcription accuracy

### Issue: Out of memory during training
- **Solution**: Reduce batch size, limit maximum sample length to 20 seconds

### Issue: Slow training
- **Solution**: Ensure samples are properly truncated to 30 seconds max

## References

- OpenAI Whisper Documentation
- Hugging Face Datasets Audio Processing Guide
- Speech Recognition Best Practices

---

*Last updated: 2025-01-21*
