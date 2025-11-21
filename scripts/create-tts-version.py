#!/usr/bin/env python3
"""
Create a TTS-safe version of the combined notebook.
Removes code blocks, simplifies formatting, and optimizes for text-to-speech.
"""

import re
import sys
from pathlib import Path


def process_for_tts(content):
    """
    Process markdown content to make it TTS-friendly.
    """

    # Remove code blocks (both fenced and indented)
    content = re.sub(r'```[\s\S]*?```', '[Code block removed for TTS]', content)
    content = re.sub(r'^    .*$', '', content, flags=re.MULTILINE)

    # Remove URLs but keep descriptive text
    # Replace markdown links [text](url) with just the text
    content = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', content)
    # Remove standalone URLs
    content = re.sub(r'https?://[^\s]+', '[URL removed]', content)

    # Convert star ratings to words
    content = re.sub(r'⭐{5}', 'five stars', content)
    content = re.sub(r'⭐{4}', 'four stars', content)
    content = re.sub(r'⭐{3}', 'three stars', content)
    content = re.sub(r'⭐{2}', 'two stars', content)
    content = re.sub(r'⭐{1}', 'one star', content)

    # Remove/simplify tables - convert to simpler format
    # This is complex, so we'll just add a note
    content = re.sub(r'\|[^\n]+\|[^\n]+\n\|[-:\s|]+\|[^\n]+\n(\|[^\n]+\n)*',
                     '[Table removed for TTS - see original document]\n\n', content)

    # Simplify headings - remove # symbols but keep hierarchy with "Section:" prefix
    def heading_replacer(match):
        level = len(match.group(1))
        text = match.group(2)
        if level == 1:
            return f"\n\nMAIN SECTION: {text}\n\n"
        elif level == 2:
            return f"\n\nSection: {text}\n\n"
        elif level == 3:
            return f"\n\nSubsection: {text}\n\n"
        else:
            return f"\n\n{text}\n\n"

    content = re.sub(r'^(#{1,6})\s+(.+)$', heading_replacer, content, flags=re.MULTILINE)

    # Convert bullet points to numbered sentences
    # Keep the bullets but remove special characters
    content = re.sub(r'^\s*[-*+]\s+', '- ', content, flags=re.MULTILINE)

    # Remove horizontal rules
    content = re.sub(r'^---+$', '', content, flags=re.MULTILINE)
    content = re.sub(r'^\*\*\*+$', '', content, flags=re.MULTILINE)

    # Simplify bold and italic (remove the markdown but keep text)
    content = re.sub(r'\*\*\*([^\*]+)\*\*\*', r'\1', content)  # bold+italic
    content = re.sub(r'\*\*([^\*]+)\*\*', r'\1', content)  # bold
    content = re.sub(r'\*([^\*]+)\*', r'\1', content)  # italic
    content = re.sub(r'__([^_]+)__', r'\1', content)  # bold
    content = re.sub(r'_([^_]+)_', r'\1', content)  # italic

    # Replace common abbreviations with pronounceable versions
    replacements = {
        'STT': 'S T T',
        'ASR': 'A S R',
        'TTS': 'T T S',
        'LLM': 'L L M',
        'GPU': 'G P U',
        'CPU': 'C P U',
        'VRAM': 'V RAM',
        'AMD': 'A M D',
        'ROCm': 'rock m',
        'CUDA': 'CUDA',
        'PyTorch': 'Pie Torch',
        'ONNX': 'on x',
        'API': 'A P I',
        'SDK': 'S D K',
        'CLI': 'C L I',
        'UI': 'U I',
        'ML': 'M L',
        'AI': 'A I',
        'GB': 'gigabytes',
        'MB': 'megabytes',
        'KB': 'kilobytes',
        'Hz': 'hertz',
        'kHz': 'kilohertz',
        'ms': 'milliseconds',
        'e.g.': 'for example',
        'i.e.': 'that is',
        'etc.': 'etcetera',
    }

    for abbr, spoken in replacements.items():
        # Use word boundaries to avoid partial replacements
        content = re.sub(r'\b' + re.escape(abbr) + r'\b', spoken, content)

    # Remove inline code markers
    content = re.sub(r'`([^`]+)`', r'\1', content)

    # Clean up excessive whitespace
    content = re.sub(r'\n{4,}', '\n\n\n', content)

    # Remove backticks and other code-related characters
    content = content.replace('`', '')

    # Remove file path indicators like "file.py:123"
    content = re.sub(r'[a-zA-Z0-9_/.-]+\.py:\d+', '[file reference removed]', content)
    content = re.sub(r'[a-zA-Z0-9_/.-]+\.sh:\d+', '[file reference removed]', content)

    # Add pronunciation hints for common technical terms
    content = content.replace('Whisper', 'Whisper')  # This one is fine
    content = content.replace('RDNA', 'R D N A')
    content = content.replace('gfx1101', 'G F X eleven oh one')
    content = content.replace('gfx1100', 'G F X eleven hundred')
    content = content.replace('gfx1030', 'G F X ten thirty')

    # Remove bash/shell prompts
    content = re.sub(r'^\$\s+', '', content, flags=re.MULTILINE)
    content = re.sub(r'^#\s+', '', content, flags=re.MULTILINE)

    return content


def main():
    # File paths
    input_file = Path('/home/daniel/repos/github/STT-Fine-Tuning-Notebook/notebook/combined-notebook.md')
    output_file = Path('/home/daniel/repos/github/STT-Fine-Tuning-Notebook/notebook/combined-notebook-tts.md')

    print(f"Reading {input_file}...")

    # Read the input file
    with open(input_file, 'r', encoding='utf-8') as f:
        content = f.read()

    print(f"Original size: {len(content):,} characters")

    # Process for TTS
    print("Processing for TTS...")
    tts_content = process_for_tts(content)

    print(f"TTS version size: {len(tts_content):,} characters")

    # Add header
    header = """# STT Fine-Tuning Notebook - TTS-Optimized Version

This is a text-to-speech optimized version of the STT Fine-Tuning Notebook.
Code blocks, tables, and complex formatting have been removed or simplified for audio reading.
For the complete version with code examples and technical details, see the original combined-notebook.md file.

---

"""

    tts_content = header + tts_content

    # Write output
    print(f"Writing to {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(tts_content)

    print(f"✓ TTS-safe version created successfully!")
    print(f"  Input:  {input_file}")
    print(f"  Output: {output_file}")
    print(f"  Size reduction: {len(content) - len(tts_content):,} characters ({((len(content) - len(tts_content)) / len(content) * 100):.1f}%)")


if __name__ == '__main__':
    main()
