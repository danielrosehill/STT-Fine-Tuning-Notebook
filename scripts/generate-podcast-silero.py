#!/usr/bin/env python3
"""
Generate podcast audio from SSML files using Silero TTS (local, free).
Silero provides high-quality TTS models that run locally without API costs.
"""

import os
import sys
from pathlib import Path
import json
from datetime import datetime
import re
import torch
import torchaudio

# Configuration
SSML_DIR = Path(__file__).parent.parent / "podcast-ssml"
OUTPUT_DIR = Path(__file__).parent.parent / "podcast-audio"

# Silero TTS Configuration
SAMPLE_RATE = 48000  # Silero outputs at 48kHz
DEFAULT_SPEAKER = 'en_0'  # English speaker
# Other speakers: en_1, en_2, en_3, ... (different voices)

def strip_ssml_tags(ssml_content):
    """Remove SSML tags and return plain text for TTS."""
    # Remove all XML/SSML tags
    text = re.sub(r'<[^>]+>', ' ', ssml_content)
    # Clean up multiple spaces
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def initialize_silero_model():
    """Initialize and return Silero TTS model."""
    print("Loading Silero TTS model...")
    try:
        # Load Silero model from torch hub
        model, _ = torch.hub.load(
            repo_or_dir='snakers4/silero-models',
            model='silero_tts',
            language='en',
            speaker='v3_en'
        )

        model.to('cpu')  # Use CPU (can change to 'cuda' if GPU available)
        print("✓ Silero model loaded")
        return model
    except Exception as e:
        print(f"Error loading Silero model: {e}")
        print("\nMake sure you have torch and torchaudio installed:")
        print("  pip install torch torchaudio")
        sys.exit(1)


def convert_ssml_to_audio_silero(ssml_file, output_file, model, speaker=DEFAULT_SPEAKER):
    """Convert SSML file to audio using Silero TTS."""
    print(f"Converting {ssml_file.name} to audio...")

    try:
        # Read SSML content
        with open(ssml_file, 'r', encoding='utf-8') as f:
            ssml_content = f.read()

        # Strip SSML tags to get plain text
        text = strip_ssml_tags(ssml_content)

        if not text:
            print(f"Warning: No text content found in {ssml_file.name}")
            return False

        # Split text into chunks if too long (Silero works best with shorter chunks)
        max_chunk_length = 1000
        chunks = []
        words = text.split()

        current_chunk = []
        current_length = 0

        for word in words:
            current_chunk.append(word)
            current_length += len(word) + 1

            if current_length >= max_chunk_length:
                chunks.append(' '.join(current_chunk))
                current_chunk = []
                current_length = 0

        if current_chunk:
            chunks.append(' '.join(current_chunk))

        # Generate audio for each chunk
        audio_chunks = []
        for chunk in chunks:
            audio = model.apply_tts(
                text=chunk,
                speaker=speaker,
                sample_rate=SAMPLE_RATE
            )
            audio_chunks.append(audio)

        # Concatenate all audio chunks
        full_audio = torch.cat(audio_chunks)

        # Save audio file
        torchaudio.save(
            str(output_file),
            full_audio.unsqueeze(0),
            SAMPLE_RATE
        )

        print(f"✓ Audio saved to: {output_file}")
        return True

    except Exception as e:
        print(f"Error converting {ssml_file.name}: {e}")
        return False


def concatenate_audio_files(audio_files, output_file):
    """Concatenate multiple audio files into a single podcast episode."""
    import subprocess

    print(f"\nConcatenating {len(audio_files)} audio files...")

    # Create a temporary file list for ffmpeg
    file_list_path = OUTPUT_DIR / "concat_list.txt"

    try:
        with open(file_list_path, 'w', encoding='utf-8') as f:
            for audio_file in audio_files:
                f.write(f"file '{audio_file.absolute()}'\n")

        # Use ffmpeg to concatenate
        cmd = [
            "ffmpeg",
            "-f", "concat",
            "-safe", "0",
            "-i", str(file_list_path),
            "-c", "copy",
            "-y",  # Overwrite output file
            str(output_file)
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode == 0:
            print(f"✓ Podcast saved to: {output_file}")
            return True
        else:
            print(f"Error concatenating audio: {result.stderr}")
            return False

    except Exception as e:
        print(f"Error concatenating audio: {e}")
        return False
    finally:
        # Clean up temporary file list
        if file_list_path.exists():
            file_list_path.unlink()


def process_ssml_directory(speaker=DEFAULT_SPEAKER, concatenate=True):
    """Process all SSML files in the directory."""
    # Create output directory
    OUTPUT_DIR.mkdir(exist_ok=True)

    # Load manifest from SSML conversion
    manifest_file = SSML_DIR / "manifest.json"
    if not manifest_file.exists():
        print("Error: manifest.json not found. Run convert-to-ssml.py first.")
        sys.exit(1)

    # Get all SSML files
    ssml_files = sorted(SSML_DIR.rglob("*.ssml"))

    if not ssml_files:
        print("Error: No SSML files found. Run convert-to-ssml.py first.")
        sys.exit(1)

    print(f"Found {len(ssml_files)} SSML files to process")
    print(f"Using speaker: {speaker}")

    # Initialize Silero model
    model = initialize_silero_model()

    # Create podcast manifest
    podcast_manifest = {
        "generated_at": datetime.now().isoformat(),
        "tts_engine": "silero",
        "speaker": speaker,
        "sample_rate": SAMPLE_RATE,
        "total_files": len(ssml_files),
        "files": []
    }

    audio_files = []

    for i, ssml_file in enumerate(ssml_files, 1):
        # Get relative path for organizing output
        rel_path = ssml_file.relative_to(SSML_DIR)

        # Create corresponding output directory structure
        output_subdir = OUTPUT_DIR / rel_path.parent
        output_subdir.mkdir(parents=True, exist_ok=True)

        # Output audio file path
        audio_file = output_subdir / f"{ssml_file.stem}.wav"

        print(f"\n[{i}/{len(ssml_files)}] Processing: {rel_path}")

        # Convert to audio
        success = convert_ssml_to_audio_silero(ssml_file, audio_file, model, speaker)

        if success:
            audio_files.append(audio_file)
            podcast_manifest["files"].append({
                "ssml_source": str(rel_path),
                "audio_output": str(audio_file.relative_to(OUTPUT_DIR)),
                "status": "success"
            })
        else:
            podcast_manifest["files"].append({
                "ssml_source": str(rel_path),
                "status": "failed"
            })

    # Save individual files manifest
    individual_manifest_file = OUTPUT_DIR / "individual-files-manifest.json"
    with open(individual_manifest_file, 'w', encoding='utf-8') as f:
        json.dump(podcast_manifest, f, indent=2)

    print(f"\n✓ Individual audio files generated!")
    print(f"✓ Manifest saved to: {individual_manifest_file}")

    # Optionally concatenate all audio files into a single podcast
    if concatenate and audio_files:
        print("\n" + "=" * 60)
        print("Concatenating audio files into full podcast...")
        print("=" * 60)

        full_podcast_file = OUTPUT_DIR / f"stt-finetune-podcast-{datetime.now().strftime('%Y%m%d')}.wav"
        if concatenate_audio_files(audio_files, full_podcast_file):
            # Get file size
            file_size_mb = full_podcast_file.stat().st_size / (1024 * 1024)

            # Update manifest
            podcast_manifest["full_podcast"] = {
                "file": str(full_podcast_file.relative_to(OUTPUT_DIR)),
                "size_mb": round(file_size_mb, 2),
                "track_count": len(audio_files)
            }

            # Save final manifest
            final_manifest_file = OUTPUT_DIR / "podcast-manifest.json"
            with open(final_manifest_file, 'w', encoding='utf-8') as f:
                json.dump(podcast_manifest, f, indent=2)

            print(f"\n✓ Full podcast created!")
            print(f"  File: {full_podcast_file}")
            print(f"  Size: {file_size_mb:.2f} MB")
            print(f"  Tracks: {len(audio_files)}")


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Generate podcast audio from SSML using Silero TTS (local)")
    parser.add_argument("--speaker", default=DEFAULT_SPEAKER, help=f"TTS speaker to use (default: {DEFAULT_SPEAKER})")
    parser.add_argument("--no-concatenate", action="store_true", help="Don't concatenate into single file")

    args = parser.parse_args()

    print("=" * 60)
    print("STT Fine-Tuning Podcast Generator - Silero TTS (Local)")
    print("=" * 60)
    print()

    # Check if ffmpeg is available for concatenation
    if not args.no_concatenate:
        try:
            import subprocess
            subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("Warning: ffmpeg not found. Audio concatenation will not be available.")
            print("Install ffmpeg to enable full podcast generation:")
            print("  sudo apt install ffmpeg")
            args.no_concatenate = True

    process_ssml_directory(speaker=args.speaker, concatenate=not args.no_concatenate)


if __name__ == "__main__":
    main()
