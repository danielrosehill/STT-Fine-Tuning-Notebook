#!/usr/bin/env python3
"""
Generate podcast audio from SSML files using Google Cloud Text-to-Speech.
Supports Neural2 and Studio voices for high-quality output.
"""

import os
import sys
from pathlib import Path
import json
from datetime import datetime
import re

# Check for Google Cloud TTS library
try:
    from google.cloud import texttospeech
except ImportError:
    print("Error: Google Cloud Text-to-Speech library not installed")
    print("Install with: pip install google-cloud-texttospeech")
    sys.exit(1)

# Configuration
SSML_DIR = Path(__file__).parent.parent / "podcast-ssml"
OUTPUT_DIR = Path(__file__).parent.parent / "podcast-audio"

# Google Cloud TTS Configuration
# Neural2 voices: High quality, natural sounding
# Studio voices: Premium quality with more expressiveness
DEFAULT_VOICE = "en-US-Neural2-F"  # Female Neural2 voice
# Other good options:
# en-US-Neural2-A (male), en-US-Neural2-C (female), en-US-Neural2-D (male)
# en-US-Studio-O (female), en-US-Studio-Q (male)

def strip_ssml_for_gcloud(ssml_content):
    """
    Clean SSML for Google Cloud TTS compatibility.
    Google Cloud TTS supports SSML but has specific requirements.
    """
    # Remove XML declaration if present
    ssml = re.sub(r'<\?xml[^>]*\?>', '', ssml_content)

    # Ensure content is wrapped in <speak> tags
    if not ssml.strip().startswith('<speak'):
        ssml = f'<speak>{ssml}</speak>'

    return ssml.strip()


def convert_ssml_to_audio_gcloud(ssml_file, output_file, voice_name=DEFAULT_VOICE):
    """Convert SSML file to audio using Google Cloud TTS."""
    print(f"Converting {ssml_file.name} to audio...")

    try:
        # Initialize the TTS client
        client = texttospeech.TextToSpeechClient()

        # Read SSML content
        with open(ssml_file, 'r', encoding='utf-8') as f:
            ssml_content = f.read()

        # Clean SSML for Google Cloud
        ssml_text = strip_ssml_for_gcloud(ssml_content)

        # Set the text input to be synthesized
        synthesis_input = texttospeech.SynthesisInput(ssml=ssml_text)

        # Build the voice request
        voice = texttospeech.VoiceSelectionParams(
            language_code="en-US",
            name=voice_name
        )

        # Select the type of audio file
        audio_config = texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.MP3,
            speaking_rate=1.0,  # Normal speed
            pitch=0.0,  # Normal pitch
        )

        # Perform the text-to-speech request
        response = client.synthesize_speech(
            input=synthesis_input,
            voice=voice,
            audio_config=audio_config
        )

        # Write the response to the output file
        with open(output_file, 'wb') as out:
            out.write(response.audio_content)

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


def process_ssml_directory(voice_name=DEFAULT_VOICE, concatenate=True):
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
    print(f"Using voice: {voice_name}")

    # Create podcast manifest
    podcast_manifest = {
        "generated_at": datetime.now().isoformat(),
        "tts_engine": "google-cloud-tts",
        "voice": voice_name,
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
        audio_file = output_subdir / f"{ssml_file.stem}.mp3"

        print(f"\n[{i}/{len(ssml_files)}] Processing: {rel_path}")

        # Convert to audio
        success = convert_ssml_to_audio_gcloud(ssml_file, audio_file, voice_name)

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

        full_podcast_file = OUTPUT_DIR / f"stt-finetune-podcast-{datetime.now().strftime('%Y%m%d')}.mp3"
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


def list_available_voices():
    """List available Google Cloud TTS voices."""
    try:
        client = texttospeech.TextToSpeechClient()

        # Performs the list voices request
        voices = client.list_voices()

        print("\nAvailable English voices:\n")
        print(f"{'Voice Name':<30} {'Gender':<10} {'Type':<15}")
        print("-" * 55)

        for voice in voices.voices:
            for language_code in voice.language_codes:
                if language_code.startswith('en-'):
                    voice_type = "Neural2" if "Neural2" in voice.name else \
                                "Studio" if "Studio" in voice.name else \
                                "Standard" if "Standard" in voice.name else \
                                "WaveNet" if "Wavenet" in voice.name else "Other"

                    gender = "Male" if voice.ssml_gender == 1 else \
                            "Female" if voice.ssml_gender == 2 else "Neutral"

                    print(f"{voice.name:<30} {gender:<10} {voice_type:<15}")
                    break

    except Exception as e:
        print(f"Error listing voices: {e}")
        print("\nMake sure you have:")
        print("1. Installed: pip install google-cloud-texttospeech")
        print("2. Set up authentication: export GOOGLE_APPLICATION_CREDENTIALS=path/to/key.json")


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Generate podcast audio from SSML using Google Cloud TTS")
    parser.add_argument("--voice", default=DEFAULT_VOICE, help=f"TTS voice to use (default: {DEFAULT_VOICE})")
    parser.add_argument("--list-voices", action="store_true", help="List available voices")
    parser.add_argument("--no-concatenate", action="store_true", help="Don't concatenate into single file")

    args = parser.parse_args()

    print("=" * 60)
    print("STT Fine-Tuning Podcast Generator - Google Cloud TTS")
    print("=" * 60)
    print()

    if args.list_voices:
        list_available_voices()
        return

    # Check for Google Cloud credentials
    if not os.getenv("GOOGLE_APPLICATION_CREDENTIALS"):
        print("Warning: GOOGLE_APPLICATION_CREDENTIALS not set")
        print("Set it with: export GOOGLE_APPLICATION_CREDENTIALS=path/to/key.json")
        print("Or use: gcloud auth application-default login")
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

    process_ssml_directory(voice_name=args.voice, concatenate=not args.no_concatenate)


if __name__ == "__main__":
    main()
