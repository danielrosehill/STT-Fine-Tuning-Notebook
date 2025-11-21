#!/usr/bin/env python3
"""
Build 4-part PDF book series from notebook markdown files
Divides content into logical volumes for easier reading and printing
"""

import os
from pathlib import Path
from typing import Dict, List
import re

# Define topic organization with book assignments
BOOK_ORGANIZATION = {
    'book1': {
        'title': 'Book 1: Foundations',
        'subtitle': 'Background, Models & Data Preparation',
        'topics': ['background-context', 'models', 'data-preparation']
    },
    'book2': {
        'title': 'Book 2: Implementation',
        'subtitle': 'Fine-Tuning, Inference & Hardware Optimization',
        'topics': ['fine-tuning', 'inference', 'amd']
    },
    'book3': {
        'title': 'Book 3: Specialized Topics',
        'subtitle': 'Mobile ASR, File Formats & Vocabulary',
        'topics': ['mobile-asr', 'formats', 'vocab']
    },
    'book4': {
        'title': 'Book 4: Practical Guide',
        'subtitle': 'Pitfalls, Q&A & Additional Notes',
        'topics': ['pitfalls', 'q-and-a', 'notes']
    }
}

# Define topic details
TOPIC_INFO = {
    'background-context': {
        'title': 'Part I: Background & Context',
        'description': 'Historical context and evolution of ASR technology'
    },
    'models': {
        'title': 'Part II: ASR Models',
        'description': 'Overview and comparison of ASR models'
    },
    'data-preparation': {
        'title': 'Part III: Data Preparation',
        'description': 'Audio data preparation and dataset creation'
    },
    'fine-tuning': {
        'title': 'Part IV: Fine-Tuning',
        'description': 'Fine-tuning strategies and techniques'
    },
    'inference': {
        'title': 'Part V: Inference & Deployment',
        'description': 'Running and deploying ASR models'
    },
    'amd': {
        'title': 'Part VI: AMD GPU Optimization',
        'description': 'AMD-specific hardware considerations'
    },
    'mobile-asr': {
        'title': 'Part VII: Mobile ASR',
        'description': 'Mobile and edge device deployment'
    },
    'formats': {
        'title': 'Part VIII: File Formats',
        'description': 'Audio and model file formats'
    },
    'vocab': {
        'title': 'Part IX: Vocabulary & Language',
        'description': 'Vocabulary recognition and language considerations'
    },
    'pitfalls': {
        'title': 'Part X: Common Pitfalls',
        'description': 'Common issues and how to avoid them'
    },
    'q-and-a': {
        'title': 'Part XI: Q&A',
        'description': 'Frequently asked questions'
    },
    'notes': {
        'title': 'Part XII: Additional Notes',
        'description': 'Supplementary topics and observations'
    }
}


def clean_title(filename: str) -> str:
    """Convert filename to readable title"""
    title = filename.replace('.md', '').replace('-', ' ')
    return ' '.join(word.capitalize() for word in title.split())


def collect_files() -> Dict[str, List[Path]]:
    """Collect all markdown files organized by topic"""
    notebook_dir = Path('../notebook')
    organized = {topic: [] for topic in TOPIC_INFO.keys()}

    for topic_dir in notebook_dir.iterdir():
        if topic_dir.is_dir() and topic_dir.name in TOPIC_INFO:
            md_files = sorted(topic_dir.glob('*.md'))
            organized[topic_dir.name] = md_files

    return organized


def generate_book_toc(book_key: str, organized_files: Dict[str, List[Path]]) -> str:
    """Generate table of contents for a specific book"""
    book_info = BOOK_ORGANIZATION[book_key]

    toc = [
        "# Speech-to-Text Fine-Tuning Guide",
        "",
        f"## {book_info['title']}",
        "",
        f"_{book_info['subtitle']}_",
        "",
        "---",
        "",
        "## Table of Contents",
        ""
    ]

    for topic_key in book_info['topics']:
        if organized_files.get(topic_key):
            topic = TOPIC_INFO[topic_key]
            chapter_count = len(organized_files[topic_key])
            toc.append(f"**{topic['title']}**  ")
            toc.append(f"{topic['description']} ({chapter_count} chapters)")
            toc.append("")

    toc.append("---")
    toc.append("")

    return '\n'.join(toc)


def read_file_content(file_path: Path) -> str:
    """Read and return file content"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        return content.strip()
    except Exception as e:
        return f"<!-- Error reading {file_path}: {e} -->"


def build_book_document(book_key: str, organized_files: Dict[str, List[Path]]) -> str:
    """Build a complete book document"""
    parts = []
    book_info = BOOK_ORGANIZATION[book_key]

    # Add TOC
    parts.append(generate_book_toc(book_key, organized_files))

    # Add content by topic
    for topic_key in book_info['topics']:
        if not organized_files.get(topic_key):
            continue

        topic = TOPIC_INFO[topic_key]

        # Topic section header (h1 - will get page break from CSS)
        parts.append("")
        parts.append(f"# {topic['title']}")
        parts.append("")
        parts.append(f"_{topic['description']}_")
        parts.append("")
        parts.append("---")
        parts.append("")

        # Add each chapter in this topic
        for file_path in organized_files[topic_key]:
            chapter_title = clean_title(file_path.name)

            # Chapter header (h2 - no page break)
            parts.append("")
            parts.append(f"## {chapter_title}")
            parts.append("")

            # Chapter content
            content = read_file_content(file_path)

            # Remove any existing top-level headers to avoid conflicts
            content = re.sub(r'^#\s+.*$', '', content, flags=re.MULTILINE)

            parts.append(content)
            parts.append("")

    return '\n'.join(parts)


def main():
    """Main execution"""
    print("Building Speech-to-Text Fine-Tuning Guide (4-Book Series)...")

    # Collect files
    print("Collecting markdown files...")
    organized_files = collect_files()

    total_files = sum(len(files) for files in organized_files.values())
    print(f"Found {total_files} files across {len(TOPIC_INFO)} topics")

    # Build each book
    for book_key in BOOK_ORGANIZATION.keys():
        book_info = BOOK_ORGANIZATION[book_key]
        print(f"\nBuilding {book_info['title']}...")

        # Build document
        book_content = build_book_document(book_key, organized_files)

        # Write markdown
        output_md = Path(f'STT-Fine-Tuning-Guide-{book_key.upper()}.md')
        print(f"  Writing to {output_md}...")
        with open(output_md, 'w', encoding='utf-8') as f:
            f.write(book_content)

        print(f"  ✓ Created: {output_md}")
        print(f"    Size: {len(book_content):,} characters")

    print("\n" + "="*60)
    print("All books created successfully!")
    print("\nNext: Run ./regenerate_4books.sh to convert to HTML and PDF")


if __name__ == '__main__':
    main()
