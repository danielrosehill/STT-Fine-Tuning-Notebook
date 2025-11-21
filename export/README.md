# STT Fine-Tuning Guide - Export

This directory contains the exported PDF book version of the Speech-to-Text Fine-Tuning notebook.

## Available Formats

### Complete Guide (Single Volume)
- **STT-Fine-Tuning-Guide.pdf** - The complete book in PDF format (401 pages)
- **STT-Fine-Tuning-Guide.md** - Master markdown document
- **STT-Fine-Tuning-Guide.html** - HTML version with styling

### 4-Book Series (For Easier Reading & Printing)
Each book covers a logical grouping of topics with professional styling:

- **Book 1: Foundations** (422 KB PDF)
  - Part I: Background & Context
  - Part II: ASR Models
  - Part III: Data Preparation

- **Book 2: Implementation** (419 KB PDF)
  - Part IV: Fine-Tuning
  - Part V: Inference & Deployment
  - Part VI: AMD GPU Optimization

- **Book 3: Specialized Topics** (165 KB PDF)
  - Part VII: Mobile ASR
  - Part VIII: File Formats
  - Part IX: Vocabulary & Language

- **Book 4: Practical Guide** (247 KB PDF)
  - Part X: Common Pitfalls
  - Part XI: Q&A
  - Part XII: Additional Notes

## Build Scripts

- **build_book.py** - Generates single complete guide
- **build_4_books.py** - Generates 4-book series
- **regenerate.sh** - Rebuilds complete guide
- **regenerate_4books.sh** - Rebuilds all 4 books
- **book-style.css** - Custom CSS for professional PDF styling

## Book Features

✓ **Compact Table of Contents** - 12 thematic parts with chapter counts (not individual chapters)
✓ **Page Numbers** - Footer displays page numbers on every page (except title page)
✓ **Professional Typography** - Georgia serif font for readability, Helvetica for headings
✓ **Smart Page Breaks** - Only major sections (Parts) start new pages, chapters flow continuously
✓ **Organized by Topic** - Content structured in logical progression:
  - Part I: Background & Context
  - Part II: ASR Models
  - Part III: Data Preparation
  - Part IV: Fine-Tuning
  - Part V: Inference & Deployment
  - Part VI: AMD GPU Optimization
  - Part VII: Mobile ASR
  - Part VIII: File Formats
  - Part IX: Vocabulary & Language
  - Part X: Common Pitfalls
  - Part XI: Q&A
  - Part XII: Additional Notes

✓ **Page Breaks** - Clean separation between chapters and sections
✓ **Readable Layout** - Justified text, proper margins (2.5cm top, 2cm sides, 3cm bottom)
✓ **Code Highlighting** - Syntax-friendly formatting for code blocks
✓ **Footer Metadata** - Document title in footer for easy reference

## Regenerating the Books

### Complete Guide
```bash
./regenerate.sh
```

### 4-Book Series
```bash
./regenerate_4books.sh
```

Both scripts will:
1. Build markdown files from notebook content
2. Convert to HTML with styling
3. Generate PDFs using pandoc and WeasyPrint

## Statistics

- **Total Chapters**: 41
- **Total Parts**: 12
- **Document Size**: ~488,000 characters
- **PDF Size**: 1.1 MB (401 pages)
- **Format**: A4, professional book layout
- **TOC**: Compact (parts only, not individual chapters)

---

Generated using pandoc and WeasyPrint with custom CSS styling.
