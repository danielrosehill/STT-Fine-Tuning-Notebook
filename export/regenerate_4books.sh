#!/bin/bash

echo "==================================================================="
echo "Speech-to-Text Fine-Tuning Guide - 4-Book Series Generator"
echo "==================================================================="
echo ""

# Step 1: Build markdown files
echo "Step 1: Building markdown files for all 4 books..."
python3 build_4_books.py

if [ $? -ne 0 ]; then
    echo "Error: Failed to build markdown files"
    exit 1
fi

echo ""
echo "Step 2: Converting to HTML and PDF..."
echo ""

# Array of book identifiers
books=("BOOK1" "BOOK2" "BOOK3" "BOOK4")
book_names=(
    "Book 1 - Foundations"
    "Book 2 - Implementation"
    "Book 3 - Specialized Topics"
    "Book 4 - Practical Guide"
)

# Process each book
for i in "${!books[@]}"; do
    book="${books[$i]}"
    name="${book_names[$i]}"

    echo "-------------------------------------------------------------------"
    echo "Processing: $name"
    echo "-------------------------------------------------------------------"

    md_file="STT-Fine-Tuning-Guide-${book}.md"
    html_file="STT-Fine-Tuning-Guide-${book}.html"
    pdf_file="STT-Fine-Tuning-Guide-${book}.pdf"

    if [ ! -f "$md_file" ]; then
        echo "Warning: $md_file not found, skipping..."
        continue
    fi

    # Convert to HTML
    echo "  Converting $md_file to HTML..."
    pandoc "$md_file" \
        -f markdown \
        -t html5 \
        --standalone \
        --css=book-style.css \
        --metadata title="Speech-to-Text Fine-Tuning Guide - $name" \
        -o "$html_file"

    if [ $? -eq 0 ]; then
        echo "  ✓ Created: $html_file"
    else
        echo "  ✗ Failed to create HTML"
        continue
    fi

    # Convert to PDF
    echo "  Converting $html_file to PDF..."
    pandoc "$html_file" \
        -f html \
        -t pdf \
        --pdf-engine=weasyprint \
        --css=book-style.css \
        -o "$pdf_file"

    if [ $? -eq 0 ]; then
        echo "  ✓ Created: $pdf_file"

        # Get file sizes
        html_size=$(du -h "$html_file" | cut -f1)
        pdf_size=$(du -h "$pdf_file" | cut -f1)
        echo "  Sizes: HTML=$html_size, PDF=$pdf_size"
    else
        echo "  ✗ Failed to create PDF"
    fi

    echo ""
done

echo "==================================================================="
echo "Build Complete!"
echo "==================================================================="
echo ""
echo "Generated files:"
ls -lh STT-Fine-Tuning-Guide-BOOK*.{md,html,pdf} 2>/dev/null | awk '{print "  " $9 " (" $5 ")"}'
echo ""
echo "All 4 books are ready for reading and printing!"
