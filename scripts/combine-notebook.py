#!/usr/bin/env python3
"""
Combine all individual markdown files in the notebook directory into a single file.
"""

import os
from pathlib import Path
from collections import defaultdict

def get_section_title(directory_name):
    """Convert directory name to a readable section title."""
    return directory_name.replace('-', ' ').title()

def main():
    # Get the notebook directory
    script_dir = Path(__file__).parent
    notebook_dir = script_dir.parent / 'notebook'
    output_file = notebook_dir / 'combined-notebook.md'

    # Organize files by directory
    files_by_dir = defaultdict(list)

    # Find all markdown files
    for md_file in sorted(notebook_dir.rglob('*.md')):
        # Skip the combined output file if it exists
        if md_file.name == 'combined-notebook.md':
            continue

        # Get relative path from notebook dir
        rel_path = md_file.relative_to(notebook_dir)
        parent_dir = str(rel_path.parent) if rel_path.parent != Path('.') else 'root'

        files_by_dir[parent_dir].append(md_file)

    # Write combined file
    with open(output_file, 'w', encoding='utf-8') as outf:
        outf.write("# STT Fine-Tuning Notebook - Complete Reference\n\n")
        outf.write("This document combines all individual notes from the STT Fine-Tuning Notebook.\n\n")
        outf.write("---\n\n")

        # Process each directory
        for dir_path in sorted(files_by_dir.keys()):
            if dir_path == 'root':
                section_title = "General Notes"
            else:
                # Use the last part of the path for section title
                section_name = Path(dir_path).name
                section_title = get_section_title(section_name)

            outf.write(f"# {section_title}\n\n")

            # Process each file in this directory
            for md_file in sorted(files_by_dir[dir_path]):
                # Write file header
                file_title = md_file.stem.replace('-', ' ').title()
                outf.write(f"## {file_title}\n\n")

                # Write file contents
                with open(md_file, 'r', encoding='utf-8') as inf:
                    content = inf.read()
                    outf.write(content)
                    outf.write("\n\n")

                outf.write("---\n\n")

    print(f"Combined {sum(len(files) for files in files_by_dir.values())} files")
    print(f"Output written to: {output_file}")

if __name__ == '__main__':
    main()
