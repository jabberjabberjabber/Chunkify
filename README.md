# Chunkify: A Python Script for Text Processing with Large Language Models

## Overview

Chunkify was made as a proof-of-concept for a chunking method that doesn't rely on a tokenizer. The text processing features were added because they are commonly used and useful. Aya Expanse is particularly good at these tasks and is downloaded automatically the first time you run it if you use the batch file.

## Key Features

- **Document Chunking:** Divides large documents into manageable chunks, intelligently identifying breaks based on chapters, headings, paragraphs, or sentences.
- **Automatic Template Selection:** Adapts to the loaded model, ensuring the correct instruction template is used.
- **Real-time Monitoring:** Provides continuous feedback on the generation process, allowing users to track progress.
- **Compatible with multiple document formats, including PDF and HTML**
- **Multiple Processing Modes:**
  - **Summary:** Generates concise summaries of the content.
  - **Translation:** Translates text into a language you can specify (default is English).
  - **Distillation:** Rewrites content for conciseness while retaining key information.
  - **Correction:** Fixes grammar, spelling, and style issues.
- **File Output Support:** Saves results to specified output files.

## Requirements

- Python 3.8 or later
- KoboldCpp executable in the script directory
- Essential Python packages:
  - `requests`
  - `extractous` (for text extraction)
  - `PyQt6` (for GUI)

## Installation

#### Windows Installation:

1. Clone the repository or download the ZIP file from GitHub.
2. Install Python 3.8 or later if not already present.
3. Download KoboldCPP.exe from the [KoboldCPP releases](https://github.com/LostRuins/koboldcpp/releases) page and place it in the project folder.
4. Run `chunkify-run.bat`. This script will install necessary dependencies and download the Aya Expanse 8b Q6_K model.
5. Upon completion, you should see a message: "Please connect to custom endpoint at http://localhost:5001".

#### macOS Installation:

1. Follow the Windows installation steps, ensuring you use the appropriate KoboldCPP binary for macOS.

#### Linux Installation:

1. Similar to Windows, clone the repository, install Python 3.8 or later, and download the Linux KoboldCPP binary from the releases page.
2. Run the script using: `./chunkify-run.sh`.

## Usage

1. **GUI Launch:**
   - Windows: Run `chunkify-run.bat`.
   - macOS/Linux: Execute `python3 chunkify-gui.py`.

2. Ensure KoboldCPP is running and displaying the message: "Please connect to custom endpoint at http://localhost:5001".

3. Configure settings and API details through the GUI or a configuration JSON file.

4. Click "Process" to initiate the text processing task.

5. Monitor progress in the GUI's output area.

## Configuration

Configuration can be managed through:

- Command-line arguments
- `chunkify_config.json` file
- GUI settings

## Command-Line Usage

```bash
python chunkify.py --content input.txt --task summary
```

or with a configuration file:

```bash
python chunkify.py --config config.json --content input.txt --task translate
```

## Output Format

When using the `--file` option, the script generates a Markdown-formatted output file containing:

- Document metadata
- Task-specific results

The default output file is `output.txt` in the script directory, or the GUI will save files with an added '_processed' suffix.


## Limitations

- Context length is model-dependent.
- Chunking and generation length are set to half the context size.
- Speed varies based on API response time.
- Consider a GPU with 8GB VRAM or a powerful CPU with 16GB RAM for optimal performance.

## Contribution and License

Feel free to contribute and submit issues or pull requests. The script is licensed under the MIT license.

