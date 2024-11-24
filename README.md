# Chunkify

A Python script for processing text through Large Language Models (LLMs) via the Kobold API. The script supports chunking large documents, handling various instruction templates, and multiple processing tasks including summarization, translation, text distillation, and correction.

This script was created as a proof-of-concept for a non-tokenizer based chunker that could stop at natural text breaks. It is the most basic form of chunking text in that only regex is used along with a basic size limiter based on a 1.5:1 word:token conversion. 

The processing tasks were added to give the script some utility, and it should provide a good basis for all sort of simple text processing functions using LLMs.

KoboldCpp is a one-file, cross-platform inference engine with a built in Web GUI and API, and will serve language models and vision models, as well as image diffusion models. It is based on Llama.cpp and uses GGUF model weights.

For fast processing, I recommend the following model weights:
- Llama 3.2 3b Instruct (Q6_K)
- Phi 3.5 Mini Instruct (Q6_K)
- Qwen 2.5 3b Instruct (Q6_K)

Enable 'Flash Attention'!

**Make sure that the model's filename has the name of the base model in it! Otherwise it won't know which template to use!**

Good Example: `qwen2.5-3b-instruct-q6_k.gguf`

Bad Example: `finetuned-3b-q6_k.gguf` <-- MUST HAVE ENTRY IN APPROPRIATE ADAPTER IN TEMPLATES

If you have a model without the base name in the filename, edit the appropriate adapter in the templates folder and add part of the filename to the "aka" key

This script requires Python but needs no external libraries to be installed!

![Real time chunking and processing of the Alice in Wonderland in its entirety, taking 1m02s with Llama 3.2 3B at 4096 context on a 3080 10GB](./screen.webp)
*Realtime processing of the entire text of Alice in Wonderland in ~1 minute using consumer hardware*

## Features

- Document chunking with intelligent break points (chapters, headings, paragraphs, sentences)
- Automatic template selection based on loaded model
- Real-time generation monitoring
- Multiple processing modes:
  - Summary: Creates detailed summary with chunk-by-chunk analysis
  - Translation: Translates content to English
  - Distillation: Rewrites content more concisely
  - Correction: Fixes grammar, spelling, and style issues
- Support for custom instruction templates
- Progress visualization during generation
- File output support with formatted results

## Requirements

- Python 3.7+
- A running instance of KoboldCpp API
- Required Python packages:
  ```
  requests
  dataclasses (included in Python 3.7+)
  ```

## Configuration

The script can be configured either through command-line arguments or a JSON configuration file.

### Configuration File Format
```json
{
    "templates_directory": "./templates",
    "api_url": "http://localhost:5001",
    "api_password": "your_password",
    "text_completion": false,
    "gen_count": 500,
    "temp": 0.7,
    "rep_pen": 1.0,
    "min_p": 0.2
}
```

### Command Line Arguments

```
--config       Path to JSON config file
--instruction  System instruction for processing
--content      Content to process (file path)
--api-url      URL for the LLM API (default: http://localhost:5001)
--api-password Password for the LLM API
--templates    Directory for instruct templates (default: ./templates)
--task         Task to perform: summary, translate, distill, or correct
--file         Output file path (optional)
```

## Usage Examples

1. Basic usage with default settings:
```bash
python process.py --content input.txt --task summary
```

2. Using a config file:
```bash
python process.py --config config.json --content input.txt --task translate
```

3. With custom instruction and output file:
```bash
python process.py --content input.txt --task distill --instruction "Focus on technical details" --file output.md
```

## Output Format

When using the `--file` option, the script generates a Markdown formatted file containing:

- Document metadata (title, type, subject, structure)
- Task-specific results:
  - For summaries: Individual chunk responses and final summary
  - For other tasks: Complete processed content


If you do not specify an output file, the output will be written to `output.txt` in the script directory.

## Template System

The script uses a template system for different LLM instruction formats. Templates are JSON files stored in the templates directory with the following structure:

```json
{
    "name": ["template_name"],
    "akas": ["alternative_names"],
    "system_start": "### System:",
    "system_end": "\n",
    "user_start": "### Human:",
    "user_end": "\n",
    "assistant_start": "### Assistant:",
    "assistant_end": "\n"
}
```

By default we use the templates included in the KoboldCpp repo under `/kcpp_adapters`

## Limitations

- Maximum context length is determined by the loaded model
- Processing speed depends on the API response time
- Templates must match the format expected by the LLM

## Breakdown

This is what the script does:

1. Looks for configuration, if not found will use default
2. Calls the Kobold API and asks for the name of the running model, then parses out the most likely instruct template based on that name and loads the appropriate JSON adapter
3. Calls the Kobold API and asks for the max context length, then cuts that in half, converts that approximately to words and sets that as max_size
4. Takes the first 1000 words from the content and sends it to the model and asks for a structured response with metadata, including title and document type
5. Uses the regex in chunker.py to find break points in the content, then matches one of those points to the largest piece that will fit in the max size, and continues until the content is chunked
6. Depending on the task, sends the chunks to the model with prompts directing it to perform an action on them
7. In a separate thread, queries the API continually asking for the partial generation results and outputs them to the console
8. Combines the responses and the structured metadata into a text file and saves it

## Contributing

Feel free to submit issues and enhancement requests!

## License

[Your chosen license here]
