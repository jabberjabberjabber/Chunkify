import argparse
import asyncio
import json
import os
import random
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Iterator, Union, Any
import requests
from requests.exceptions import RequestException

from extractous import Extractor
from chunker_regex import chunk_regex


class ChunkingProcessor:
    """ Handles splitting content into manageable chunks using natural breaks """

    def __init__(self, api_url: str, 
                 max_chunk_length: int,
                 api_password: Optional[str] = None,
                 max_total_chunks: int = 1000):
        """ Initialize the chunking processor
        
        Args:
            api_url: URL to the KoboldAPI server
            max_chunk_length: Maximum token length for a single chunk
            api_password: Optional API password/key
            max_total_chunks: Maximum number of chunks to process
        """
        if max_chunk_length <= 0:
            raise ValueError("max_chunk_length must be positive")
            
        self.api_url = api_url
        self.max_chunk = max_chunk_length
        self.max_total_chunks = max_total_chunks
        self.api_password = api_password
        self.headers = {
            "Content-Type": "application/json",
        }
        
        if api_password:
            self.headers["Authorization"] = f"Bearer {api_password}"
            
        # Generate a unique key for this processing session
        self.genkey = self._create_genkey()
        
        # Verify API and get max context length if needed
        self.api_max_context = self._get_max_context_length()
        if self.max_chunk > self.api_max_context // 2:
            print(f"Warning: Reducing chunk size to fit model context window")
            self.max_chunk = self.api_max_context // 2
    
    def _create_genkey(self) -> str:
        """ Create a unique generation key to prevent cross-request contamination """
        return f"KCPP{''.join(str(random.randint(0, 9)) for _ in range(4))}"
        
    def _get_max_context_length(self) -> int:
        """ Get the maximum context length from the KoboldAPI """
        try:
            response = requests.get(f"{self.api_url}/api/extra/true_max_context_length")
            if response.status_code == 200:
                max_context = int(response.json().get("value", 8192))
                print(f"Model has maximum context length of: {max_context}")
                return max_context
            else:
                print(f"Warning: Could not get max context length. Defaulting to 8192")
                return 8192
        except Exception as e:
            print(f"Error getting max context length: {str(e)}. Defaulting to 8192")
            return 8192
            
    def count_tokens(self, text: str) -> int:
        """ Count tokens in the provided text using KoboldAPI """
        try:
            payload = {"prompt": text, "genkey": self.genkey}
            response = requests.post(
                f"{self.api_url}/api/extra/tokencount",
                json=payload,
                headers=self.headers
            )
            if response.status_code == 200:
                return int(response.json().get("value", 0))
            else:
                # Fallback estimation
                return len(text.split()) 
        except Exception as e:
            print(f"Error counting tokens: {str(e)}. Using word count as estimate.")
            return len(text.split())
    
    def chunk_text(self, content: str) -> List[Tuple[str, int]]:
        """ Split content into chunks using natural breakpoints
        
        Args:
            content: The text content to chunk
            
        Returns:
            List of (chunk_text, token_count) tuples
        """
        if not content:
            return []
            
        chunks = []
        remaining = content
        chunk_num = 0
        
        while remaining and chunk_num < self.max_total_chunks:
            # KoboldCPP has max char limit of 50k
            current_section = remaining[:45000]
            remaining = remaining[45000:]
            
            chunk = self._get_chunk(current_section)
            chunk_len = len(chunk)
            
            if chunk_len == 0:
                continue
                
            chunk_tokens = self.count_tokens(chunk)
            chunks.append((chunk, chunk_tokens))
            
            # Update remaining with what wasn't included in this chunk
            remaining = current_section[len(chunk):].strip() + remaining

            chunk_num += 1
            print(f"Created chunk {chunk_num}: {chunk_tokens} tokens")
            
        if remaining and chunk_num >= self.max_total_chunks:
            raise ValueError(f"Text exceeded maximum of {self.max_total_chunks} chunks")
            
        return chunks

    def _get_chunk(self, content: str) -> str:
        """ Get appropriately sized chunk using natural breaks
        
        Args:
            content: Text content to chunk
            
        Returns:
            A chunk of text within token limits
        """
        total_tokens = self.count_tokens(content)
        if total_tokens < self.max_chunk:
            return content

        # chunk_regex is designed to break at natural language points
        # to preserve context and readability
        matches = chunk_regex.finditer(content)
        current_size = 0
        chunks = []
        
        for match in matches:
            chunk = match.group(0)
            chunk_size = self.count_tokens(chunk)
            if current_size + chunk_size > self.max_chunk:
                if not chunks:
                    chunks.append(chunk)
                break
            chunks.append(chunk)
            current_size += chunk_size
        
        return ''.join(chunks)

    def chunk_file(self, file_path) -> Tuple[List[Tuple[str, int]], Dict]:
        """ Chunk text from file
        
        Args:
            file_path: Path to text file (str or Path object)
            
        Returns:
            Tuple of (chunks with token counts, file metadata)
        """
        extractor = Extractor()
        extractor = extractor.set_extract_string_max_length(100000000)
        
        try:
            content, metadata = extractor.extract_file_to_string(str(file_path))
            chunks = self.chunk_text(content)
            return chunks, metadata
        except Exception as e:
            print(f"Error extracting file: {str(e)}")
            return [], {"error": str(e)}


class SSEProcessingClient:
    """ Client for processing chunks with OpenAI-compatible endpoints via SSE streaming """
    
    def __init__(self, api_url: str, api_password: Optional[str] = None):
        """ Initialize the processing client
        
        Args:
            api_url: URL to the OpenAI-compatible API
            api_password: Optional API key/password
        """
        self.api_url = api_url
        self.api_password = api_password
        
        # Ensure API URL ends with correct endpoint for OpenAI compatibility
        if not self.api_url.endswith('/v1/chat/completions'):
            self.api_url = f"{self.api_url.rstrip('/')}/v1/chat/completions"
            
        self.headers = {
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
        }
        
        if api_password:
            self.headers["Authorization"] = f"Bearer {api_password}"
    
    def _create_payload(self, instruction: str, content: str, 
                       max_tokens: int = 2048,
                       temperature: float = 0.2, 
                       top_p: float = 1.0,
                       top_k: int = 0,
                       rep_pen: float = 1.0,
                       min_p: float = 0.05) -> Dict:
        """ Create the API payload with standard parameters
        
        Args:
            instruction: Instruction text for the model
            content: Content text to process
            max_tokens: Maximum tokens to generate
            temperature: Temperature parameter
            top_p: Top-p parameter
            top_k: Top-k parameter
            rep_pen: Repetition penalty
            min_p: Minimum p parameter
            
        Returns:
            Dictionary payload for the API
        """
        system_content = "You are a helpful assistant."
        
        combined_content = f"<START_TEXT>{content}<END_TEXT>\n{instruction}"
        
        return {
            "messages": [
                {"role": "system", "content": system_content},
                {"role": "user", "content": combined_content}
            ],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "repetition_penalty": rep_pen,
            "min_p": min_p,
            "stream": True
        }
    
    async def process_chunk(self, instruction: str, content: str,
                         max_tokens: int = 2048,
                         temperature: float = 0.2) -> str:
        """ Process a single chunk with streaming output
        
        Args:
            instruction: Instruction for processing
            content: Text content to process
            max_tokens: Maximum tokens to generate
            temperature: Generation temperature
            
        Returns:
            Generated text
        """
        payload = self._create_payload(
            instruction=instruction,
            content=content,
            max_tokens=max_tokens,
            temperature=temperature
        )
        
        result = []
        partial_line = ""
        
        try:
            response = requests.post(
                self.api_url,
                json=payload,
                headers=self.headers,
                stream=True
            )
            
            for line in response.iter_lines():
                if not line:
                    continue
                    
                # Remove the "data: " prefix and decode
                line_text = line.decode('utf-8')
                
                if line_text.startswith('data: '):
                    line_text = line_text[6:]
                
                # Handle the "[DONE]" message
                if line_text == '[DONE]':
                    break
                    
                # Parse the JSON from the SSE stream
                try:
                    data = json.loads(line_text)
                    
                    # Extract the delta content from the received data
                    if 'choices' in data and len(data['choices']) > 0:
                        if 'delta' in data['choices'][0]:
                            if 'content' in data['choices'][0]['delta']:
                                token = data['choices'][0]['delta']['content']
                                # Handle any newlines in the token
                                if token.endswith('\n'):
                                    partial_line += token[:-1]
                                    print(partial_line)
                                    partial_line = ""
                                elif '\n' in token:
                                    parts = token.split('\n')
                                    for i, part in enumerate(parts):
                                        if i < len(parts) - 1:
                                            print(partial_line + part)
                                            partial_line = ""
                                        else:
                                            partial_line += part
                                else:
                                    partial_line += token
                                
                                result.append(token)
                except json.JSONDecodeError:
                    continue
                
            if partial_line:
                print(partial_line)
                
            return ''.join(result)
                
        except Exception as e:
            print(f"Error in API call: {str(e)}")
            return ""


class TextProcessor:
    """ Handles processing of text documents with KoboldAPI """
    
    def __init__(self, api_url: str, 
                 max_chunk_size: int = 4096,
                 api_password: Optional[str] = None):
        """ Initialize the text processor
        
        Args:
            api_url: URL to the KoboldAPI server
            max_chunk_size: Maximum chunk size in tokens
            api_password: Optional API key/password
        """
        self.api_url = api_url
        self.api_password = api_password
        self.task_configs = {
            'translate': {
                'chunk_ratio': 0.45,
                'instruction_template': (
                    "Translate the text into {language}. "
                    "Maintain linguistic flourish and authorial style."
                )
            },
            'summary': {
                'chunk_ratio': 0.8,
                'instruction_template': (
                    "Extract the key points and themes from the text "
                    "without developing conclusions. "
                    "Be thorough but concise."
                )
            },
            'correct': {
                'chunk_ratio': 0.45,
                'instruction_template': (
                    "Correct any grammar, spelling, and style errors. "
                    "Preserve the original meaning and style."
                )
            },
            'distill': {
                'chunk_ratio': 0.8,
                'instruction_template': (
                    "Rewrite the text to be concise without losing meaning."
                )
            }
        }
        
        # Initialize the chunking processor
        self.chunker = ChunkingProcessor(
            api_url=api_url,
            max_chunk_length=max_chunk_size,
            api_password=api_password
        )
        
        # Use SSE processing for streaming output
        self.processor = SSEProcessingClient(
            api_url=api_url,
            api_password=api_password
        )
        
    def _get_task_config(self, task: str, language: str = "English") -> dict:
        """ Get configuration for specified task
        
        Args:
            task: Processing task name
            language: Target language for translation
            
        Returns:
            Task configuration dictionary
        """
        if task not in self.task_configs:
            raise ValueError(f"Unknown task: {task}")
            
        config = self.task_configs[task].copy()
        config['chunk_size'] = int(self.chunker.api_max_context * config['chunk_ratio'])
        config['instruction'] = config['instruction_template'].format(
            language=language
        )
        return config
    
    async def process_text(self, task: str, file_path: Union[str, Path], 
                         language: str = "English") -> Tuple[List[str], Dict]:
        """ Process text document with specified task
        
        Args:
            task: Processing task name
            file_path: Path to the text file
            language: Target language for translation
            
        Returns:
            Tuple of (results, metadata)
        """
        task_config = self._get_task_config(task, language)
        
        # Update the chunker's max size based on the task
        self.chunker.max_chunk = task_config['chunk_size']
        
        # Chunk the file
        chunks, metadata = self.chunker.chunk_file(file_path)
        
        print(f"\nProcessing {len(chunks)} chunks...")
        results = []
        
        for i, (chunk, tokens) in enumerate(chunks, 1):
            print(f"\nChunk {i}/{len(chunks)} ({tokens} tokens):")
            try:
                result = await self.processor.process_chunk(
                    instruction=task_config['instruction'],
                    content=chunk
                )
                results.append(result)
            except Exception as e:
                print(f"\nError processing chunk {i}: {e}")
                results.append(f"[Error processing chunk {i}]")
                
        # Update metadata
        metadata.update({
            'processing_time': datetime.now().isoformat(),
            'task': task,
            'chunks_processed': len(chunks),
            'language': language
        })
        
        return results, metadata


def write_output(output_path: str, results: List[str], metadata: Dict) -> None:
    """ Write processing results to a file
    
    Args:
        output_path: Path to output file
        results: List of processed text chunks
        metadata: Metadata about the processing
    """
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(f"File: {metadata.get('resourceName', 'Unknown')}\n")
            f.write(f"Type: {metadata.get('Content-Type', 'Unknown')}\n")
            f.write(f"Task: {metadata.get('task', 'Unknown')}\n")
            f.write(f"Processed: {metadata.get('processing_time', 'Unknown')}\n")
            f.write(f"Chunks: {metadata.get('chunks_processed', 0)}\n\n")
            
            for i, result in enumerate(results, 1):
                f.write(f"--- Chunk {i} ---\n\n")
                f.write(f"{result}\n\n")
                
        print(f"\nOutput written to: {output_path}")
    except Exception as e:
        print(f"Error writing output file: {str(e)}")


async def process_file(api_url: str, input_path: Path, task: str, 
                      output_path: Optional[str] = None,
                      language: str = "English", 
                      max_chunk_size: int = 4096,
                      api_password: Optional[str] = None) -> int:
    """ Process a text file and save results
    
    Args:
        api_url: URL to the KoboldAPI server
        input_path: Path to input file
        task: Processing task name
        output_path: Path for output file (defaults to input + task)
        language: Target language for translation
        max_chunk_size: Maximum chunk size in tokens
        api_password: Optional API key/password
        
    Returns:
        Exit code (0 for success, 1 for error)
    """
    if not output_path:
        input_stem = Path(input_path).stem
        output_path = f"{input_stem}_{task}.txt"
        
    try:
        processor = TextProcessor(
            api_url=api_url,
            max_chunk_size=max_chunk_size,
            api_password=api_password
        )
        
        results, metadata = await processor.process_text(task, input_path, language)
        
        write_output(output_path, results, metadata)
        print("\nProcessing complete.")
        return 0
        
    except Exception as e:
        print(f"Error: {e}")
        return 1


async def main():
    parser = argparse.ArgumentParser(
        description="Process text documents with KoboldAPI"
    )
    parser.add_argument(
        'input',
        type=str,
        help='Input text file path'
    )
    parser.add_argument(
        '--task',
        required=True,
        choices=['summary', 'translate', 'correct', 'distill'],
        help='Processing task to perform'
    )
    parser.add_argument(
        '--api-url',
        default='http://localhost:5001',
        help='KoboldAPI URL'
    )
    parser.add_argument(
        '--api-password',
        default='',
        help='API key/password'
    )
    parser.add_argument(
        '--language',
        default='English',
        help='Target language for translation'
    )
    parser.add_argument(
        '--output',
        default=None,
        help='Output file path'
    )
    parser.add_argument(
        '--max-chunk-size',
        default=4096,
        type=int,
        help='Maximum token size for a chunk'
    )
    
    args = parser.parse_args()
    
    return await process_file(
        api_url=args.api_url,
        input_path=args.input,
        task=args.task,
        output_path=args.output,
        language=args.language,
        max_chunk_size=args.max_chunk_size,
        api_password=args.api_password
    )


if __name__ == '__main__':
    result = asyncio.run(main())
    sys.exit(result)
