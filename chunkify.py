import os
import re
import json
import random
import requests
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Optional, Union
from urllib.parse import urlparse
from chunker_regex import chunk_regex 
from clean_json import clean_json 
import threading
import time
from extractous import Extractor

def clearConsole():
    """ Borrowed this from somewhere else, hence the casing
    """
    command = 'clear'
    if os.name in ('nt', 'dos'):
        command = 'cls'
    os.system(command)
    
@dataclass
class LLMConfig:
    """ Configuration for LLM processing
    """
    templates_directory: str
    api_url: str
    api_password: str
    text_completion: bool = False
    gen_count: int = 500 #not used, set to 1/2 max_context
    temp: float = 0.0
    rep_pen: float = 1.05
    min_p: float = 0.02
    top_k: int = 1
    top_p: int = 1
    summary_instruction="Extract the key points, themes and actions from the text succinctly without developing any conclusions or commentary."
    translate_instruction="Translate the following chunk into English. Do not preface or add any text; only translate."
    distill_instruction="Rewrite the following chunk to be as concise as possible without losing meaning."
    correct_instruction="Correct any grammar, spelling, style, or format errors in this chunk."
    
    def __post_init__(self):
        """ Validate configuration after initialization
        """
        templates_path = Path(self.templates_directory)
        if not templates_path.exists():
            raise ValueError(f"Templates directory does not exist: {self.templates_directory}")
        if not templates_path.is_dir():
            raise ValueError(f"Templates path is not a directory: {self.templates_directory}")
        try:
            parsed = urlparse(self.api_url)
            if not all([parsed.scheme, parsed.netloc]):
                raise ValueError
        except ValueError:
            raise ValueError(f"Invalid API URL format: {self.api_url}")
        if not 0 <= self.temp <= 2:
            raise ValueError(f"Temperature must be between 0 and 2, got {self.temp}")
        if self.gen_count < 1:
            raise ValueError(f"gen_count must be positive, got {self.gen_count}")
        if self.rep_pen < 0:
            raise ValueError(f"rep_pen must be non-negative, got {self.rep_pen}")
        if not 0 <= self.min_p <= 1:
            raise ValueError(f"min_p must be between 0 and 1, got {self.min_p}")

    @classmethod
    def from_json(cls, path: str):
        """ Load configuration from JSON file
            Expects a JSON object with the same field names as the class
        """
        with open(path) as f:
            config_dict = json.load(f)
        return cls(**config_dict)

class LLMProcessor:
    def __init__(self, config):
        """ Initialize the LLM processor with given configuration
        """
        self.config = config
        self.api_function_urls = {
            "tokencount": "/api/extra/tokencount",
            "interrogate": "/api/v1/generate",
            "max_context_length": "/api/extra/true_max_context_length",
            "check": "/api/extra/generate/check",
            "abort": "/api/extra/abort",
            "version": "/api/extra/version",
            "model": "/api/v1/model",
            "generate": "/api/v1/generate",
        }
        self.templates_directory = config.templates_directory
        self.summary_instruction = config.summary_instruction
        self.translate_instruction = config.translate_instruction
        self.distill_instruction = config.distill_instruction
        self.correct_instruction = config.correct_instruction
        self.api_url = config.api_url
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {config.api_password}",
        }
        self.genkey = self._create_genkey()
        self.templates = self._get_templates()
        self.model = self._get_model()
        self.max_context = self._get_max_context_length()
        self.generated = False
        self.system_instruction = "You are a helpful assistant."
        self.max_length = int(self.max_context // 2)
        
    def _get_templates(self):
        """ Look in the templates directory and load JSON template files
            Falls back to Alpaca format if no valid templates found
        """
        templates = {}
        alpaca_template = {
            "name": ["alpaca"],
            "system_start": "### System:",
            "system_end": "\n",
            "user_start": "### Human:",
            "user_end": "\n",
            "assistant_start": "### Assistant:", 
            "assistant_end": "\n"
        }
        try:
            template_path = Path(self.templates_directory)
            if not template_path.exists():
                return {"alpaca": alpaca_template}

            for file in template_path.glob('*.json'):
                try:
                    with open(file) as f:
                        template = json.load(f)
                    required_fields = [
                        "system_start", "system_end",
                        "user_start", "user_end",
                        "assistant_start", "assistant_end"
                    ]
                    if all(field in template for field in required_fields):
                        base_name = file.stem
                       
                        if "akas" not in template:
                            template["akas"] = []
                        if "name" not in template:
                            template["name"] = [base_name]
                        templates[base_name] = template     
                except (json.JSONDecodeError, KeyError) as e:
                    print(f"Error loading template {file}: {str(e)}")
                    continue
            if not templates:
                return {"alpaca": alpaca_template}
            return templates
        except Exception as e:
            print(f"Error loading templates directory: {str(e)}")
            return {"alpaca": alpaca_template}

    def _get_model(self):
        """ Queries Kobold for current model name and finds matching template
            Prefers exact matches, then version matches, then base model matches
            Falls back to alpaca template if no match found
        """
        if self.config.text_completion:
            print("Using text completion mode")
            return {
                "name": ["Completion"],
                "user": "",
                "assistant": "",
                "system": None,
            }
        model_name = self._call_api("model")
        if not model_name:
            print("Could not get model name from API, falling back to alpaca template")
            #return self.templates.get("alpaca")
            model_name = "alpaca"
        print(f"Kobold reports model: {model_name}")

        def normalize(s):
            """ Remove special chars and lowercase for matching
            """
            return re.sub(r"[^a-z0-9]", "", s.lower())
            
        model_name_normalized = normalize(model_name)
        best_match = None
        best_match_length = 0
        best_match_version = 0

        for template in self.templates.values():
            names_to_check = template.get("name", [])
            if isinstance(names_to_check, str):
                names_to_check = [names_to_check]
            names_to_check.extend(template.get("akas", []))

            for name in names_to_check:
                normalized_name = normalize(name)
                if normalized_name in model_name_normalized:
                   
                    version_match = re.search(r'(\d+)(?:\.(\d+))?', name)
                    current_version = float(f"{version_match.group(1)}.{version_match.group(2) or '0'}") if version_match else 0
                    name_length = len(normalized_name)
                    if current_version > best_match_version or \
                       (current_version == best_match_version and name_length > best_match_length):
                        best_match = template
                        best_match_length = name_length
                        best_match_version = current_version

        if best_match:
            print(f"Selected template: {best_match.get('name', ['Unknown'])[0]}")
            return best_match
            
        print(f"No version-specific template found, trying base model match...")
        for template in self.templates.values():
            names_to_check = template.get("name", [])
            if isinstance(names_to_check, str):
                names_to_check = [names_to_check]
            names_to_check.extend(template.get("akas", []))

            for name in names_to_check:
                normalized_name = normalize(name)
               
                base_name = re.sub(r'\d+(?:\.\d+)?', '', normalized_name)
                if base_name in model_name_normalized:
                    name_length = len(base_name)
                    if name_length > best_match_length:
                        best_match = template
                        best_match_length = name_length

        if best_match:
            print(f"Selected base template: {best_match.get('name', ['Unknown'])[0]}")
            return best_match

        print("No matching template found, falling back to alpaca")
        return self.templates.get("alpaca")

    def _call_api(self, api_function, payload=None):
        """ Call the Kobold API
            Some API calls are POSTs and some are GETs
        """
        if api_function not in self.api_function_urls:
            raise ValueError(f"Invalid API function: {api_function}")
        url = f"{self.api_url}{self.api_function_urls[api_function]}"
        try:
            if api_function in ["tokencount", "generate", "check", "interrogate"]:
                response = requests.post(url, json=payload, headers=self.headers)
                result = response.json()
                if api_function == "tokencount":
                    return int(result.get("value"))
                else:
                    return result["results"][0].get("text")
            else:
                response = requests.get(url, json=payload, headers=self.headers)
                result = response.json()
                if resulted := result.get("result", None):
                    return resulted
                else:
                    return int(result.get("value", None))
        except requests.RequestException as e:
            print(f"Error calling API: {str(e)}")
            return None

    def _get_initial_chunk(self, content: str, max_size: int):
        """ We are chunking based on natural break points. 
        """
        max_size_words = self._convert_tokens_and_words("tokens", max_size)
        matches = chunk_regex.finditer(content)
        current_size = 0
        chunks = []
        
        for match in matches:
            chunk = match.group(0)
            chunk_size = len(chunk.split())
            
            if current_size + chunk_size > max_size_words:
                if not chunks:
                    chunks.append(chunk)
                break
                
            chunks.append(chunk)
            current_size += chunk_size
            
        return ''.join(chunks)
        
    def analyze_document(self, content: str):
        """ Analyzes the first chunk of a document to determine its characteristics
        """
        max_chunk = 1024
        first_chunk = self._get_initial_chunk(content, max_chunk)
        
        analysis_prompt = self.compose_prompt(
            instruction= """Return a JSON object as follows: {"title": DOCUMENT TITLE,"type": DOCUMENT TYPE,"subject": DOCUMENT SUBJECT,"language": DOCUMENT LANGUAGE,"keywords": [RELEVANT SEARCH TERMS]}""",
            content=first_chunk
        )
        
        response = self._call_api("generate", analysis_prompt)
        return clean_json(response)
        
    def compose_prompt(self, instruction="", content="", metadata=None):
        prompt = self.get_prompt(instruction, content, metadata)
        payload = {
            "prompt": prompt,
            "max_length": self.max_length,
            "genkey": self.genkey,
            "top_p": self.config.top_p,
            "top_k": self.config.top_k,
            "temp": self.config.temp,
            "rep_pen": self.config.rep_pen,
            "min_p": self.config.min_p,
        }
        return payload

    def get_prompt(self, instruction="", content="", metadata=None):
        if not self.model:
            raise ValueError("No model template loaded")
        if self.model["name"] == ["Completion"]:
            return f"{instruction} {content}".strip()
        user_text = f"{instruction} {content} {instruction}".strip()
        if not user_text:
            raise ValueError("No user text provided (both instruction and content are empty)")
        prompt_parts = []
        
        if self.model.get("system") is not None:
            prompt_parts.extend([
                self.model["system_start"],
                self.system_instruction,
                self.model["system_end"]
            ])

        prompt_parts.extend([
            self.model["user_start"],
            user_text,
            self.model["user_end"],
            self.model["assistant_start"]
        ])
        return "".join(prompt_parts)

    def process_in_chunks(self, instruction="", content="", metadata=None):
        metadata = metadata or {}
        title = metadata.get('title', 'Untitled Document')
        type = metadata.get('type', 'Unknown')
        subject = metadata.get('subject', 'Unknown')
        keywords = metadata.get('keywords', [])

        max_chunk = int(self.max_context // 2)
        chunks = []
        remaining = content
        while remaining:
            chunk = self._get_initial_chunk(remaining, max_chunk)
            chunk_len = len(chunk)
            print(f"Got chunk of length {chunk_len}")  # Debug
            if chunk_len == 0:
                print("Warning: Got zero-length chunk")
                break  # Prevent infinite loop
            chunks.append(chunk)
            remaining = remaining[len(chunk):].strip()
        
        responses = []
        total_chunks = len(chunks)
        print("Starting chunk processing...")
        for i, chunk in enumerate(chunks, 1):
            print(f"Processing chunk {i}/{total_chunks}")
            chunk_context = f"""\n\n## Metadata\nDocument: {title}\nType: {type}\nSubject: {subject}\nChunk: {i} of {total_chunks}\n\n<DOCUMENT>\n\n{chunk}</DOCUMENT>\n\n"""
            response = self.generate_with_status(self.compose_prompt(
                instruction=instruction,
                content=chunk_context
            ))
            print(f"Chunk {i} complete")
            if response:
                responses.append(response)
        return responses

    def route_task(self, task="summary", content="", metadata=None):
        metadata = metadata or self.analyze_document(content)
        
        if task in ["correct", "translate", "distill"]:
            instruction = getattr(self, f"{task}_instruction")
            responses = self.process_in_chunks(instruction, content, metadata)
            return responses, metadata
        elif task == "summary":
            instruction = getattr(self, f"summary_instruction")
            summaries, final_summary = self.process_summary(instruction, content, metadata)
            return summaries, final_summary, metadata
        else:
            raise ValueError(f"Unknown task: {task}")    
    def process_summary(self, instruction="", content="", metadata=None, rolling_summary_threshold=3):
        """ Process content in chunks with both individual and rolling summaries.
            
            Args:
                instruction (str): Base instruction for processing
                content (str): Content to be processed
                metadata (dict): Document metadata
                rolling_summary_threshold (int): Number of summaries to accumulate before distilling
            
            Returns:
                tuple: (individual_summaries, rolling_summaries, final_summary)
        """
        metadata = metadata or {}
        title = metadata.get('title', 'Untitled Document')
        doc_type = metadata.get('type', 'Unknown')
        subject = metadata.get('subject', 'Unknown')
        keywords = metadata.get('keywords', [])

        max_chunk = int(self.max_context // 2)
        chunks = []
        remaining = content
        
        # Split content into chunks
        while remaining:
            chunk = self._get_initial_chunk(remaining, max_chunk)
            chunk_len = len(chunk)
            if chunk_len == 0:
                print("Warning: Got zero-length chunk")
                break
            chunks.append(chunk)
            remaining = remaining[len(chunk):].strip()

        individual_summaries = []
        rolling_summaries = []
        current_rolling_group = []
        total_chunks = len(chunks)
        
        print(f"Starting chunk processing... ({total_chunks} chunks)")
        
        for i, chunk in enumerate(chunks, 1):
        # Process individual chunk
            chunk_context = f"""## Metadata
Document: {title}
Type: {doc_type}
Subject: {subject}
Chunk: {i} of {total_chunks}

Previous Summary: {rolling_summaries[-1] if rolling_summaries else 'None'}

<DOCUMENT>
{chunk}
</DOCUMENT>
"""
        
            # Get individual chunk summary
            chunk_response = self.generate_with_status(self.compose_prompt(
                instruction=instruction,
                content=chunk_context
            ))
            
            if chunk_response:
                individual_summaries.append(chunk_response)
                current_rolling_group.append(chunk_response)
                
                # Check if we need to create a rolling summary
                if len(current_rolling_group) >= rolling_summary_threshold:
                    rolling_context = f"""## Previous Summaries

{chr(10).join(current_rolling_group)}

## Task
Create a concise rolling summary that captures the key points from these summaries while maintaining coherent narrative flow.
Emphasize recurring themes and major developments. The summary should be shorter than the combined input summaries."""

                    rolling_response = self.generate_with_status(self.compose_prompt(
                        instruction="Synthesize these summaries into a single coherent summary:",
                        content=rolling_context
                    ))
                    
                    if rolling_response:
                        rolling_summaries.append(rolling_response)
                        current_rolling_group = [rolling_response]  # Keep last summary for context
                        
                    print(f"Created rolling summary at chunk {i}")

    # Create final summary if there are any remaining chunks
        if current_rolling_group:
            final_context = f"""## Document Information
Title: {title}
Type: {doc_type}
Subject: {subject}

## All Summaries
{chr(10).join(individual_summaries)}

## Task
Create a comprehensive final summary of the entire document. Focus on:
1. Major themes and developments
2. Key points from each section
3. Overall narrative flow and conclusions
The summary should be thorough but more concise than the combined input."""

            final_summary = self.generate_with_status(self.compose_prompt(
                instruction="Create a comprehensive final summary:",
                content=final_context
            ))
        else:
            final_summary = None
        summaries = {
            "individual_summaries": individual_summaries,
            "rolling_summaries": rolling_summaries,
            }
        return summaries, final_summary

    def generate_with_status(self, prompt):
        """ Threads generation so that we can stream the output onto the 
            console otherwise we stare at a blank screen
        """
        self.generated = False
        
        monitor = threading.Thread(
            target=self._monitor_generation,
            daemon=True
        )
        monitor.start()
        
        try:
            result = self._call_api("generate", prompt)
            self.generated = True
            monitor.join()
            return result
        except Exception as e:
            print(f"Generation error: {e}")
            return None
                
    def _monitor_generation(self):
        """ Write generation onto the terminal as it is created
        """
        generating = False
        payload = {
            'genkey': self.genkey
        }
        while not self.generated:
            try:
                result = self._call_api("check", payload)
                if result:
                    if result == '' and generating is False:
                        continue
                    if result == '' and generating is True:
                        break
                    else:
                        generating = True
                        clearConsole()
                        print(f"\r{result}", end="\n", flush=True)
            except Exception as e:
                pass
            time.sleep(2)
          
    @staticmethod
    def _create_genkey():
        """ Create a unique generation key
            Prevents kobold from returning your generation to another query
        """
        return f"KCPP{''.join(str(random.randint(0, 9)) for _ in range(4))}"

    def _get_max_context_length(self):
        """ Get the maximum context length from the API
        """
        max_context = self._call_api("max_context_length")
        print(f"Model has maximum context length of: {max_context}")
        return max_context

    def _get_token_count(self, content):
        """ Get the token count for a piece of content
        """
        payload = {"prompt": content, "genkey": self.genkey}
        return self._call_api("tokencount", payload)
        
    def _convert_tokens_and_words(self, conversion_type="tokens", amount=1):
        """ Super simple conversion from counting words to tokens and back
        """
        try:
            if conversion_type == "tokens":
                return int(amount * 0.75)
            else:
                return int(amount * 1.5)
        except:
            print("Error converting tokens or words")
            return 0
                
    def _get_content(self, content):            
        """ Read text from a file to chunk
        """
        extractor = Extractor()
        #if os.path.splitext(content)[1] in ["txt", "pdf", "md"]:
        result, metadata = extractor.extract_file_to_string(content)
             
        return result, metadata
        #return
def write_output(output_path: str, task: str, responses, metadata):
    """ Write the task response to a file with metadata headers
    """
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("# Document Analysis\n\n")
            f.write("## Metadata\n")
            for key, value in metadata.items():
                f.write(f"{key}: {value}\n")
            
            f.write("\n## Chunk Responses\n")
            for i, chunk in enumerate(responses, 1):
                f.write(f"\nChunk {i}:\n{chunk}\n")
        
            print(f"\nOutput written to: {output_path}")
    except Exception as e:
        print(f"Error writing output file: {str(e)}")
        
def write_summary_output(output_path: str, summaries, final_summary, metadata):
    """ Write all summaries to file with clear section separation.
    """
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("# Document Analysis\n\n")
            
            f.write("## Metadata\n")
            for key, value in metadata.items():
                f.write(f"{key}: {value}\n")
            
            f.write("\n## Final Summary\n")
            if final_summary:
                f.write(f"{final_summary}")
            
            f.write("\n\n## Rolling Summaries\n")
            for i, summary in enumerate(summaries["rolling_summaries"], 1):
                f.write(f"Rolling Summary {i}:\n{summary}\n")
            
            f.write("\n## Individual Chunk Summaries\n")
            for i, summary in enumerate(summaries["individual_summaries"], 1):
                f.write(f"\nChunk {i}:\n{summary}\n")
        
        print(f"\nOutput written to: {output_path}")
    except Exception as e:
        print(f"Error writing output file: {str(e)}")

        
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="LLM Processor for Kobold API")
    parser.add_argument("--config", type=str, help="Path to JSON config file")
    parser.add_argument("--content", type=str, help="Content to process")
    parser.add_argument("--api-url", type=str, default="http://localhost:5001", help="URL for the LLM API")
    parser.add_argument("--api-password", type=str, default="", help="Password for the LLM API")
    parser.add_argument("--templates", type=str, default="./templates", help="Directory for instruct templates")
    parser.add_argument("--task", type=str, default="summary", help="Task: summary, translate, distill, correct")
    parser.add_argument("--file", type=str, default="output.txt", help="Output to file path")
   
    args = parser.parse_args()
    try:
        if args.config:
            config = LLMConfig.from_json(args.config)
        else:
                config = LLMConfig(
                    templates_directory=args.templates,
                    api_url=args.api_url,
                    api_password=args.api_password,
                )
            
        task = args.task.lower()
        processor = LLMProcessor(config)
        content, metadata = processor._get_content(args.content)    
        print(metadata)
        file = args.file
        if task in ["translate", "distill", "correct"]:
            responses, llm_metadata = processor.route_task(task, content)
            write_output(file, task, responses, llm_metadata) 
        elif task == "summary":
            summaries, final_summary, llm_metadata = processor.route_task("summary", content)
            write_summary_output(file, summaries, final_summary, llm_metadata)
        else:
            print("Error - No task selected from: summary, translate, distill, correct")
            
    except KeyboardInterrupt:
        print("\nExiting...")
        exit(0)
    except Exception as e:
        print(f"Fatal error: {str(e)}")
        import traceback
        traceback.print_exc()
        exit(1)
        
