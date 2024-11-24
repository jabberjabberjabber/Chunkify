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
    gen_count: int = 500
    temp: float = 0.5
    rep_pen: float = 1.05
    min_p: float = 0.02
    summary_instruction="Summarize the text succinctly."
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

            
    def compose_prompt(self, system_instruction="", instruction="", content=""):
        """ Create a prompt using the template and send to Kobold
            Returns the generated text
        """
        prompt = self.get_prompt(system_instruction, instruction, content)
        payload = {
            "prompt": prompt,
            "max_length": int(self.max_context // 2),
            "genkey": self.genkey,
            "top_p": 1,
            "top_k": 0,
            "temp": self.config.temp,
            "rep_pen": self.config.rep_pen,
            "min_p": self.config.min_p,
        }
        return payload

    def get_prompt(self, system_instruction="", instruction="", content=""):
        """ Constructs a prompt using the model's template format
        """
        if not self.model:
            raise ValueError("No model template loaded")
        if self.model["name"] == ["Completion"]:
            return f"{instruction} {content}".strip()
        user_text = f"{instruction} {content} {instruction}".strip()
        if not user_text:
            raise ValueError("No user text provided (both instruction and content are empty)")
        prompt_parts = []
        
        if self.model.get("system") is not None and system_instruction:
            prompt_parts.extend([
                self.model["system_start"],
                system_instruction,
                self.model["system_end"]
            ])

        prompt_parts.extend([
            self.model["user_start"],
            user_text,
            self.model["user_end"]
        ])

        prompt_parts.extend([
            self.model["assistant_start"]
           
        ])
        return "".join(prompt_parts)
        
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
        
        #max_chunk = int(self.max_context // 2)
        max_chunk = 1024
        first_chunk = self._get_initial_chunk(content, max_chunk)
        
        analysis_prompt = self.compose_prompt(
            instruction= """Return a JSON object as follows: {"title": DOCUMENT TITLE,"type": DOCUMENT TYPE,"subject": DOCUMENT SUBJECT,"language": DOCUMENT LANGUAGE,"keywords": [RELEVANT SEARCH TERMS]}""",
            content=first_chunk
        )
        
        response = self._call_api("generate", analysis_prompt)
        return clean_json(response)
        
    def process_in_chunks(self, metadata, content, instruction, system_instruction):
        """ Process content in chunks while maintaining context
        """
        title = metadata.get('title', 'Untitled Document')
        type = metadata.get('type', 'Unknown')
        subject = metadata.get('subject', 'Unknown')
        keywords = metadata.get('keywords', [])
   
        max_chunk = int(self.max_context // 2)
        chunks = []
        remaining = content
        while remaining:
            chunk = self._get_initial_chunk(remaining, max_chunk)
            chunks.append(chunk)
            remaining = remaining[len(chunk):]
        
        responses = []
        total_chunks = len(chunks)
        
        for i, chunk in enumerate(chunks, 1):
            
            chunk_context = f"""\n\n## Metadata\nDocument: {title}\nType: {type}\nSubject: {subject}\nChunk: {i} of {total_chunks}\n\nSTART:\n\n{chunk}"""
            
            response = self.generate_with_status(self.compose_prompt(
                system_instruction=system_instruction,
                instruction=instruction,
                content=chunk_context
            ))
            if response:
                responses.append(response)
            else:
                continue
        return responses
    
    def route_task(self, system_instruction, task, content):
        """ Route task according to required job
        """
        
        metadata = self.analyze_document(content)
        
        if task == "summary":
            responses = self.process_in_chunks(metadata, content, self.summary_instruction, system_instruction)
            
            content = "\n\n".join(responses)
            
            max_size = int(self._convert_tokens_and_words("tokens", self.max_context) *.75)
            
            if len(content.split()) > max_size:
                current_size = 0
                ongoing_content = []
                for response in responses:
                    ongoing_content.append(response)
                    ongoing_content.append(response)
                    if len(ongoing_content.split()) > max_size:
                        break
                content = ongoing_content      
            
            summary = self.generate_with_status(self.compose_prompt(
                system_instruction="Use these individual summaries to compose an overall summary",
                instruction="Provide a coherent summary combining the main points from all chunks.",
                content=content
            ))
            return {
                'metadata': metadata,
                'responses': responses,
                'summary': summary
            }
        elif task == "correct":
            responses = self.process_in_chunks(metadata, content, self.correct_instruction, system_instruction)
            return {
                'metadata': metadata,
                'response': "\n\n".join(responses)
            }
        elif task == "translate":
            responses = self.process_in_chunks(metadata, content, self.translate_instruction, system_instruction)
            return {
                'metadata': metadata,
                'response': "\n\n".join(responses)
            }
        elif task == "distill":
            responses = self.process_in_chunks(metadata, content, self.distill_instruction, system_instruction)
            return {
                'metadata': metadata,
                'response': "\n\n".join(responses)
            }
        else:
            return
            
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
        try:
            with open(content, 'r', encoding='utf-8') as file:
                return file.read()
        except UnicodeDecodeError:
           
            try:
                with open(content, 'r', encoding='latin-1') as file:
                    return file.read()
            except Exception as e:
                raise ValueError(f"Could not read file {content}: {str(e)}")
        except Exception as e:
            raise ValueError(f"Error reading file {content}: {str(e)}")      
        return
        
def write_output(output_path: str, task: str, response):
    """ Write the task response to a file with metadata headers
    """
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("# Document Analysis\n\n")
            f.write("## Metadata\n")
            for key, value in response['metadata'].items():
                f.write(f"{key}: {value}\n")
            
            if task == "summary":
                f.write("\n## Chunk Responses\n")
                for i, chunk in enumerate(response['responses'], 1):
                    f.write(f"\nChunk {i}:\n{chunk}\n")
                    
                f.write("\n## Final Summary\n")
                f.write(response['summary'])
            else:
                f.write(f"\n## {task.title()} Result\n")
                f.write(response['response'])
            
            print(f"\nOutput written to: {output_path}")
    except Exception as e:
        print(f"Error writing output file: {str(e)}")
        
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="LLM Processor for Kobold API")
    parser.add_argument("--config", type=str, help="Path to JSON config file")
    parser.add_argument("--instruction", type=str, default="", help="System instruction")
    #parser.add_argument("--prompt", type=str, default="", help="User prompt")
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
            
        instruction = args.instruction
        task = args.task.lower()
        processor = LLMProcessor(config)
        content = processor._get_content(args.content)    
        file = args.file
        if task in ["summary", "translate", "distill", "correct"]:
            response = processor.route_task(instruction, task, content)
            write_output(file, task, response) 
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
        
