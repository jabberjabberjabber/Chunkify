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
import threading
import time
from extractous import Extractor
import sys

@dataclass
class LLMConfig:
    """ Configuration for LLM processing.
    """
    templates_directory: str
    api_url: str
    api_password: str
    text_completion: bool = False
    gen_count: int = 500 #not used
    temp: float = 0.2
    rep_pen: float = 1
    min_p: float = 0.02
    top_k: int = 0
    top_p: int = 1
    summary_instruction="Extract the key points, themes and actions from the text succinctly without developing any conclusions or commentary."
    translate_instruction="Translate the entire document into English. Maintain linguistic flourish and authorial style as much as possible. Write the full contents without condensing the writing or modernizing the language."
    #translate_instruction = """Generate a faithful English translation of this text.
#- Translate each sentence completely
#- Keep the original's pacing and paragraph structure
#- Maintain any metaphors or imagery, changing only when necessary for understanding
#- Use the same level of formality and tone as the source
#- Try to emulate the author's style
#"""
    distill_instruction="Rewrite the text to be as concise as possible without losing meaning."
    correct_instruction="Correct any grammar, spelling, style, or format errors in the text. Do not alter the text or otherwise change the meaning or style."
    

    @classmethod
    def from_json(cls, path: str):
        """ Load configuration from JSON file.
            Expects a JSON object with the same field names as the class.
        """
        with open(path) as f:
            config_dict = json.load(f)
        return cls(**config_dict)

class LLMProcessor:
    def __init__(self, config, task):
        """ Initialize the LLM processor with given configuration.
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
        self.task = task
        self.max_chunk = int((self.max_context // 2) *.9) # give room for template
        self.max_length = self.max_context // 2
        
    def _get_templates(self):
        """ Look in the templates directory and load JSON template files.
            Falls back to Alpaca format if no valid templates found.
        """
        templates = {}
        alpaca_template = {
            "name": ["alpaca"],
            "akas": [],
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
        """ Queries Kobold for current model name and finds matching template.
            Prefers exact matches, then version matches, then base model matches.
            Exits the script if no match found.
            Overly complicated.
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
            print("Could not get model name from API, exiting script")
            sys.exit(1)
        print(f"Kobold reports model: {model_name}")

        def normalize(s):
            """ Remove special chars and lowercase for matching.
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
        print("No matching template found, exiting script")
        sys.exit(1)

    def _call_api(self, api_function, payload=None):
        """ Call the Kobold API.
            Some API calls are POSTs and some are GETs.
        """
        if api_function not in self.api_function_urls:
            raise ValueError(f"Invalid API function: {api_function}")
        url = f"{self.api_url}{self.api_function_urls[api_function]}"
        try:
            
            if api_function in ["tokencount", "generate", "check", "interrogate", "abort"]:
                response = requests.post(url, json=payload, headers=self.headers)
                result = response.json()
                if api_function == "tokencount":
                    return int(result.get("value"))
                elif api_function == "abort":
                    return result.get("success")
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

    def _get_initial_chunk(self, content):
        """ We are chunking based on natural break points. 
            Only works well for Germanic and Romance languages.
            Ideally content is in markdown format.
        """
        total_tokens = self._get_token_count(content)
        print(f"Content tokens to chunk: {total_tokens}")
        if total_tokens < self.max_chunk:
            return content
        matches = chunk_regex.finditer(content)
        current_size = 0
        chunks = []
        for match in matches:
            chunk = match.group(0)
            chunk_size = self._get_token_count(chunk)       
            if current_size + chunk_size > self.max_chunk:
                if not chunks:
                    chunks.append(chunk)
                break   
            chunks.append(chunk)
            current_size += chunk_size       
        return ''.join(chunks)       

    def compose_prompt(self, instruction="", content=""):
        """ Create the prompt that gets sent to the LLM and specify samplers.
        """
        prompt = self.get_prompt(instruction, content)
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

    def get_prompt(self, instruction="", content=""):
        """ Create a prompt to send to the LLM using the instruct template
            or basic text completion.
        """
        if not self.model:
            raise ValueError("No model template loaded")
        if self.model["name"] == ["Completion"]:
            return f"{content}".strip()
        user_text = f"<START_TEXT>{content}<END_TEXT>{instruction}"
        if not user_text:
            raise ValueError("No user text provided (both instruction and content are empty)")
        prompt_parts = []
        if self.model.get("system") is not None:
            prompt_parts.extend([
                self.model["system_start"],
                self.model["system_instruction"],
                self.model["system_end"]
            ])
        prompt_parts.extend([
            self.model["user_start"],
            user_text,
            self.model["user_end"],
            self.model["assistant_start"]
        ])
        return "".join(prompt_parts)

    def route_task(self, task="summary", content=""):
        """ Send to appropriate function.
        """
        if task in ["correct", "translate", "distill", "summary"]:
            instruction = getattr(self, f"{task}_instruction")
            responses = self.process_in_chunks(instruction, content)
            return responses
        else:
            raise ValueError(f"Unknown task: {task}")
            
    def process_in_chunks(self, instruction="", content=""):
        """ Process the content into chunks.
        """
        chunks = []
        remaining = content
        while remaining:
            chunk = self._get_initial_chunk(remaining)
            chunk_len = len(chunk)
            if chunk_len == 0:
                print("Warning: Got zero-length chunk")
                break
            chunks.append(chunk)
            remaining = remaining[len(chunk):].strip()
        responses = []
        total_chunks = len(chunks)
        print("Starting chunk processing...")
        for i, chunk in enumerate(chunks, 1):
            chunk_tokens = self._get_token_count(chunk)
            print(f"Chunk {i} of {total_chunks}, Size: {chunk_tokens}\n")
            time.sleep(2)
            response = self.generate_with_status(self.compose_prompt(
                instruction=instruction,
                content=chunk
            ))
            if response:
                responses.append(response)
        return responses

    def generate_with_status(self, prompt):
        """ Threads generation so that we can stream the output onto the
            console otherwise we stare at a blank screen.
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
        """ Write generation onto the terminal as it is created.
        """
        generating = False
        payload = {
            'genkey': self.genkey
        }
        while not self.generated:
            result = self._call_api("check", payload)
            if not result:
                time.sleep(2)
                continue
            time.sleep(1)
            clear_console()
            print(f"{result}")
            
    @staticmethod
    def _create_genkey():
        """ Create a unique generation key.
            Prevents kobold from returning your generation to another query.
        """
        return f"KCPP{''.join(str(random.randint(0, 9)) for _ in range(4))}"

    def _get_max_context_length(self):
        """ Get the maximum context length from the API.
        """
        max_context = self._call_api("max_context_length")
        print(f"Model has maximum context length of: {max_context}")
        return max_context

    def _get_token_count(self, content):
        """ Get the token count for a piece of content.
        """
        payload = {"prompt": content, "genkey": self.genkey}
        return self._call_api("tokencount", payload)
                
    def _get_content(self, content):            
        """ Read text from a file to chunk.
        """
        extractor = Extractor()
        result, metadata = extractor.extract_file_to_string(content)    
        return result, metadata
 
def check_api(api_url):
    """ See if the API is ready
    """
    url = f"{api_url}/api/v1/info/version/"
    if requests.get(url, json="", headers=""):
        return True
    return False
    
def write_output(output_path, task, responses, metadata):
    """ Write the task response to a file.
    """
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(f"File: {metadata.get('resourceName', 'Unknown')}\n")
            f.write(f"Type: {metadata.get('Content-Type', 'Unknown')}\n")
            f.write(f"Encoding: {metadata.get('Content-Encoding', 'Unknown')}\n")
            f.write(f"Length: {metadata.get('Content-Length', 'Unknown')}\n\n")
            for response in responses:
                f.write(f"{response}\n\n")
            print(f"\nOutput written to: {output_path}")
    except Exception as e:
        print(f"Error writing output file: {str(e)}")

def clear_console():
    """ Clears the screen so the output can refresh.
    """
    command = 'clear'
    if os.name in ('nt', 'dos'):
        command = 'cls'
    os.system(command)
            
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
        processor = LLMProcessor(config, task)
        content, metadata = processor._get_content(args.content)    
        file = args.file
        if task in ["translate", "distill", "correct", "summary"]:
            responses = processor.route_task(task, content)
            write_output(file, task, responses, metadata) 
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
        
