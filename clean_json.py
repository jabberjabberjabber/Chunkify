from json_repair import repair_json as rj
from fix_busted_json import first_json
import re
import json
from typing import Dict, List, Optional, Union

def clean_json(data):
    """ LLMs like to return all sorts of garbage.
        Even when asked to give a structured output
        the will wrap text around it explaining why
        they chose certain things. This function 
        will pull basically anything useful and turn it
        into a dict
    """
    if data is None:
        return None
    if isinstance(data, dict):
        return data
    if isinstance(data, str):
        
        # Try to extract JSON markdown code
        pattern = r"```json\s*(.*?)\s*```"
        match = re.search(pattern, data, re.DOTALL)
        if match:
            data = match.group(1).strip()
        else:
            
            # If no JSON block found, try to find anything that looks like JSON
            json_str = re.search(r"\{.*\}", data, re.DOTALL)
            if json_str:
                data = json_str.group(0)
                
        # Remove extra newlines and funky quotes 
        data = re.sub(r"\n", " ", data)
        data = re.sub(r'["""]', '"', data)
        try:
            return json.loads(rj(data))
                
            # first_json will return the first json found in a string
            # rj tries to repair json using some heuristics
            return json.loads(first_json(rj(data)))
            
            # Is it a markdown list?
            if result := markdown_list_to_dict(data):
                return result
            
            
        except:
            print(f"Failed to parse JSON: {data}")
            
    return None
