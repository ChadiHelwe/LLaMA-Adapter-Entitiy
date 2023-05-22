import re
import json
from typing import Any, Dict, List  

def extract_id(id_line:str):
    return re.findall(r'(?<=_)[^_:]*', id_line)[0]

def read_jsonl(dataset_path: str) -> List[Dict[str, Any]]:
    with open(dataset_path, "r", encoding="utf-8") as out:
        jsonl = list(out)

    return [json.loads(i) for i in jsonl]


def read_wikipedia_links(data):
    with open(data, "r", encoding="utf-8") as text:
        sections = text.read().split('-DOCSTART-')

        parsed_data = {}

        for section in sections:
            if section.strip() == '':
                continue

            # Extract the ID within the parentheses
            id_match = re.search(r'\((.*?)\)', section)
            if id_match:
                id_value = id_match.group(1)

            # Extract the lines after '-DOCSTART-'
            lines = section.split('\n')[1:]

            # Remove any empty lines
            lines = [line.split("\t") for line in lines if line.strip() != '' and "--NME--" not in line]

            parsed_data[id_value] = lines

        return parsed_data