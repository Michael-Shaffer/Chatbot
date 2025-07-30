import re
import csv
from typing import Dict, List

class AcronymExpander:
def **init**(self, acronym_file_path: str):
“””
Initialize the expander with acronyms from a CSV file.
Expected CSV format: acronym,definition
“””
self.acronyms = self._load_acronyms(acronym_file_path)

```
def _load_acronyms(self, file_path: str) -> Dict[str, str]:
    """Load acronyms from CSV file into a dictionary."""
    acronyms = {}
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            # Try to detect if first row is header
            sample = file.read(1024)
            file.seek(0)
            sniffer = csv.Sniffer()
            has_header = sniffer.has_header(sample)
            
            reader = csv.reader(file)
            if has_header:
                next(reader)  # Skip header row
            
            for row in reader:
                if len(row) >= 2:
                    acronym = row[0].strip().upper()
                    definition = row[1].strip()
                    acronyms[acronym] = definition
    except FileNotFoundError:
        print(f"Error: File {file_path} not found")
    except Exception as e:
        print(f"Error loading acronyms: {e}")
    
    return acronyms

def find_acronyms(self, text: str) -> List[str]:
    """
    Find all acronyms in the text and return their expansions.
    
    Args:
        text: Input string to search for acronyms
        
    Returns:
        List of expansion strings in format "ACRONYM means Definition."
    """
    # Pattern to match potential acronyms (2+ consecutive uppercase letters)
    acronym_pattern = r'\b[A-Z]{2,}\b'
    
    found_acronyms = re.findall(acronym_pattern, text)
    expansions = []
    
    for acronym in set(found_acronyms):  # Use set to avoid duplicates
        if acronym in self.acronyms:
            expansion = f"{acronym} means {self.acronyms[acronym]}."
            expansions.append(expansion)
    
    return expansions

def expand_text(self, text: str) -> str:
    """
    Return the original text with acronym expansions appended.
    
    Args:
        text: Input string
        
    Returns:
        Original text with expansions added
    """
    expansions = self.find_acronyms(text)
    if expansions:
        return text + " " + " ".join(expansions)
    return text

def add_acronym(self, acronym: str, definition: str):
    """Add a new acronym to the dictionary."""
    self.acronyms[acronym.upper()] = definition

def get_acronym(self, acronym: str) -> str:
    """Get the definition of a specific acronym."""
    return self.acronyms.get(acronym.upper(), "Acronym not found")
```

# Example usage

if **name** == “**main**”:
# Initialize with your exported DOORS CSV file
expander = AcronymExpander(“acronyms.csv”)

```
# Test the function
test_string = "I love ATC and NASA missions"
result = expander.find_acronyms(test_string)

print(f"Input: {test_string}")
print(f"Expansions: {result}")

# You can also expand inline
expanded_text = expander.expand_text(test_string)
print(f"Expanded text: {expanded_text}")

# Example of adding acronyms programmatically
expander.add_acronym("ATC", "Air Traffic Control")
expander.add_acronym("NASA", "National Aeronautics and Space Administration")

# Test again
result = expander.find_acronyms(test_string)
print(f"After adding acronyms: {result}")
```