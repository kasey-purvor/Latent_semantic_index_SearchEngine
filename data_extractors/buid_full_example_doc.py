import os
import json

def build_full_example_doc(directory='../'):
    """
    Iterate through the irCOREdata directory to build a dictionary containing
    all keys from unique_keys.txt with example values.
    
    Args:
        directory (str): Path to the directory containing the JSON files
        
    Returns:
        dict: Example dictionary with all keys from unique_keys.txt
    """
    # Read the unique keys from the file
    unique_keys_path = os.path.join(directory, 'unique_keys.txt')
    unique_keys = []
    
    with open(unique_keys_path, 'r') as f:
        for line in f:
            key = line.strip()
            if key:  # Skip empty lines
                unique_keys.append(key)
    
    # Initialize the example dictionary
    example_doc = {key: None for key in unique_keys}
    
    # Track which keys we still need to find examples for
    keys_to_find = set(unique_keys)
    
    # Iterate through JSON files until we have an example for each key
    file_count = 0
    
    for root, dirs, files in os.walk(directory):
        for file in files:
            if not keys_to_find:  # If we've found all keys, stop iterating
                break
                
            if file.endswith('.json'):
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, 'r') as f:
                        file_count += 1
                        data = json.load(f)
                        
                        # Check if this document has any of the keys we're looking for
                        for key in list(keys_to_find):  # Use list() to avoid modifying during iteration
                            if key in data and data[key] is not None:
                                example_doc[key] = data[key]
                                keys_to_find.remove(key)
                                
                        print(f"\rProcessed {file_count} documents. Keys remaining: {len(keys_to_find)}", end="")
                        
                except json.JSONDecodeError:
                    print(f"\nError decoding JSON from file: {file_path}")
            
        if not keys_to_find:  # If we've found all keys, stop iterating
            break
    
    print(f"\nCompleted! Found examples for {len(unique_keys) - len(keys_to_find)} out of {len(unique_keys)} keys.")
    
    return example_doc

if __name__ == "__main__":
    example_doc = build_full_example_doc()
    
    # Save the example document to a file
    with open("full_example_doc.json", "w") as f:
        json.dump(example_doc, f, indent=2)
    
    print(f"Example document saved to full_example_doc.json")
