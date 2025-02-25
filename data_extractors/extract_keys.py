import os
import json

def flatten_ircoredata_dir(directory):
    unique_keys = set()
    file_count = 0
    
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.json'):
                file_path = os.path.join(root, file)
                with open(file_path, 'r') as f:
                    try:
                        file_count += 1
                        data = json.load(f)
                        unique_keys.update(data.keys())
                        print(f"\rProcessed document: {file_count}", end="")
                    except json.JSONDecodeError:
                        print(f"Error decoding JSON from file: {file_path}")
            else:
                print(f"Skipping non-JSON file: {file}")

    print()  # Add a newline after all files are processed
    return unique_keys

# Define the directory variable
directory = '../'  # Update this path as needed

output_file_path = os.path.join(directory, 'unique_keys.txt')
unique_keys = flatten_ircoredata_dir(directory)

with open(output_file_path, 'w') as output_file:
    for key in unique_keys:
        output_file.write(f"{key}\n")
