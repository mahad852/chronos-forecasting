import os

def log_event(event_file: str, event_str: str):
    with open(event_file, "a") as f:
        f.write(f"{event_str}\n")

def find_round_offset(log_path: str) -> int:
    max_offset = 0

    if not os.path.exists(log_path):
        return max_offset
    
    for file in os.listdir(log_path):
        if ".npz" not in file or not "round" in file:
            continue

        offset = int(file.split(".")[0].split("-")[1].strip())
        if offset > max_offset:
            max_offset = offset
    
    return max_offset