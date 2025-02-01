import os

def log_event(event_file: str, event_str: str):
    with open(event_file, "a") as f:
        f.write(f"{event_str}\n")

def find_round_offset(log_path: str) -> int:
    max_offset = 0

    if not os.path.exists(log_path):
        return max_offset
    
    for file in os.listdir(log_path):
        if ".npz" not in file or not "round" in file or not "weights" in file:
            continue

        offset = int(file.split(".")[0].split("-")[1].strip())
        if offset > max_offset:
            max_offset = offset
    
    return max_offset

def remove_dir(dir):
    if not os.path.isdir(dir):
        os.remove(dir)
        return
    
    for f in os.listdir(dir):
        remove_dir(os.path.join(dir, f))
    os.rmdir(dir)

def remove_old_dirs(path: str, num: int = 20):
    dirs = []
    
    for dir in os.listdir(path):
        if "run-" in dir:
            dirs.append((os.path.join(path, dir), int(dir.split("-")[1])))
    
    dirs = sorted(dirs, key=lambda x: x[1])

    num = min(len(dirs), num)

    dirs = [remove_dir(d) for d, _ in dirs[:num]]
