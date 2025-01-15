def log_event(event_file: str, event_str: str):
    with open(event_file, "a") as f:
        f.write(f"{event_str}\n")
