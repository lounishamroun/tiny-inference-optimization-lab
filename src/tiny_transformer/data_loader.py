from pathlib import Path 

def return_text(path:Path|str):
    with open(Path(path),"r") as f:
        input_text=f.read()
    return input_text

