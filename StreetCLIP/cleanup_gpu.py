import gc
import torch

def main():
    # run Python’s garbage collector
    gc.collect()
    # free all unused cached memory on all visible GPUs
    torch.cuda.empty_cache()
    print("✔ GPU cache cleared.")

if __name__ == "__main__":
    main()
