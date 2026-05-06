import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from model import run

if __name__ == "__main__":
    print("=== Patient Risk Pipeline ===\n")
    run()
    print("\n✅ Pipeline complete")
