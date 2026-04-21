import os, psutil, time, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

process = psutil.Process(os.getpid())
mem_before = process.memory_info().rss / 1024 / 1024

from src.services.itn_service import _get_processor
_get_processor()

mem_after = process.memory_info().rss / 1024 / 1024
print(f"Memory used by ITN: {mem_after - mem_before:.2f} MB")
