import os
import platform
import ctypes
import importlib.util

def fix_torch_dll():
    """Pre-load c10.dll to work around WinError 1114 on PyTorch 2.9+."""
    if platform.system() == "Windows":
        try:
            spec = importlib.util.find_spec("torch")
            if spec and spec.origin:
                dll_path = os.path.join(os.path.dirname(spec.origin), "lib", "c10.dll")
                if os.path.exists(dll_path):
                    ctypes.CDLL(os.path.normpath(dll_path))
                    print("c10.dll pre-loaded successfully.")
                else:
                    print("c10.dll not found at:", dll_path)
        except Exception as e:
            print(f"Could not pre-load c10.dll: {e}")