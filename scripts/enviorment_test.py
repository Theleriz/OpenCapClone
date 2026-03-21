import importlib

required_modules = [
    "numpy",
    "pandas",
    "matplotlib",
    "mediapipe",
    "cv2"
]

def check_modules(modules):
    for module in modules:
        try:
            mod = importlib.import_module(module)
            version = getattr(mod, "__version__", "Unknown version")
            print(f"{module} is installed (version: {version})")
        except ImportError:
            print(f"{module} is NOT installed")

if __name__ == "__main__":
    check_modules(required_modules)
