from setuptools import setup, find_packages

def reqs():
    with open("requirements.txt") as f:
        return [
            line.strip()
            for line in f
            if line.strip() and not line.startswith("-e")
        ]

setup(
    name = "kipo",
    version = "0.1.0",
    packages = find_packages(where = "src"),
    package_dir = {"": "src"},
    install_requires = reqs(),
    python_requires=">=3.8,<=3.10",
)