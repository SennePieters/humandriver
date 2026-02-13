from setuptools import setup, find_packages
from pathlib import Path

this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding="utf-8")

setup(
    name="humandriver",
    version="1.0.3",
    description="Human-like mouse and keyboard automation for zendriver",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="SennePieters",
    author_email="senne.pieters02@gmail.com",
    url="https://github.com/SennePieters/humandriver",
    project_urls={
        "Bug Tracker": "https://github.com/SennePieters/humandriver/issues",
    },
    packages=find_packages(),
    install_requires=[
        "zendriver",
        "Pillow",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
)
