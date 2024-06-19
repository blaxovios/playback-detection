# List of modules doing various executions

### Installation

1. Python version 3.12 required.
2. Secondly, create a virtual environment for Python 3 with `python3.12 -m venv venv` and then activate with `source venv/bin/activate`.
3. Before installing this package, ensure you have `cmake` installed on your system:
For Linux:

```bash
sudo apt update
sudo apt install cmake
```
4. Then, we need to install dependencies based on [pyproject.toml](pyproject.toml) file. Use `pip install --upgrade --upgrade-strategy eager -e .`.
⚠️ Be aware that if you use the existing requirements file to install dependencies, you may have compatibility issues due to different machines.
5. To avoid pushing files larger than 100 MB, use `find . -size +100M | cat >> .gitignore` and `find . -size +100M | cat >> .git/info/exclude`.

### Project Description

A comprehensive audio processing and analysis toolkit designed to extract, process, and analyze audio from video files. This project leverages advanced signal processing techniques to separate audio into vocal and instrumental components, compute spectrograms for detailed frequency analysis, and apply noise reduction for enhanced clarity. Ideal for music producers, sound engineers, and researchers in digital signal processing.