# List of modules doing various executions

### Installation

1. First of all, Python version 3.9 and above is required.
2. Secondly, create a virtual environment for Python 3 with `python3 -m venv venv` and then activate with `source venv/bin/activate`.
3. Install Dlib on Ubuntu as described [here](https://learnopencv.com/install-dlib-on-ubuntu/)
4. Then, we need to install dependencies. Use `pip install pip-tools` and then run `pip-compile && pip-sync` (for Windows `python -m piptools compile && python -m piptools sync`) to install requirements from setup.py file. Then, and ***only if*** it is the first time you run the spider, excute in the shell the command `playwright install`. Another way to install required dependencies is by using `pip install -r requirements.txt` based on an precompiled requirements file. 
-:warning:- Be aware that if you use the existing requirements file to install dependencies, you may have compatibility issues due to different machines.

### Troubles encountered and solutions
There is a specific order of packages that need to be downloaded and install manually (if anaconda is not used) prior to using geopandas.
This can be better explained [here](https://towardsdatascience.com/geopandas-installation-the-easy-way-for-windows-31a666b3610f) and in the [gps_to_map.py](gps_to_map.py) file.

### Notes
Since this project consists of many python modules that do various things and/or communicate between them, required libraries and packages may have conflicts in different machines and operating systems.

### Purpose
The purpose of this project is to keep a list of modules that help through productivity and can be easily maintained throughout, due too its lack of dependency between its modules.
Statistics calculation, copy-pasting, scraping, file conversions, information extraction from files, data manipulation and even more function can be found and used.
This is an on going project that supports my personal work but also can help others.