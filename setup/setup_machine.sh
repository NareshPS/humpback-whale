
#Install python
sudo apt update
sudo apt install python3-dev python3-pip
sudo pip3 install -U virtualenv  # system-wide install

#Install jupyter notebook. Configuration: https://jupyter-notebook.readthedocs.io/en/stable/public_server.html
sudo apt install python3-notebook jupyter-core python3-ipykernel jupyter-notebook

#OpenCV dependencies
sudo apt install libsm6 libxrender1 libfontconfig1

#Create a new virtual environment.
virtualenv --system-site-packages -p python3 ~/venv

#Activate virtual environment
source ~/venv/bin/activate  # sh, bash, ksh, or zsh

#Upgrade pip3
pip install --upgrade pip

#Install tensorflow
pip install tensorflow

#Install opencv
pip install opencv-python

#Install scikit
pip install scikit-image

#Install keras
pip install keras

#Install pydot
pip install pydot
pip install graphviz

#Install TQDM to create progress bars
pip install tqdm