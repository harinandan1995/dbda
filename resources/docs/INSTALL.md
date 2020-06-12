## Installation

### Packages

Before executing the code a few python packages have to be installed. Below are two of the best ways to install a virtual env to make sure you dont have conflicting packages installed in your system


#### Virtualenv

Run the following commands to setup a virtual environment and install the packages in this environment 

Install **pip** first

    sudo apt-get install python3-pip

Then install **virtualenv** using pip3

    sudo pip3 install virtualenv 

Now create a virtual environment 

    virtualenv venv 

> you can use any name instead of **venv**

Active your virtual environment   
    
    source venv/bin/activate
    
Install necessary packages

    pip3 install -r requirements.txt

#### Conda

Install the latest version of conda with python3.7 support from [here](https://docs.conda.io/projects/conda/en/latest/user-guide/install/)

After conda is installed run the following commands to setup a virtual environment.

Create a virtual environment named venv

    conda create -n venv
    
Activate the virtual environment

    conda activate venv
   
Install the packages from requirements.txt
    
    conda install --file requirements.txt

Run ```which python``` and it should point to the python from the virtual environment. 