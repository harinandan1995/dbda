## Installation

### Packages

Before executing the code a few python packages have to be installed. We recommend using conda virtual environment to setup dependencies for project

#### Conda

Install the latest version of conda with latest python support from [here](https://docs.conda.io/projects/conda/en/latest/user-guide/install/)

After conda is installed run the following commands to setup a virtual environment.

Create a conda environment
```
conda env create --file resources/env.yml
``` 
> The above command should create an environment with name dbda

Update an existing environment
```
conda env update --file resources/env.yml
```

Activate the environment
```
conda activate dbda
```   

Run ```which python``` and it should point to the python from the virtual environment. 