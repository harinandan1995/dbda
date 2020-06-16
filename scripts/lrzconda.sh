wget https://repo.anaconda.com/archive/Anaconda3-2020.02-Linux-ppc64le.sh -O ~/conda.sh
bash ~/conda.sh -b -p $HOME/conda

sudo conda env create --f ~/building-design-assistant/resources/env.yml