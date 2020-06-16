wget https://repo.anaconda.com/archive/Anaconda3-2020.02-Linux-x86_64.sh -O ~/conda.sh
bash ~/conda.sh -b -p $HOME/anaconda3

$HOME/anaconda3/bin/conda init

conda env create --file ~/building-design-assistant/resources/env.yml