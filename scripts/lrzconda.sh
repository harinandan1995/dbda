wget https://repo.anaconda.com/archive/Anaconda3-2020.02-Linux-x86_64.sh -O ~/conda.sh
bash ~/conda.sh -b -p $HOME/anaconda3

sudo apt-get install  -y --no-install-recommends zsh
sh -c "$(curl -fsSL https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh)" "" --unattended

$HOME/anaconda3/bin/conda init zsh

source ~/.zshrc

conda env create --file ~/dbda/resources/env.yml