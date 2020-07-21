## Training

### Generator training with GLO

Architecture used for this step is inspired from [Pix2Pix](https://arxiv.org/abs/1611.07004), which uses a auto 
encoder as a generator to improve the image to image translation.

The input for the generator is the shape of the building and the output is 13 channeled output : 
walls(1), doors(1), windows(1), room types(10)

To train the GAN setup using GLO run

    python3 run.py --config config/glo_train.yaml p2p_glo_train 
    
> Default config used is [config/glo_train.yaml](../../config/glo_train.yaml)

Run `python3 train.py --help` to check the available arguments and descriptions.


### Corner model training

A separate corner model is used to train corner detection from walls. To train the model run

    python3 run.py --config config/corner_train.yaml corner_train

> Default config used is [config/corner_train.yaml](../../config/corner_train.yaml)


### Model summaries

To know the architecture of the models (corner and p2p) and the number of parameters run
    
    python3 run.py summary

### Summaries and Checkpoints

Checkpoints (model parameters) for corner train and p2p train are stored for every few epochs. Losses and other visualizations are added as tensorboard summaries.
By default both summaries and checkpoints are stored in ./out/{date}/{time}

To view the summaries run the following command

    tensorboard --logdir=./out/{date}/{time}/summaries/.