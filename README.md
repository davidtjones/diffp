# Differential Privacy

Running the code is simple, simply perform
`python run.py active|expert|gan train|evaluate dataset`

Where dataset is some folder located in `./datasets`



## Useful run script commands

Currently in refactor

## Notes/To do/Future work

* Expert Classifier
  - learning rate scheduling: current model converges after 2-3 epochs to around 73%
  - class balancing?
  - data augmentation? (probably through lighting changes?)
* GAN 
  - class balancing?
  - larger kernel size in earlier convolutional layers
  - better regularization
  - increase output resolution??
  - potentially do more epochs
  
* Need to make some damn plots
  - GAN sample
  - expert classifier confusion matrix



