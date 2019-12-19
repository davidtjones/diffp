# Diabetic Retinopathy Uncertainty
--

Place state dicts in project root folder. These will be necessary to run the commands below successfully. 

## Useful run script commands

To train the expert and produce new state dicts:
`python run.py expert -t`

To generate new sample images:
`python run.py gan -g -d DIRECTORY -s NUM_SAMPLES`

To generate new sample images with the double gan method:
`python run.py gan -g -d DIRECTORY -s NUM_SAMPLES --double_gan`

To classify a directory of sample images:
`python run.py expert -c -d DIRECTORY`

This will produce two csv files, one is the dataset used and another with the predictions for that dataset.

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



