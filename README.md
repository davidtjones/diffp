# Diabetic Retinopathy Uncertainty
--

Place state dicts in project root folder. These will be necessary to run the commands below successfully. 

To train the expert and produce new state dicts:
`python run.py expert -t`

To generate new sample images:
`python run.py gan -g -d gan/generated_images -s X` where X is the number of samples you would like

