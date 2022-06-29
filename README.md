# Semester_Project
Segmenting brain metastases with minimal annotations for radiotherapy applications.

- ./results : all the pictures presented in the report are in this folder + some other interesting plot/visualisation 

- ./logs : some logs of the interestings runs of the project

- ./dataset : .json datasets used during the project

-./scripts : 

	-init.py : file where some parameters for the model/training can be tuned. Used to pass arguments to another script (e.g. --epochs=200, --network='UNet'...)

	-train.py : execution of the training process, the script will always try to get the best dice accuracy and save the corresponding model to another folder. The dataset should be in a .json format, some example dataset can be found in the ./dataset folder.

	-inference.py and metrics.py : those two scripts are quite similar, but are used for different purposes when running experiments. 'inference' is for the qualitative results (visualisation) whereas 'metrics' gives the quantitative results (with all the metrics and statistics presented in the report).

	-ROI_training.py : is the try for the training on regions of interest.

