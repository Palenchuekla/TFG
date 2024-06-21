# TFG
José Antonio López Palenzuela bachelor's thesis repository. 

The goal of the project was to implement a ML/DL model that correctly estimates a person's legal age (age >= 18?) based on a panoramic radiography of it's teeth (ortopantomographies). In other words, a binary classifer of images. This repository stores the key functionalities developed for the project along with some examples on how this software can be used.

Images used to implement the actual models can not be included for legal reasons. Notice that examples do NOT contain sensible data.

# Repository Content
- [env.yml](env.yml): Exported conda enviroment to get all packages, libraries and dependencies needed to execute project's code.
- [code/utilities](./code/utilities): Folder containing the key funtionalities of the project.
- [code/experiment_example](./code/experiment_example.ipynb): Jupyter Notebook to showing how to use some of the key functionalities of the project [RAMDataset](./code/utilities/dataset.py), [SingleLogitResnet](./code/utilities/model.py), [LRFind](./code/utilities/lrfind.py), [MixUp](./code/utilities/mixup.py) and [train](./code/utilities/train.py). Notebook cannot be executed, as the dataset is not of public. Progress bars do not display properly.
- [results](./results/experiment_example.ipynb): Folder containing the results of some the notebooks.
