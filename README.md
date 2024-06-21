# TFG
José Antonio López Palenzuela bachelor's thesis repository. 

The goal of the project was to implement a ML/DL model that correctly estimates a person's legal age (age >= 18?) based on a panoramic radiography of it's teeth (ortopantomographies). In other words, a binary classifer of images. This repository stores the key functionalities developed for the project along with some examples on how this software can be used.

The whole dataset could not be included for legal reasons.

# Repository Content
- [env.yml](env.yml) → Exported conda enviroment to get all packages, libraries and dependencies needed to execute project's code.
- [code/utilities](./code/utilities) → Folder containing the key funtionalities of the project.
- [code/experiment_example](./code/experiment_example.ipynb) → Jupyter Notebook to showing how the key functionalities of the project were used during experimentation: [RAMDataset](./code/utilities/dataset.py), [SingleLogitResnet](./code/utilities/model.py), [LRFind](./code/utilities/lrfind.py), [MixUp](./code/utilities/mixup.py) and [train](./code/utilities/train.py). Notebook results cannopt be replicated, as the dataset is not public. Progress bars do not display properly.
- [code/justify_example](./code/justify_example.ipynb) → Jupyter Notebook to showing how to use explainability related functionalities: [justify](./code/utilities/eval.py) and [gradCAM](./code/utilities/eval.py).
- [results](./results): Folder containing the results of some the notebooks.
