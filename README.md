# TFG
José Antonio López Palenzuela bachelor's thesis repository. 

The goal of the project was to implement a ML/DL model that correctly estimates a person's legal age (age >= 18?) based on a panoramic radiography of it's teeth (ortopantomographies). In other words, a binary classifer of images. This repository stores the key functionalities developed for the thesis along with some examples on how this software was and can be used.

The dataset could not be uploaded for legal reasons.

# Repository Content
- [env.yml](env.yml) → Exported conda enviroment to get all packages, libraries and dependencies needed to execute project's code.
- [code/utilities](./code/utilities) → Folder containing the key funtionalities of the project.
- [code/DL_experiment_example](./code/DL_experiment_example.ipynb) → DL experiment example. Jupyter Notebook that shows how to use the key implemented functinoalities of the project in the context of DL: [RAMDataset](./code/utilities/dataset.py), [SingleLogitResnet](./code/utilities/model.py), [LRFind](./code/utilities/lrfind.py), [MixUp](./code/utilities/mixup.py) and [train](./code/utilities/train.py). All the DL results of the thesis were generated executing a similar notebook.
- [code/ML_experiment_example](./code/ML_experiment_example.ipynb) →  ML experiment example. The example uses an SVM classfier, but just by changing the used model and the hyperparameters, the code is functional for a logistic regressor and a random forest classifier.
- [code/justify_example](./code/justify_example.ipynb) → Jupyter Notebook to showing how to use explainability related functionalities: [justify](./code/utilities/eval.py) and [gradCAM](./code/utilities/eval.py).
- [results](./results): Folder containing the results of the example the notebooks.
