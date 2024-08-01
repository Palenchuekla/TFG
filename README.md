# TFG
José Antonio López Palenzuela bachelor's thesis repository. 

The goal of the project was to implement a ML/DL model that correctly estimates a person's legal age (age >= 18?) based on a panoramic radiography of it's teeth (ortopantomographies). In other words, a binary classifer of images. This repository stores the key functionalities developed for the thesis along with some examples on how this software was and can be used.

The dataset could not be uploaded for legal reasons.

# Repository Content
- [TFG_Jose_Antonio_Lopez_Palenzuela.pdf](TFG_Jose_Antonio_Lopez_Palenzuela.pdf) → Thesis (written in Spanish).
- - [Presentacion.pdf](Presentacion.pdf) → Thesis Slides.
- [env.yml](env.yml) → Exported conda enviroment to get all packages, libraries and dependencies needed to execute project's code.
- [code/utilities](./code/utilities) → Folder containing the key funtionalities of the project.
- [code/DL_experiment_example](./code/DL_experiment_example.ipynb) → DL experiment example. Jupyter Notebook that shows how to use the key implemented functinoalities of the project in the context of DL: [RAMDataset](./code/utilities/dataset.py), [SingleLogitResnet](./code/utilities/model.py), [LRFind](./code/utilities/lrfind.py), [MixUp](./code/utilities/mixup.py) and [train](./code/utilities/train.py). All the DL results of the thesis were generated executing a similar notebook, just by changing the value of the hyperparameters of the models (architecture of the feature extractor, learning rate, oversampling, data augmentation transformations).
- [code/ML_experiment_example](./code/ML_experiment_example.ipynb) →  ML experiment example. All the ML results of the thesis were generated executing a similar notebook, just by changing the used model and the hyperparameters explored.
- [code/justify_example](./code/justify_example.ipynb) → Jupyter Notebook to showing how to use explainability related functionalities: [justify](./code/utilities/eval.py) and [gradCAM](./code/utilities/eval.py).
- [results](./results) → Folder containing the results of the example the notebooks. Results subfolder match the name of the notebook that generated them.
