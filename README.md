activity-recognition-KTH
========================
#### Human Activity Recognition based on Summarized Semi-detailed Frame Information and Contextual Features

This repository contains code for our [paper](https://ieeexplore.ieee.org/document/9106564):
```
D. Chatterjee, C. Arora, S. Chakraborty and S. K. Saha, "Human Activity Recognition based on Summarized 
Semi-detailed Frame Information and Contextual Features," 2020 IEEE Calcutta Conference (CALCON), 
Kolkata, India, 2020
```

Please considering citing our paper if you find our code useful:
```
@inproceedings{chatterjee2020human,
  title={Human Activity Recognition based on Summarized Semi-detailed Frame Information and Contextual Features},
  author={Chatterjee, Dibyadip and Arora, Charu and Chakraborty, Saurajit and Saha, Sanjoy Kumar},
  booktitle={2020 IEEE Calcutta Conference (CALCON)},
  pages={127--131},
  year={2020},
  organization={IEEE}
}
```

Requirements
------------
* Python3
* Numpy
* Matplotlib
* OpenCV
* sklearn

File Structure
--------------
Within **`./src/main.py`**, following python files are available:
* [main.py](src/main.py) The main file which will be run on the terminal with command line arguments for different functionalities.
* [motion_2x2.py](src/motion_2x2.py) This file contains the function that produces motion based features (HOF) when the POI is divided into 2x2 grids.
* [motion_3x3.py](src/motion_3x3.py) This file contains the function that produces motion based features (HOF) when the POI is divided into 3x3 grids.
* [motion_4x4.py](src/motion_4x4.py) This file contains the function that produces motion based features (HOF) when the POI is divided into 4x4 grids.
* [contextual_2x2.py](src/contextual_2x2.py) This file contains the function that produces motion based contextual features when the POI is divided into 2x2 grids.
* [contextual_3x3.py](src/contextual_3x3.py) This file contains the function that produces motion based contextual features when the POI is divided into 3x3 grids.
* [contextual_4x4.py](src/contextual_4x4.py) This file contains the function that produces motion based contextual features when the POI is divided into 4x4 grids.
* [orientation_features.py](src/orientation_features.py) This file contains the functions that produce orientation based (HOG) normal and contextual features i.e. features A and C as referred in the paper respectively.
* [classifier.py](src/classifier.py) This file contains the functions for training, testing and preparing data for our baseline classifier, Support Vector Machine (SVM).
* [utils.py](src/utils.py) This file contains utility functions required for proper functioning of the other files.

Working
-------
The 'cascadG.xml' can be downloaded from [here](https://drive.google.com/open?id=11YV5oFM6aQ4xgvdZjBq_rstm-ICMC2hp).
### Processing features
```sh
python3 main.py process
```
For e.g. if you want to process all the contextual features (described in the paper as C+D) the "process" section of the **`./src/main.py`** is supposed to look like
```python
#a = motion_2x2(str)
#b = motion_3x3(str)
#c = motion_4x4(str)
#d = HOG_central(str)
# an added context feature is also processed for motion to get consistency between the valid videos processed
#e, f_n = contextual_2x2(str)

a, f_na = contextual_2x2(str)
b, f_nb = contextual_3x3(str)
c, f_nc = contextual_4x4(str)
d = HOG_context(str)

b = np.concatenate((a,b,c,d), axis=0)
```
The processed features get saved in the **`./src/bin`** folder as a .npy file.

### Training individual features
Load the required numpy feature file from the **`./src/bin`** folder and run the following command on the terminal
```sh
python3 main.py train
```

Before training, set the SVM parameters through grid search for which change the code under "train" section in the **`./src/main.py`** as
```python
# for training and classification
#SVM(feature_vector)

# for grid search
SVM_grid(feature_vector)
```

### Training an ensemble of classifiers
Load any 2 required numpy feature files from the **`./src/bin`** folder and run the following command on the terminal
```sh
python3 main.py train_ensemble
```

### Create an ensemble of features (concatenation) from any two individual features
Load any 2 required numpy feature files from the **`./src/bin`** folder and run the following command on the terminal
```sh
python3 main.py concat_features
```

Experimental Results
-------
Results are measured on KTH Dataset. It consists of 599 videos which belong to 6 different classes viz. *boxing*, *handclapping*, *handwaving*, *jogging*, *running* and *walking*.
Features | Accuracy
--------|--------
Orientation Based (A)| 66.39%
Motion Based (B)| 88.23%
Orientation Based Contextual Feature (C) | 73.87%
Motion Based Contextual Feature (D) | 84.87%
A+B | 89.07%
C+D | 87.34
<b>(A+B)*(C+D)</b>| **92.00%**

where 
* ' + ' : Ensemble of features (Concatenation)
* ' * ' : Ensemble of classifiers

License
-------
MIT

Author
-----
[Dibyadip Chatterjee](https://dibschat.github.io/)
