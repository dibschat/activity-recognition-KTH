# semi-detailed-contextual
Human Activity Recognition based on Summarized Semi-detailed Frame Information and Contextual Features

Requirements
------------
*Python3
*Numpy
*Matplotlib
*OpenCV
*sklearn

Working
-------
### Processing features
```sh
python3 main.py process
```
For e.g. if you want to process all the contextual features (described in the paper as C+D) the process section of the **`./src/main.py`** is supposed to look like
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
Load the required numpy feature file from the **`./src/bin`** folder and run on the terminal
```sh
python3 main.py train
```

Before training, set the SVM parameters through grid search for which change the code as following under train section in the **`./src/main.py`**
```python
    # for training and classification
		#SVM(feature_vector)

		# for grid search
		SVM_grid(feature_vector)
```

### Training an ensemble of classifiers
Load any 2 required numpy feature files from the **`./src/bin`** folder and run on the terminal
```sh
python3 main.py train_ensemble
```

### Create an ensemble of features (concatenation) from any two individual features
Load any 2 required numpy feature files from the **`./src/bin`** folder and run on the terminal
```sh
python3 main.py concat_features
```

Results
-------
Features | Accuracy
--------|--------
Orientation Based (A)| 66.39%
Motion Based (B)| 88.23%
Orientation Based Contextual Feature (C) | 73.87%
Motion Based Contextual Feature (D) | 84.87%
A+B | 89.07%
C+D | 87.34
<b>(A+B)(C+D)</b>| 92.00%
