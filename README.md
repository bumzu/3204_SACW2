# 3204_SACW2
Coursework 2, ICT3204 Security Analytics

Usage: ```python3 main.py <training_csv> <testing_csv>```\
e.g. ```python3 main.py train.csv UnseenData.csv```

Setup?:
1. Add this to path ```C:\Users\<user>\anaconda3\condabin```
2. In VSCode, type ">Python: Create Environment"
3. Select Conda
4. VSCode will do something like ```conda activate c:\Users\<user>\3204_SACW2\.conda```
5. Type ```conda create -n sklearn-env -c conda-forge scikit-learn``` and enter
6. Type ```conda activate sklearn-env``` and enter

## Repo Files

- ```evaluation.py``` is the file used to evaluate the best model.
- ```gradient.py``` is to search for the best hyperparmeters in our best model.
- ```mypredict.py``` is the file used to predict the actual 'label' column. (this is to be submitted)
- ```generate.py``` is the file used to generate your own unseendata.csv

Sample UnseenData has already been provided.
