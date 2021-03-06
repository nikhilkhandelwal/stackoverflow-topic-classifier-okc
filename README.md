##  StackOverflow Off-topic Question Detection

The project used stackoverflow data to build a machine learning model to classify if a given stackoverflow question will be marked as off topic

### The input data :

1. Body of question
2. Title of question
3. Label (0: On topic, 1: off topic)

### Features of the data:

1. 100k training examples, the two classes are equally represented in the training data.
2. The on topic questions are under sampled from a larger dataset to create this artificially balanced dataset.

### Project Resources
1. Jupyter notebook with eploratory data analysis.
   #### [Exploratory Data Analysis](etc/so-topic-classifier-EDA.ipynb)

2. Since the nice plotly visualization not show up in github
   #### [Exploratory Data Analysis - HTML](etc/so-topic-classifier-EDA.html)

3. Response to the data and model specific questions  
   #### [Answers](answers.md) 

### Usage:

#### 1: Training the model:

python train.py path/to/train/dataset path/to/model/output/dir

#### 2. classify a dataset:

python classify.py model/dir/path path/of/input/file

