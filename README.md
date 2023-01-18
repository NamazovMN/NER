# Named Entity Recognition
## Goal
The main goal of the project is, to classify tokens (words/phrases) in the provided sentences ino 4 classes of Named Entity Recognition, namely PER for person, ORG for Organization, LOC for location and O for others.
## Model 
LSTM model along with FCN on the top of it was built for the classification task. If you check [model.py](model.py), you can see that model size is dynamic and is defined by user. I will mention how to do it in configuration section, later.
## Dataset
There is ambiguity about the confidentiality of the dataset, so that I cannot publish. I was provided by this dataset thanks to my academic classes. However I will provide playground techniques that you can check the resulting model.
## Playground
In order to test model you can follow the steps that are given: \\
* Initially, you need to pull the project into your local machine; \\
* Them, you should run the following snippet to install all required dependencies: \\
  ```python
  python main.py -r requirements.txt
* Now you are all set to run the following snippet (Note: The source code can be found in [playground.py](playground.py).) \\
  ```python
  python main.py --playground_only --bidirectional --play_bis --experiment_number 26
  
 ## What is new?
 In order to see the result, we need to have tokenizer to split sentence into the words. In order to do this, I used BIS model that can be found in my repository. play_bis parameter in the code snippet that was given above activate it. If you do not set it, model will use NLTK tokenizer.
 
 I hope you will enjoy it!
 
 ***Regards,***

***Mahammad Namazov***
