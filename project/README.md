# The Role of Word Embeddings in Text Classification

This is a group project for the **Natural Language Programming (CMPT-713)** course that aims to classify news data into different categories using different word embeddings. The project uses various libraries, including transformers, pandas, torch, scikit-learn, spacy, seaborn, matplotlib, and fasttext, to develop a deep learning model that can classify news articles into 15 different categories Eg.(Politics, sports, and business etc.). The project focuses on comparing the performance of different word embeddings, such as **GloVe, FastText, and BERT** and using a baseline neural network model.

## Installation

1. Create a virtual environment: 
    ```
    python3 -m venv env
    ```
2. Activate the virtual environment:
    ```
    source env/bin/activate
    ```
3. Install the required libraries:
    ```
    pip install -r requirements.txt
    ```
    The libraries to be installed are:
    * numpy
    * scipy
    * tqdm
    * pandas-profiling
    * notebook
    * jupyter_contrib_nbextensions
    * jupyter_nbextensions_configurator
    * jupyter nbextension enable --py widgetsnbextension
    * transformers
    * pandas
    * torch
    * scikit-learn
    * spacy
    * seaborn
    * matplotlib
    * fasttext

4. Download Spacy en_core_web_lg model for the lemmatization process to be executed successfully:
    ```
    python -m spacy download en_core_web_lg
    ```

## Usage

- To run **CNN model with GloVe Word Embeddings** navigate to the `project/answer/` directory and follow the steps below. 
    1. To download the pre-trained glove embeddings. 
    ```
    !wget https://nlp.stanford.edu/data/glove.6B.zip
    ```
    2. To unzip the zip file
    ```
    unzip glove.6B.zip
    ```
    3. Move the pre-trained glove embedding file with 100d to a different folder. 
    ```
    mv glove.6B.100d.txt data/glove/.
    ```
    4. Run the below code to train and evaluate the model. 
    ```
    python3 txtclassifier_glove.py
    ```
    This command automatically takes in the necessary input files like GloVe word embeddings and the News dataset and outputs the best model file and prints the best score for the model. Once the model is successfully executed you should be able to see the output as below (sample screenshot shown with 1 epoch): 

    <img src="/project/answer/data/glove/Capture.PNG" alt="alt text" width="500" height="300">

- To run **CNN with FastText word embedding** navigate to the `project/answer/` directory and follow the steps below. 
    1. To download the pre-trained FastText embeddings.
    ```
    !wget https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.en.300.bin.gz
    ```

    2. To unzip the zip file
    ```
    !gunzip cc.en.300.bin.gz
    ```
    
    3. Move the pre-trained fasttext embedding file with 100d to a different folder. 
    ```
    cc.en.300.bin data/fasttext/.
    ```

    4. Run the below code to train and evaluate the model. 
    ```
    python3 txtclassifier_fasttext.py
    ```

- To run **BERT clssifier** navigate to `project` directory and follow the steps below:
    1. To download the trained BERT model, use the following link to download the bert_trained_2.pth file:
    https://drive.google.com/file/d/1ihnm2tXgFLlGJk39gssX6emPAG7QUwia/view?usp=share_link

    2. Place the downloaded bert_trained_2.pth file under `project/data/bert/`.

    3. Run the command below to start evaluation using the trained BERT model.
    ```
    python3 answer/txtclassifier_bert.py
    ```
    4. (Optional) Run the command below to run both training and evalutiaon. 
    **CAUTION**: It may take more than 2 hours to finish one EPOCH of the training prcess.
    ```
    python3 answer/txtclassifier_bert.py -f True
    ```


## Acknowledgments

We would like to thank our instructor **Angel Xuan Chang** and teaching assistants **Pooya Kabiri, Yilong Gu, and Ruiqi Wang** for their guidance and support during this project. We would also like to thank the creators of the libraries and resources we used in this project.
