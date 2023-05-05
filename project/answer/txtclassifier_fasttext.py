#  python3 install -r requirements.txt
#  python -m spacy download en_core_web_lg
import json
import logging
import optparse
import os
import sys
import gzip
import re
from random import shuffle
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data_utils
import pandas as pd
import numpy as np
import string
import spacy
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import classification_report
import fasttext
import nltk

nltk.download('punkt')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def label_exporter(category_list, filename, filename_decode):
    
    label_dict = {}
    label_decode_dict = {}
    
    for i, category in enumerate(category_list):
        label_dict[category] = i
        label_decode_dict[i] = category
    
    with open(filename, 'w') as fp:
        json.dump(label_dict, fp)
     
    with open(filename_decode, 'w') as fp:
        json.dump(label_decode_dict, fp)
        

def label_importer(filename):
    
    with open(filename) as fp:
        label = json.load(fp)
    
    return label


def category_merger(df ,cat_to_merge, target_cat_name):
    
    categories = df['category'].value_counts().index
    
    for category in categories:
        if category in cat_to_merge:
            df.loc[df['category'] == category, 'category'] = target_cat_name
    
    return df



def text_cleaner(df):
    
    def punctuation_remover(text):
        punc_free = "".join([char for char in text if char not in string.punctuation])
        return punc_free
    
    def spacy_lemmatize(lemma_model, text):
        doc = lemma_model(text)
        lemmatized = ' '.join([token.lemma_ for token in doc])
        return lemmatized
    
    # Remove punctuations
    df["text"] = df["text"].apply(lambda a: punctuation_remover(a))
    
    # Convert all the words to lower case
    df["text"] = df["text"].apply(lambda a: a.lower())
    
    # Lemmatize texts using Spacy
    nlp = spacy.load('en_core_web_lg')
    df["text"] = df["text"].apply(lambda a: spacy_lemmatize(nlp, a))
    
    return df


def preprocess(inputfile, labelfile, decodelabelfile):

    org_df = pd.read_json(inputfile, lines=True)
    
    # drop columns that are not useful for our model
    main_df = org_df.drop(columns=['link', 'authors', 'date'])

    # Combine headline and short_description into one column
    main_df['text'] = main_df['headline'] + " " + main_df['short_description']
    main_df = main_df.drop(columns=['headline', 'short_description'])

    # Perform text cleaning
    main_df = text_cleaner(main_df)
    
    # Merge related topics into a single category.
    refined_df = category_merger(main_df, cat_to_merge= ['WELLNESS', 'HEALTHY LIVING']\
                            , target_cat_name =  'WELLNESS')
    refined_df = category_merger(refined_df, cat_to_merge= ['HOME & LIVING','STYLE & BEAUTY' ,'STYLE']\
                            , target_cat_name =  'LIFESTYLE')
    refined_df = category_merger(refined_df, cat_to_merge= ['PARENTING', 'PARENTS']\
                            , target_cat_name =  'PARENT')
    refined_df = category_merger(refined_df, cat_to_merge= ['EDUCATION' ,'COLLEGE']\
                            , target_cat_name =  'EDUCATION')
    refined_df = category_merger(refined_df, cat_to_merge= ['SPORTS', 'ENTERTAINMENT', 'COMEDY', 'WEIRD NEWS', 'ARTS']\
                            , target_cat_name =  'SPORTS AND ENTERTAINMENT')
    refined_df = category_merger(refined_df, cat_to_merge= ['TRAVEL','FOOD & DRINK', 'TASTE']\
                            , target_cat_name =  'TRAVEL')
    refined_df = category_merger(refined_df, cat_to_merge= ['ARTS & CULTURE','CULTURE & ARTS']\
                            , target_cat_name =  'ARTS AND CULTURE')
    refined_df = category_merger(refined_df, cat_to_merge= ['WOMEN','QUEER VOICES', 'LATINO VOICES', 'BLACK VOICES']\
                            , target_cat_name =  'MINORITY')
    refined_df = category_merger(refined_df, cat_to_merge= ['BUSINESS' ,  'MONEY']\
                            , target_cat_name =  'ECONOMY')
    refined_df = category_merger(refined_df, cat_to_merge= ['THE WORLDPOST' , 'WORLDPOST' , 'WORLD NEWS']\
                            , target_cat_name =  'WORLD NEWS')
    refined_df = category_merger(refined_df, cat_to_merge= ['ENVIRONMENT' ,'GREEN']\
                            , target_cat_name =  'ENVIRONMENT')
    refined_df = category_merger(refined_df, cat_to_merge= ['TECH', 'SCIENCE']\
                            , target_cat_name =  'SCIENCE AND TECH')
    refined_df = category_merger(refined_df, cat_to_merge= ['WEDDINGS', 'DIVORCE']\
                            , target_cat_name =  'MARRIAGE')
    refined_df = category_merger(refined_df, cat_to_merge= ['CRIME', 'MEDIA', 'RELIGION', 'GOOD NEWS', 'IMPACT']\
                            , target_cat_name =  'SOCIAL')

    # Drop categories that do not belong to any broader categories defined above
    cat_count_df = pd.DataFrame(refined_df['category'].value_counts()).reset_index()
    cat_count_df.rename(columns={'index': 'category', 'category':'count'}, inplace=True)

    top_15 = cat_count_df['category'][:15].to_list()
    refined_df = refined_df[refined_df['category'].isin(top_15)]
    
    label_exporter(top_15, labelfile, decodelabelfile)


    # Remove records not having a valid text
    refined_df['text'] = refined_df['text'].apply(lambda a: a.strip())
    refined_df.loc[refined_df['text'] == "", 'text'] = np.nan
    refined_df.dropna(subset=['text'], inplace=True)

    # Remove duplicate tuples (exact match)
    refined_df.drop_duplicates(keep='first', inplace=True)

    # Remove records having the same text but differnt categories
    refined_df.drop_duplicates(subset=['text'],keep='first', inplace=True)

    # Shuffle the filtered records that will be used for training
    processed_df = shuffle(refined_df, random_state=10)
    processed_df.reset_index(drop=True, inplace=True)

    return processed_df



def data_split(df, train=.75, test=.15):
    np.random.seed(111)
    df_train, df_val, df_test = np.split(df.sample(frac=1, random_state=40), 
                                     [int(train*len(df)), int((1-test)*len(df))])
    
    return df_train, df_val, df_test


"""Loading Dataset: The preprocessed data is used for here.  
Loading pre-trained fasttext embeddings for reference
"""
def text_to_embedding(text, max_length):
    tokens = nltk.word_tokenize(text)
    tokens = tokens[:max_length]  # Truncate to max_length
    embeddings = [fasttext_model.get_word_vector(token) for token in tokens]
    padding_length = max_length - len(embeddings)
    embeddings.extend([np.zeros(fasttext_model.get_dimension()) for _ in range(padding_length)])
    return np.array(embeddings)

"""Fasttext + CNN implementation"""

class FastTextCNN(nn.Module):
    def __init__(self, embedding_dim, num_filters, filter_sizes, num_classes, dropout_prob):
        super(FastTextCNN, self).__init__()

        self.convs1 = nn.ModuleList([
            nn.Conv2d(in_channels=1, out_channels=num_filters, kernel_size=(fs, embedding_dim))
            for fs in filter_sizes
        ])

        self.convs2 = nn.ModuleList([
            nn.Conv2d(in_channels=num_filters, out_channels=num_filters, kernel_size=(fs, 1))
            for fs in filter_sizes
        ])

        self.dropout = nn.Dropout(dropout_prob)

        self.fc = nn.Linear(len(filter_sizes) * num_filters, num_classes)

    def forward(self, x):
        x = x.unsqueeze(1)  # Add channel dimension (batch_size, 1, max_length, embedding_dim)

        # Apply first set of convolutional layers with different filter sizes
        conv_outputs1 = []
        for conv1 in self.convs1:
            conv_output1 = torch.relu(conv1(x))
            conv_outputs1.append(conv_output1)

        # Apply second set of convolutional layers with different filter sizes
        conv_outputs2 = []
        for conv2, conv_output1 in zip(self.convs2, conv_outputs1):
            conv_output2 = torch.relu(conv2(conv_output1)).squeeze(3)
            pooled_output = torch.max_pool1d(conv_output2, conv_output2.size(2)).squeeze(2)
            conv_outputs2.append(pooled_output)

        x = torch.cat(conv_outputs2, 1)
        x = self.dropout(x)  # Dropout layer
        x = self.fc(x)  # Fully connected layer (batch_size, num_classes)
        return x

"""Fasttext + LSTM implementation"""

class FastTextBiLSTM(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, num_layers, num_classes, dropout):
        super(FastTextBiLSTM, self).__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(p=dropout)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_dim).to(device)
        c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_dim).to(device)

        out, (h_n, c_n) = self.lstm(x, (h0, c0))

        # Concatenate the hidden states of the forward and backward LSTM layers
        out = torch.cat((h_n[-2,:,:], h_n[-1,:,:]), dim=1)
        out = self.dropout(out)
        out = self.fc(out)

        return out


"""Training and test functions for model training and evaluation"""

def training(model, train_loader, val_loader, criterion, optimizer, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'), num_epochs=20, modeltype='CNN'):
    best_val_loss = float('inf')
    for epoch in range(num_epochs):
        # Training loop-
        model.train()
        train_loss = 0
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # Validation loop
        model.eval()
        val_loss = 0
        val_accuracy = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                val_predictions = torch.argmax(outputs, dim=1)
                val_accuracy += (val_predictions == labels).float().mean().item()

        # Print epoch results
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        val_accuracy /= len(val_loader)
        # Save the best model based on validation loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = model.state_dict()
            torch.save(best_model, '../data/fasttext/fasttext_best_model_{}.pth'.format(modeltype))
                
        print(f'Epoch {epoch + 1}/{num_epochs}, Training Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy * 100:.2f}%')

def test(model, test_loader):
    model.eval()
    test_accuracy = 0
    with torch.no_grad():
        true_labels = []
        predicted_labels = []
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            test_predictions = torch.argmax(outputs, dim=1)
            test_accuracy += (test_predictions == labels).float().mean().item()
            true_labels.extend(labels.cpu().numpy())
            predicted_labels.extend(test_predictions.cpu().numpy())

    test_accuracy /= len(test_loader)
    
    # Convert label indices to label names
    classes = le.inverse_transform(list(range(num_classes)))
    true_labels_name = [classes[label_index] for label_index in true_labels]
    predicted_labels_name = [classes[label_index] for label_index in predicted_labels]
    precision, recall, f1_score, _ = precision_recall_fscore_support(true_labels, predicted_labels, average='weighted')
    
    print(f'Test accuracy: {test_accuracy * 100:.2f}%')
    print(f'Precision: {precision:.2f}, Recall: {recall:.2f}, F1-score: {f1_score:.2f}')


    return true_labels_name, predicted_labels_name

class DatasetLoader(Dataset):
    def __init__(self, texts, labels, text_to_embedding, max_length):
        self.texts = texts
        self.labels = labels
        self.text_to_embedding = text_to_embedding
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        text_embedding = self.text_to_embedding(text, self.max_length)
        return torch.tensor(text_embedding, dtype=torch.float32), torch.tensor(label, dtype=torch.long)

if __name__ == '__main__':
    optparser = optparse.OptionParser()
    optparser.add_option("-c", "--classlabelfile", dest='classlabel', default=os.path.join('data', 'label.json'), help="class label dictionary stored in json")
    optparser.add_option("-d", "--decodeclasslabelfile", dest='decodeclasslabel', default=os.path.join('data', 'label_decode.json'), help="class label (reversed) dictionary used for decoding")
    optparser.add_option("-i", "--inputfile", dest="input", default=os.path.join('data', 'News_Category_Dataset_v3.json'), help="News Category Dataset")
    optparser.add_option("-f", "--force", dest="force", action="store_true", default=False, help="force training phase (warning: can take a few hours)")
    optparser.add_option("-l", "--logfile", dest="logfile", default=None, help="log file for debugging")
    (opts, _) = optparser.parse_args()
    
    # print("Pre-processing the dataset")
    # processed_df = preprocess(inputfile="../data/News_Category_Dataset_v3.json", labelfile=opts.classlabel, decodelabelfile=opts.decodeclasslabel)
    
    # Alternative-way to use the save pre-processed file.
    processed_df = pd.read_csv("../data/processed_df.csv")
    df_train, df_val, df_test = data_split(processed_df)
    df=processed_df
        
    # Load pre-trained FastText embeddings
    fasttext_model = fasttext.load_model('./data/fasttext/cc.en.300.bin')

    train_texts = df_train['text'].astype(str).tolist()

    train_labels = df_train['category'].tolist()
    val_texts = df_val['text'].astype(str).tolist()
    val_labels = df_val['category'].tolist()
    test_texts = df_test['text'].astype(str).tolist()
    test_labels = df_test['category'].tolist()

    le = LabelEncoder()
    df['category'] = le.fit_transform(df['category'])
    texts = df['text'].tolist()
    labels = df['category'].tolist()
    embedding_dim = fasttext_model.get_dimension()  
    num_filters = 200
    filter_sizes = [2, 3, 4]
    num_classes = len(le.classes_)
    print(num_classes)
    max_length = 250
    hidden_dim= 64
    num_layers=3

    # Create a mapping of labels to integers
    label_to_idx = {label: idx for idx, label in enumerate(set(train_labels))}
    train_labels_num = [label_to_idx[label] for label in train_labels]
    val_labels_num = [label_to_idx[label] for label in val_labels]
    test_labels_num = [label_to_idx[label] for label in test_labels]

    train_dataset = DatasetLoader(train_texts, train_labels_num, text_to_embedding, max_length)
    val_dataset = DatasetLoader(val_texts, val_labels_num, text_to_embedding, max_length)
    test_dataset = DatasetLoader(test_texts, test_labels_num, text_to_embedding, max_length)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)
    test_loader = DataLoader(test_dataset, batch_size=32)

    model_cnn = FastTextCNN(embedding_dim, num_filters, filter_sizes, num_classes, dropout_prob=0.2).to(device)
    learning_rate = 0.001  
    optimizer = optim.Adam(model_cnn.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    """FASTTEXT+CNN Training"""

    training(model_cnn, train_loader, val_loader, criterion, optimizer, device=device, num_epochs=5, modeltype='CNN')

    """FASTTEXT+CNN Testing"""

    true_labels_name_CNN, predicted_labels_name_CNN = test(model_cnn, test_loader)

    model_lstm = FastTextBiLSTM(embedding_dim, hidden_dim, num_layers, num_classes, dropout=0.2).to(device)
    learning_rate = 0.001  
    optimizer_lstm = torch.optim.Adam(model_lstm.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    """FASTTEXT+LSTM Training"""

    training(model_lstm, train_loader, val_loader, criterion, optimizer_lstm, device=device, num_epochs=5, modeltype='LSTM')

    """FASTTEXT+LSTM Testing"""

    true_labels_name_lstm, predicted_labels_name_lstm = test(model_lstm, test_loader)

    # """To run the best model for fasttext + CNN"""

    saved_model_path = "../data/fasttext/fasttext_best_model_CNN.pth"
    saved_model_state_dict = torch.load(saved_model_path)
    model_cnn.load_state_dict(saved_model_state_dict)
    true_labels_name_CNN, predicted_labels_name_CNN=test(model_cnn, test_loader)
