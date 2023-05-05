#!python -m spacy download en_core_web_lg

import json
import logging
import optparse
import os
from random import shuffle
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data_utils
import pandas as pd
import numpy as np
import string, spacy

# ================================ Common Base Code Shared by ALL models ===========================================


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
    processed_df = shuffle(refined_df)
    processed_df.reset_index(drop=True, inplace=True)

    return processed_df

def data_split(df, train=.75, test=.15):
    np.random.seed(111)
    df_train, df_val, df_test = np.split(df.sample(frac=1, random_state=40), 
                                     [int(train*len(df)), int((1-test)*len(df))])
    
    return df_train, df_val, df_test

# ========================================= Embedding-specific Code =====================================================

def load_embedding_dict(embedding_file):
    embedding_dict = {}
    with open(embedding_file, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], dtype='float32')
            embedding_dict[word] = vector
    return embedding_dict

def create_word_to_idx(X, embedding_dict):
    word_to_idx = {}
    idx = 1
    for text in X:
        words = text.split()
        for word in words:
            if word not in word_to_idx and word in embedding_dict:
                word_to_idx[word] = idx
                idx += 1
    vocab_size = len(word_to_idx) + 1
    return word_to_idx, vocab_size

def preprocess_data(df, word_to_idx, max_seq_len):
    # Split the data into train, validation, and test sets
    df_train, df_val, df_test = data_split(df)
    
    # Convert the text and label columns of the train, validation, and test sets into lists
    train_texts = df_train['text'].astype(str).tolist()
    train_labels = df_train['category'].tolist()
    val_texts = df_val['text'].astype(str).tolist()
    val_labels = df_val['category'].tolist()
    test_texts = df_test['text'].astype(str).tolist()
    test_labels = df_test['category'].tolist()
    
    # Preprocess the text by mapping each word to its corresponding index and padding each sequence to a fixed length
    train_texts = [[word_to_idx[word] for word in text.split() if word in word_to_idx][:max_seq_len] for text in train_texts]
    train_texts = torch.tensor([xi + [0]*(max_seq_len - len(xi)) for xi in train_texts], dtype=torch.long)
    train_labels = torch.tensor(pd.get_dummies(train_labels).values, dtype=torch.float32)

    val_texts = [[word_to_idx[word] for word in text.split() if word in word_to_idx][:max_seq_len] for text in val_texts]
    val_texts = torch.tensor([xi + [0]*(max_seq_len - len(xi)) for xi in val_texts], dtype=torch.long)
    val_labels = torch.tensor(pd.get_dummies(val_labels).values, dtype=torch.float32)

    test_texts = [[word_to_idx[word] for word in text.split() if word in word_to_idx][:max_seq_len] for text in test_texts]
    test_texts = torch.tensor([xi + [0]*(max_seq_len - len(xi)) for xi in test_texts], dtype=torch.long)
    test_labels = torch.tensor(pd.get_dummies(test_labels).values, dtype=torch.float32)
    
    return train_texts, train_labels, val_texts, val_labels, test_texts, test_labels

def create_data_loaders(train_texts, train_labels, val_texts, val_labels, test_texts, test_labels, batch_size=32):
    train_dataset = data_utils.TensorDataset(train_texts, train_labels)
    val_dataset = data_utils.TensorDataset(val_texts, val_labels)
    test_dataset = data_utils.TensorDataset(test_texts, test_labels)

    train_loader = data_utils.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = data_utils.DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    test_loader = data_utils.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader

class TextCNN(nn.Module):
    def __init__(self, embedding_dict, num_classes, max_seq_len):
        super(TextCNN, self).__init__()
        embedding_dim = 100
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.embedding.weight.data.copy_(torch.from_numpy(np.random.normal(0, 1, (vocab_size, embedding_dim))))
        for word, idx in word_to_idx.items():
            if word in embedding_dict:
                self.embedding.weight.data[idx] = torch.from_numpy(embedding_dict[word])
        self.conv1 = nn.Conv1d(in_channels=embedding_dim, out_channels=100, kernel_size=3)
        self.maxpool = nn.MaxPool1d(max_seq_len - 3 + 1)
        self.fc = nn.Linear(100, num_classes)

    def forward(self, x):
        x = self.embedding(x)
        x = x.permute(0, 2, 1)
        x = nn.functional.relu(self.conv1(x))
        x = self.maxpool(x)
        x = x.view(-1, 100)
        x = self.fc(x)
        return x

class TextCNN_enhanced(nn.Module):
    def __init__(self, embedding_dict, num_classes, max_seq_len):
        super(TextCNN_enhanced, self).__init__()
        embedding_dim = 100
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.embedding.weight.data.copy_(torch.from_numpy(np.random.normal(0, 1, (vocab_size, embedding_dim))))
        for word, idx in word_to_idx.items():
            if word in embedding_dict:
                self.embedding.weight.data[idx] = torch.from_numpy(embedding_dict[word])
        self.conv1 = nn.Conv1d(in_channels=embedding_dim, out_channels=100, kernel_size=3)
        self.conv2 = nn.Conv1d(in_channels=embedding_dim, out_channels=100, kernel_size=4)
        self.conv3 = nn.Conv1d(in_channels=embedding_dim, out_channels=100, kernel_size=5)
        self.maxpool = nn.MaxPool1d(max_seq_len - 3 - 4 - 5 + 3 + 1 + 1)
        self.fc = nn.Linear(300, num_classes)

    def forward(self, x):
        x = self.embedding(x)
        x = x.permute(0, 2, 1)
        x1 = nn.functional.relu(self.conv1(x))
        x2 = nn.functional.relu(self.conv2(x))
        x3 = nn.functional.relu(self.conv3(x))
        x1 = self.maxpool(x1)
        x2 = self.maxpool(x2)
        x3 = self.maxpool(x3)
        x = torch.cat((x1, x2, x3), dim=1)
        x = x.view(-1, 300)
        x = self.fc(x)
        return x


# ### Training the model
def train(model, optimizer, loss_fn, train_loader, val_loader, num_epochs, save_every_n_epochs):
    train_losses = []
    val_losses = []

    best_model = None
    best_val_loss = float("inf")

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            y_pred = model(X_batch)
            loss = loss_fn(y_pred, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        train_loss = total_loss / len(train_loader)
        train_losses.append(train_loss)
        
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                y_pred = model(X_batch)
                loss = loss_fn(y_pred, y_batch)
                val_loss += loss.item()
        val_loss /= len(val_loader)
        val_losses.append(val_loss)

        if val_loss < best_val_loss:
            best_model = model
            best_val_loss = val_loss

        if epoch % save_every_n_epochs == 0:
            # Save the model
            if type(model).__name__ == 'TextCNN_enhanced':
                best_model_path = f'../data/glove/glove_cnn_enhance_{epoch}.pth'
            elif type(model).__name__ == 'TextCNN':
                best_model_path = f'../data/glove/glove_cnn_{epoch}.pth'
            if best_model is not None:
                torch.save(best_model.state_dict(), best_model_path)

        print(f'Epoch {epoch}: train loss = {train_loss:.4f}, val loss = {val_loss:.4f}')

    return best_model, train_losses, val_losses

# ### Testing the model
def evaluate_model(test_model, test_loader):
    
    # Set the model to evaluation mode
    test_model.eval()

    # Calculate the accuracy on the test set
    num_correct = 0
    num_samples = 0
    with torch.no_grad():
        predicted_labels = []
        for X_batch, y_batch in test_loader:
            y_pred = test_model(X_batch)
            y_pred = y_pred.argmax(dim=1)
            predicted_labels.extend(y_pred.tolist())

        true_labels = []
        for X_batch, y_batch in test_loader:
            true_labels.extend(y_batch.tolist())

        true_labels = [torch.argmax(torch.tensor(batch_labels)) for batch_labels in true_labels]
        true_labels = torch.tensor(true_labels, dtype=torch.int64)
        predicted_labels = torch.tensor(predicted_labels)

        class_counts = torch.bincount(true_labels)
        correct_counts = torch.bincount(true_labels[predicted_labels == true_labels], minlength=len(class_counts))

        accuracy = float(correct_counts.sum()) / float(class_counts.sum())
        precision = float(correct_counts[1]) / float(class_counts[1])
        recall = float(correct_counts[1]) / float(class_counts[1] + class_counts[0])
        f1_score = 2 * (precision * recall) / (precision + recall)

    return accuracy, precision, recall, f1_score

if __name__ == '__main__':
    optparser = optparse.OptionParser()
    optparser.add_option("-c", "--classlabelfile", dest='classlabel', default=os.path.join('data', 'label.json'), help="class label dictionary stored in json")
    optparser.add_option("-d", "--decodeclasslabelfile", dest='decodeclasslabel', default= os.path.join('data', 'label_decode.json'), help="class label (reversed) dictionary used for decoding")
    optparser.add_option("-i", "--inputfile", dest="input", default=os.path.join('data', 'News_Category_Dataset_v3.json'), help="News Category Dataset")
    (opts, _) = optparser.parse_args()
    
    # print("Pre-processing the dataset")
    # processed_df = preprocess(inputfile="../data/News_Category_Dataset_v3.json", labelfile=opts.classlabel, decodelabelfile=opts.decodeclasslabel)
    
    # Alternative-way to use the save pre-processed file.
    processed_df = pd.read_csv("../data/processed_df.csv", header=0).drop('Unnamed: 0', axis=1)
    X = processed_df['text']
    y = processed_df['category']
    num_classes = len(y.unique())

    max_seq_len = 100
    batch_size = 32
    num_epochs = 1
    learning_rate = 0.001
    embedding_file_path = './data/glove/glove.6B.100d.txt'
    
    print("Loading the glove word embeddings")
    embedding_dict = load_embedding_dict(embedding_file_path)

    print("Create word to index mappings")
    word_to_idx, vocab_size = create_word_to_idx(X,embedding_dict)

    print("Splitting the dataset into training, validation and test set")
    train_texts, train_labels, val_texts, val_labels, test_texts, test_labels = preprocess_data(processed_df, word_to_idx, max_seq_len=100)
    
    print("Creating data loaders")
    train_loader, val_loader, test_loader = create_data_loaders(train_texts, train_labels, val_texts, val_labels, test_texts, test_labels, batch_size=32)

    # Training simple baseline model
    cnn_model = TextCNN(embedding_dict, num_classes = num_classes, max_seq_len = max_seq_len)
    optimizer = torch.optim.Adam(cnn_model.parameters(), lr = learning_rate)

    print("Training the baseline CNN model")
    train(cnn_model, optimizer, nn.CrossEntropyLoss(), train_loader, val_loader, num_epochs,2)

    # Training enhanced CNN model
    enhanced_cnn_model = TextCNN_enhanced(embedding_dict, num_classes = num_classes, max_seq_len = max_seq_len)
    optimizer = torch.optim.Adam(enhanced_cnn_model.parameters(), lr = learning_rate)

    print("Training Enhanced TextCNN model")
    train(enhanced_cnn_model, optimizer, nn.CrossEntropyLoss(), train_loader, val_loader, num_epochs,2)

    # Evaluation Results 
    print("Evaluation of Models")
    test_model = TextCNN(embedding_dict, num_classes = num_classes, max_seq_len = max_seq_len)

    # Load the saved model state dict
    state_dict = torch.load('../data/glove/glove_cnn_4.pth')

    # Load the state dict into the model
    test_model.load_state_dict(state_dict)

    # Set the model to evaluation mode
    test_model.eval()

    accuracy, precision, recall, f1_score = evaluate_model(test_model, test_loader)

    print("Evaluation Results for Baseline Model:\n"
        "Accuracy: {:.4f}\n"
        "Precision: {:.4f}\n"
        "Recall: {:.4f}\n"
        "F1 Score: {:.4f}\n".format(accuracy, precision, recall, f1_score))
    
    test_model = TextCNN_enhanced(embedding_dict, num_classes = num_classes, max_seq_len = max_seq_len)

    # Load the saved model state dict
    state_dict = torch.load('../data/glove/glove_cnn_enhance_2.pth')

    # Load the state dict into the model
    test_model.load_state_dict(state_dict)

    # Set the model to evaluation mode
    test_model.eval()

    accuracy, precision, recall, f1_score = evaluate_model(test_model, test_loader)

    print("Evaluation Results for Enhanced CNN model:\n"
        "Accuracy: {:.4f}\n"
        "Precision: {:.4f}\n"
        "Recall: {:.4f}\n"
        "F1 Score: {:.4f}\n".format(accuracy, precision, recall, f1_score))
    
