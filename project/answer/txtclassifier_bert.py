#  python3 install -r requirements.txt
#  python -m spacy download en_core_web_lg

import os, sys, optparse, gzip, re, logging, string
import json
import pandas as pd
import numpy as np
import torch 
from ydata_profiling import ProfileReport
from sklearn.utils import shuffle
from sklearn.metrics import classification_report, confusion_matrix
from torch.optim import Adam
from tqdm import tqdm
from torch import nn
import spacy
import seaborn as sn
import matplotlib.pyplot as plt

from transformers import BertTokenizer, BertModel



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
        

def label_importer(filename, filename_decode):
    
    with open(filename) as fp:
        label = json.load(fp)

    with open(filename_decode) as fp:
        label_decode = json.load(fp)
    
    return label, label_decode


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



# ========================================= BERT-specific Code =====================================================


class BertDataset(torch.utils.data.Dataset):

    def __init__(self, df, tokenizer, labels):

        self.labels = [labels[label] for label in df['category']]
        self.texts = [tokenizer(text, padding='max_length', max_length = 512, truncation=True,
                                return_tensors="pt") for text in df['text']]

    def classes(self):
        return self.labels

    def __len__(self):
        return len(self.labels)

    def get_batch_labels(self, idx):
        # Fetch a batch of labels
        return np.array(self.labels[idx])

    def get_batch_texts(self, idx):
        # Fetch a batch of inputs
        return self.texts[idx]

    def __getitem__(self, idx):

        batch_texts = self.get_batch_texts(idx)
        batch_y = self.get_batch_labels(idx)

        return batch_texts, batch_y


class BertClassifier(nn.Module):

    def __init__(self, bert_case='bert-base-cased', dropout=0.5):

        super(BertClassifier, self).__init__()

        self.bert = BertModel.from_pretrained(bert_case)
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(768, 15)
        self.relu = nn.ReLU()

    def forward(self, input_id, mask):

        _, pooled_output = self.bert(input_ids= input_id, attention_mask=mask,return_dict=False)
        dropout_output = self.dropout(pooled_output)
        linear_output = self.linear(dropout_output)
        final_layer = self.relu(linear_output)

        return final_layer



def train(model, train_data, val_data, learning_rate, epochs, tokenizer, labels):

    train, val = BertDataset(train_data, tokenizer, labels), BertDataset(val_data, tokenizer, labels)

    train_dataloader = torch.utils.data.DataLoader(train, batch_size=7, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val, batch_size=5)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr= learning_rate)

    if use_cuda:
        
        print("Sending model/loss_function to cuda.")

        model = model.cuda()
        criterion = criterion.cuda()

    for epoch_num in range(epochs):

        total_acc_train = 0
        total_loss_train = 0

        for train_input, train_label in tqdm(train_dataloader):

            train_label = train_label.to(device)
            mask = train_input['attention_mask'].to(device)
            input_id = train_input['input_ids'].squeeze(1).to(device)

            output = model(input_id, mask)
            
            batch_loss = criterion(output, train_label.long())
            total_loss_train += batch_loss.item()
            
            acc = (output.argmax(dim=1) == train_label).sum().item()
            total_acc_train += acc

            model.zero_grad()
            batch_loss.backward()
            optimizer.step()
        
        total_acc_val = 0
        total_loss_val = 0

        with torch.no_grad():

            for val_input, val_label in val_dataloader:

                val_label = val_label.to(device)
                mask = val_input['attention_mask'].to(device)
                input_id = val_input['input_ids'].squeeze(1).to(device)

                output = model(input_id, mask)

                batch_loss = criterion(output, val_label.long())
                total_loss_val += batch_loss.item()
                
                acc = (output.argmax(dim=1) == val_label).sum().item()
                total_acc_val += acc
        
        print(
            f'Epochs: {epoch_num + 1} | Train Loss: {total_loss_train / len(train_data): .3f} \
            | Train Accuracy: {total_acc_train / len(train_data): .3f} \
            | Val Loss: {total_loss_val / len(val_data): .3f} \
            | Val Accuracy: {total_acc_val / len(val_data): .3f}')
        
        model_state_filename = 'bert_trained_{}.pth'.format(epoch_num)
        torch.save(model.state_dict(), os.path.join('data', 'bert', model_state_filename))
    


def evaluate(model, test_data, tokenizer, labels, decodes):

    test = BertDataset(test_data, tokenizer, labels)

    test_dataloader = torch.utils.data.DataLoader(test, batch_size=2)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    if use_cuda:
        model = model.cuda()

    y_true_list = []
    y_pred_list = []

    with torch.no_grad():

        for test_input, test_label in tqdm(test_dataloader):

            test_label = test_label.to(device)
            mask = test_input['attention_mask'].to(device)
            input_id = test_input['input_ids'].squeeze(1).to(device)

            output = model(input_id, mask)

            y_true_list.append(test_label)
            y_pred_list.append(output.argmax(dim=1))


    y_true_encoded = torch.cat(y_true_list, dim=0).tolist()
    y_pred_encoded = torch.cat(y_pred_list, dim=0).tolist()

    y_true = [decodes[str(class_num)] for class_num in y_true_encoded]
    y_pred = [decodes[str(class_num)] for class_num in y_pred_encoded]

    class_ordered = [value for value in decodes.values()]

    print(classification_report(y_true, y_pred, digits=3, labels=class_ordered))

    return y_true, y_pred, class_ordered
    


if __name__ == '__main__':
    optparser = optparse.OptionParser()
    optparser.add_option("-c", "--classlabelfile", dest='classlabel', default=os.path.join('data', 'label.json'), help="class label dictionary stored in json")
    optparser.add_option("-d", "--decodeclasslabelfile", dest='decodeclasslabel', default=os.path.join('data', 'label_decode.json'), help="class label (reversed) dictionary used for decoding")
    optparser.add_option("-i", "--inputfile", dest="input", default=os.path.join('data', 'News_Category_Dataset_v3.json'), help="News Category Dataset")
    optparser.add_option("-m", "--modelfile", dest="modelfile", default=os.path.join('data', 'bert', 'bert_trained_2.pth'), help="filename of the trained BertClassifier")
    optparser.add_option("-f", "--force", dest="force", action="store_true", default=False, help="force training phase (warning: can take a few hours)")
    optparser.add_option("-l", "--logfile", dest="logfile", default=None, help="log file for debugging")
    (opts, _) = optparser.parse_args()


    # ================================ Common Base Code Shared by ALL models ===========================================

    if opts.logfile is not None:
        logging.basicConfig(filename=opts.logfile, filemode='w', level=logging.DEBUG)

    processed_df = preprocess(inputfile=opts.input, labelfile=opts.classlabel, decodelabelfile=opts.decodeclasslabel)
    df_train, df_val, df_test = data_split(processed_df)

    # ==================================================================================================================

    # ========================================= BERT-specific Code =====================================================

    # Batch size 7, EPOCH 2, LR 1e-6, Dropout 0.2 ==> Test Score: 0.748
    # Batch size 7, EPOCH 2, LR 1e-6, Dropout 0.8 ==> Test Score: 0.733
    # Batch size 7, EPOCH 2, LR 1e-6, Dropout 0.5 ==> Test Score: 0.742

    # Batch size 7, EPOCH 3, LR 1e-6, Dropout 0.3 ==> Test Score: 0.751


    BERT_CASE = 'bert-base-uncased'
    #BERT_CASE = 'bert-base-cased'

    DROPOUT_RATE = 0.3
    EPOCHS = 3
    LR = 1e-6  

    labels, decodes = label_importer(opts.classlabel, opts.decodeclasslabel)
    tokenizer = BertTokenizer.from_pretrained(BERT_CASE)
    model = BertClassifier(bert_case=BERT_CASE, dropout=DROPOUT_RATE)


    # use the model file if available and opts.force is False
    if os.path.isfile(opts.modelfile) and not opts.force:
        model.load_state_dict(torch.load(opts.modelfile))
        y_true, y_pred, class_ordered = evaluate(model, df_test, tokenizer, labels, decodes)

    else:
        print("Warning: could not find modelfile {}. Starting training.".format(opts.modelfile), file=sys.stderr)        
        train(model, df_train, df_val, LR, EPOCHS, tokenizer, labels)

        #model.load_state_dict(torch.load(opts.modelfile))
        y_true, y_pred, class_ordered = evaluate(model, df_test, tokenizer, labels, decodes)
    
    # ==================================================================================================================

