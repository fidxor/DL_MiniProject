import torch
import numpy as np
from torch import nn
from torch.optim import Adam
from transformers import BertModel
from transformers import BertTokenizer

__all__ = ['Dataset', 'BertClassifier']

tokenizer = BertTokenizer.from_pretrained('bert-large-uncased')

labels = {"INFJ" : 0, "INTJ" : 1, "INFP" : 2, "INTP" : 3, "ENFJ" : 4, "ENTJ" : 5, 
              "ENFP" : 6, "ENTP" : 7, "ISFJ" : 8, "ISTJ" : 9, "ISFP" : 10, "ISTP" : 11,
                "ESFJ" : 12, "ESTJ" : 13, "ESFP" : 14, "ESTP" : 15}

# 결과 출력을 위해 labels key와 value 바꿔주기
resultLabels = {v:k for k,v in labels.items()}

class Dataset(torch.utils.data.Dataset):   
    
    def __init__(self, df):

        self.labels = [labels[label] for label in df['type']]
        self.texts = [tokenizer(text, 
                               padding='max_length', max_length = 512, truncation=True,
                                return_tensors="pt") for text in df['posts']]

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

    def __init__(self, dropout=0.5):

        super(BertClassifier, self).__init__()

        self.bert = BertModel.from_pretrained('bert-large-uncased')
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(1024, 16)
        self.relu = nn.ReLU()

    def forward(self, input_id, mask):

        _, pooled_output = self.bert(input_ids= input_id, attention_mask=mask,return_dict=False)
        dropout_output = self.dropout(pooled_output)
        linear_output = self.linear(dropout_output)
        final_layer = self.relu(linear_output)

        return final_layer
           
def Get_Device(model):
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    if use_cuda:

        model = model.cuda()

    return device
        
def Get_Device_Optimizer(model, learning_rate):
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr= learning_rate)

    if use_cuda:

            model = model.cuda()
            criterion = criterion.cuda()

    return device, criterion, optimizer

def GetModelOutput(model, input, label, device):
    label = label.to(device)
    mask = input['attention_mask'].to(device)
    input_id = input['input_ids'].squeeze(1).to(device)

    output = model(input_id, mask)

    return output


def Get_Total_Accuracy(modelOutput, label):
    total_accuracy = 0
    acc = (modelOutput.argmax(dim = 1) == label).sum().item()
    total_accuracy += acc

    return total_accuracy