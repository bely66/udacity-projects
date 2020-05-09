import torch
import torch.nn as nn
import torchvision.models as models


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features
    

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        
        super().__init__()
        self.n_hidden = hidden_size
        self.embed_size = embed_size
        self.vocab_size = vocab_size
        
        # Embeddings 
        self.embedding = nn.Embedding(vocab_size, embed_size)
        
        # Define the LSTM
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        
        # Define the fully-connected output layer
        self.fc1 = nn.Linear(hidden_size, vocab_size)
        
        #initialize weights
        self.init_weights()
        
        
    def init_weights(self):
        torch.nn.init.xavier_uniform_(self.fc1.weight)
        torch.nn.init.xavier_uniform_(self.embedding.weight)
        
    def forward(self, features, captions):
        
        captions = captions[:, :-1]
        captions = self.embedding(captions)
        
        
        features = features.unsqueeze(1)
        inputs = torch.cat((features, captions), 1)
        outputs, _ = self.lstm(inputs)
        
        
        outputs = self.fc1(outputs)
        
        return outputs

    def sample(self, inputs, states=None, max_len=20):
        
        preds = []
        count = 0
        word_i = None
        
        while count < max_len and word_i != 1 :
            
            #Predict output
            output_lstm, states = self.lstm(inputs, states)
            output = self.fc1(output_lstm)
            
            #Get max value
            prob, word = output.max(2)
            
            #append word
            word_i = word.item()
            preds.append(word_i)
            
            #next input is current prediction
            inputs = self.embedding(word)
            
            count+=1
        
        return preds