
from asyncio.windows_events import NULL
import discord
import time
import random
# from keepalive import keep_alive
import json
import torch 
from model import NeuralNet
from nltk_utils import bag_of_words,tokenize
import youtube_dl
from threading import Thread
from main2 import mainf
# import os


TOKEN = NULL

response1 = ["Sorry",
            "I don't understand" ]


         

client = discord.Client()

print(torch.cuda.is_available())

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('intents.json','r',encoding='utf-8') as f:
    intents = json.load(f)

FILE = "data.pth"

data = torch.load(FILE)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data["all_words"]
tags = data["tags"]
model_state = data["model_state"]

model = NeuralNet(input_size,hidden_size,output_size).to(device)
model.load_state_dict(model_state)
model.eval()

idx1 = 0

@client.event #event decorator/wrapper
async def on_ready():
    print(f"We have logged in as {client.user}")
    

@client.event
async def on_message(message):
    global idx1
    try :
        if not "*play" in message.content.lower():
            if not "*skip" in message.content.lower():
                if message.content.startswith('*'):

                    
                    sentence = message.content[1:]
                    
                    sentence = tokenize(sentence)
                    X = bag_of_words(sentence,all_words)
                    X = X.reshape(1,X.shape[0])
                    X = torch.from_numpy(X)

                    output = model(X)

                    _,predicted = torch.max(output,dim=1)
                    tag = tags[predicted.item()]

                    probs = torch.softmax(output,dim=1)
                    prob = probs[0][predicted.item()]
                    
                    if prob.item() > 0.60:
                        for intent in intents["intents"]:
                            if tag == intent["tag"]:
                                
                                idx2 = random.randint(0,len(intent['responses'])-1)
                                # print("response length: ",len(intent['responses']))
                                # print("idx1",idx1)
                                # print("idx2",idx2)

                                if idx2 is idx1:
                                    await message.channel.send(intent['responses'][0])
                                    idx2 = random.randint(0,len(intent['responses'])-1)

                                if idx2 is not idx1:
                                    await message.channel.send(intent['responses'][idx2])
                                    idx1 = idx2
                                    # print("after send idx1",idx1)

                                # await message.channel.send(f"{random.choice(intent['responses'])}")
                                # print(f"{bot_name}:{random.choice(intent['responses'])}")
                    else :
                        idx = random.randint(0,len(response1)-1)
                        await message.channel.send(str(response1[idx]))

       

    except:
        pass         





client.run(TOKEN)   