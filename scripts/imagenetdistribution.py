import os
import json
import yaml

"""
def synset2idx(path_to_yaml="/jmain02/home/J2AD015/axf03/cxw11-axf03/.cache/autoencoders/data/ILSVRC2012_train/index_synset.yaml"):
    with open(path_to_yaml) as f:
        di2s = yaml.safe_load(f)
    return dict((v,k) for k,v in di2s.items())

dataset_directory = '/jmain02/home/J2AD015/axf03/cxw11-axf03/.cache/autoencoders/data/ILSVRC2012_train/data'
class_labels = os.listdir(dataset_directory)

synset2idx = synset2idx()

class_counts = {}
for label in class_labels:
    class_directory = os.path.join(dataset_directory, label)
    if os.path.isdir(class_directory):
        class_count = len(os.listdir(class_directory))
        class_counts[synset2idx[label]] = class_count
"""

import torch
from torch.utils.data import DataLoader, Dataset
from ldm.data.imagenet import ImageNetTrain
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True



config = {'size': 256}

imagenet_train = ImageNetTrain(config)

batch_size = 1
data_loader = DataLoader(imagenet_train, batch_size=batch_size, shuffle=False)

class_counts = {}

# Iterate over the DataLoader
#i=0
for batch in data_loader:
    class_label = batch['class_label'][0].item()
    #print(class_label)
    if class_label not in class_counts:
        class_counts[class_label] = 1
    else:
        class_counts[class_label] += 1
    #i += 1
    #if i == 100:
    #    print(class_counts)

print(class_counts)

output_file = 'distribution.txt'
with open(output_file, 'w') as f:
    json.dump(class_counts, f)

# Read the distribution from the text file and extract a dictionary
with open(output_file, 'r') as f:
    distribution_dict = json.load(f)

print(distribution_dict)