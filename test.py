import numpy as np
from PIL import Image
import torch
import glob2 as glob
import os

k = 5

imagenet_labels = dict(enumerate(open("classes.txt")))

model = torch.load("model.pth")
model.eval()

for iter,filename in enumerate(glob.iglob('images/*.png', recursive=True)):
    print(F"The {iter+1} image is : {filename}")

    img = (np.array(Image.open(filename)) / 128) - 1  # in the range -1, 1
    inp = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(torch.float32)
    logits = model(inp)
    probs = torch.nn.functional.softmax(logits, dim=-1) # Turn logits into probs.

    # Get top k (k=5) probabilities.
    top_probs, top_ixs = probs[0].topk(k)

    for i, (ix_, prob_) in enumerate(zip(top_ixs, top_probs)):
        ix = ix_.item()
        prob = prob_.item()
        cls = imagenet_labels[ix].strip()
        print(f"{i}: {cls:<45} --- {prob:.4f}\n")