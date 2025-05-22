import pandas as pd
from PIL import image
from transformers import AutoTokenizer, Blip2QFormerModel, AutoModelForSeq2SeqLM
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# load dataset
df = pd.read_csv("your_dataset.csv")

# preprocess
for index, row in df.iterrows():
    image_path = row["image_path"]
    image = Image.open(image_path)
    image = image.resize((224, 224))
    image = np.array(image)
    df.loc[index, "image"] = image

# preprocess text
tokenizer = AutoTokenizer.from_pretraines("salesforce/blip-2-opt-2.7b")

for index, row in df.iterrows():
    text = row["text"]
    encoded_text = tokenizer(text=text, return_tensors="pt")
    df.loc[index, "encoded_text"] = encoded_text

# split into training, val, and testing
train_df, val_df, test_df = train_test_split(df, test_size=0.2)

# load model
model = Blip2QFormerModel.from_pretrained("salesforce/blip-2-opt-2.7b")

# get device
device = torch.device("cuda" if cuda.is_available() else "cpu")

# dataloaders
train_data = QFormerDataset(val_df)
