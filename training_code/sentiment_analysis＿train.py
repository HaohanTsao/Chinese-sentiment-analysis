import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import torch
from transformers import BertModel, BertTokenizer
import re
import seaborn as sns
import os
import warnings
warnings.filterwarnings('ignore')
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from torch import nn, optim
import torch.nn.functional as F
from collections import defaultdict
import time
from transformers import AdamW, get_linear_schedule_with_warmup
# 文本清理函數
def text_clean(content):
  cleaned_content = content
  cleaned_content = re.sub(r'[\n\r]', '', cleaned_content) # 換行符號
  cleaned_content = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', ' ', cleaned_content) # 移除url
  cleaned_content = re.sub(r'<.*?>', '', cleaned_content) # 移除html標籤
  cleaned_content = re.sub(r'\d+', '', cleaned_content) # 數字
  cleaned_content = re.sub(r'[^\w\s]', '', cleaned_content) # 移除標點符號
  cleaned_content = re.sub(r"\s+", "", cleaned_content) # 清除空格
  return cleaned_content
# 準備測試資料
selected_columns = ['content', 'sentiment']
folder_path = './training_data'
bert_test_data = pd.DataFrame()
for filename in os.listdir(folder_path):
    df_temp = pd.read_csv(f'{folder_path}/{filename}')
    bert_test_data = pd.concat([bert_test_data, df_temp[selected_columns]], ignore_index=True)
# Drop duplicate rows
bert_test_data.drop_duplicates(inplace=True)
bert_test_data = bert_test_data.dropna(axis=0, how='any', subset=['content'])
bert_test_data["sentiment"] = bert_test_data["sentiment"].replace({'0': 0, '1': 1, '2': 2})
# Assuming your DataFrame is called 'df' and the column you want to check is 'column_name'
invalid_values = [0, 1, 2]
# Drop the rows where the values in 'column_name' are in the invalid_values list
bert_test_data = bert_test_data[bert_test_data['sentiment'].isin(invalid_values)]
# Apply text clean function
bert_test_data['cleaned_content'] = bert_test_data['content'].apply(text_clean)
# 載入我們的模型
bert_model = BertModel.from_pretrained("./model_file")
tokenizer = BertTokenizer.from_pretrained("./model_file")
# ### Choosing Sequence Length，實際資料要再調整
test_data = bert_test_data
# ### 建立pytorch dataset
class ContentDataset(Dataset):

  def __init__(self, contents, targets, tokenizer, max_len):
    self.contents = contents
    self.targets = targets
    self.tokenizer = tokenizer
    self.max_len = max_len
  
  def __len__(self):
    return len(self.contents)
  
  def __getitem__(self, item):
    content = str(self.contents[item])
    target = int(self.targets[item])

    encoding = self.tokenizer.encode_plus(
      content,
      add_special_tokens=True,
      max_length=self.max_len,
      return_token_type_ids=False,
      pad_to_max_length=True,
      return_attention_mask=True,
      return_tensors='pt',
    )

    return {
      'content_text': content,
      'input_ids': encoding['input_ids'].flatten(),
      'attention_mask': encoding['attention_mask'].flatten(),
      'targets': torch.tensor(target, dtype=torch.long)
    }

df_train, df_val = train_test_split(test_data, test_size=0.2, random_state=2023)

def create_data_loader(df, tokenizer, max_len, batch_size):
  ds = ContentDataset(
    contents=df['cleaned_content'].to_numpy(),
    targets=df['sentiment'].to_numpy(),
    tokenizer=tokenizer,
    max_len=max_len
  )

  return DataLoader(
    ds,
    batch_size=batch_size,
    num_workers=4
  )

BATCH_SIZE = 32
MAX_LENGTH = 350
train_data_loader = create_data_loader(df_train, tokenizer, MAX_LENGTH, BATCH_SIZE)
val_data_loader = create_data_loader(df_val, tokenizer, MAX_LENGTH, BATCH_SIZE)

# class SentimentClassifier(nn.Module):
#     def __init__(self, n_classes):
#         super(SentimentClassifier, self).__init__()
#         self.bert = BertModel.from_pretrained("./model_file")
#         self.drop = nn.Dropout(p=0.5)
#         self.transformer = nn.TransformerEncoderLayer(self.bert.config.hidden_size, nhead=8)
#         self.fc = nn.Linear(self.bert.config.hidden_size, n_classes)
#         self.relu = nn.ReLU()

#     def forward(self, input_ids, attention_mask):
#         outputs = self.bert(
#             input_ids=input_ids,
#             attention_mask=attention_mask
#         )
#         pooled_output = outputs[1]
#         output = self.drop(pooled_output)
#         output = self.transformer(output.unsqueeze(0))  # Add Transformer layer
#         output = output.squeeze(0)
#         output = self.fc(output)
#         return output
# class SentimentClassifier(nn.Module):
#     def __init__(self, n_classes):
#         super(SentimentClassifier, self).__init__()
#         self.bert = BertModel.from_pretrained("./model_file")
#         self.drop = nn.Dropout(p=0.5)  # Increased dropout to 0.5
#         self.fc1 = nn.Linear(self.bert.config.hidden_size, 256)
#         self.fc2 = nn.Linear(256, 128)  # Additional fully connected layer
#         self.fc3 = nn.Linear(128, n_classes)  # Additional fully connected layer
#         self.relu = nn.ReLU()

#     def forward(self, input_ids, attention_mask):
#         outputs = self.bert(
#             input_ids=input_ids,
#             attention_mask=attention_mask
#         )
#         pooled_output = outputs[1]
#         output = self.drop(pooled_output)
#         output = self.fc1(output)
#         output = self.relu(output)
#         output = self.drop(output)
#         output = self.fc2(output)
#         output = self.relu(output)
#         output = self.drop(output)
#         output = self.fc3(output)
#         return output

class SentimentClassifier(nn.Module):

  def __init__(self, n_classes):
    super(SentimentClassifier, self).__init__()
    self.bert = BertModel.from_pretrained("./model_file")
    self.drop = nn.Dropout(p=0.3)
    self.out = nn.Linear(self.bert.config.hidden_size, n_classes)
  
  def forward(self, input_ids, attention_mask):
    outputs = self.bert(
        input_ids=input_ids,
        attention_mask=attention_mask
    )
    pooled_output = outputs[1]
    output = self.drop(pooled_output)
    return self.out(output)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device
class_names = ['N', 'M', 'P']
model = SentimentClassifier(len(class_names))
# # Load the state dict into the model
# state_dict_path = "best_model_state_add_layer.bin"
# state_dict = torch.load(state_dict_path)
# model.load_state_dict(state_dict)
model = model.to(device)
EPOCHS = 20

optimizer = AdamW(model.parameters(), lr=2e-5, correct_bias=False)
total_steps = len(train_data_loader) * EPOCHS

scheduler = get_linear_schedule_with_warmup(
  optimizer,
  num_warmup_steps=0,
  num_training_steps=total_steps
)
loss_fn = nn.CrossEntropyLoss().to(device)
# using gpu
def train_epoch_gpu(
  model, 
  data_loader, 
  loss_fn, 
  optimizer, 
  device, 
  scheduler, 
  n_examples
):
  model = model.train()

  losses = []
  correct_predictions = 0
  
  for d in data_loader:
    input_ids = d["input_ids"].to(device)
    attention_mask = d["attention_mask"].to(device)
    targets = d["targets"].to(device)
    targets = torch.tensor(targets, dtype=torch.long)  # Create the tensor

    

    outputs = model(
      input_ids=input_ids,
      attention_mask=attention_mask
    )

    _, preds = torch.max(outputs, dim=1)
    loss = loss_fn(outputs, targets)

    correct_predictions += torch.sum(preds == targets)
    losses.append(loss.item())

    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad()

  return correct_predictions.double() / n_examples, np.mean(losses)

def eval_model_gpu(model, data_loader, loss_fn, device, n_examples):
  model = model.eval()

  losses = []
  correct_predictions = 0

  with torch.no_grad():
    for d in data_loader:
      input_ids = d["input_ids"].to(device)
      attention_mask = d["attention_mask"].to(device)
      targets = d["targets"].to(device)

      outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask
      )
      _, preds = torch.max(outputs, dim=1)

      loss = loss_fn(outputs, targets)

      correct_predictions += torch.sum(preds == targets)
      losses.append(loss.item())

  return correct_predictions.double() / n_examples, np.mean(losses)


# training
start_time = time.time()
history = defaultdict(list)
best_accuracy = 0
for epoch in range(EPOCHS):

  print(f'Epoch {epoch + 1}/{EPOCHS}')
  print('-' * 10)

  train_acc, train_loss = train_epoch_gpu(
    model,
    train_data_loader,    
    loss_fn, 
    optimizer, 
    device, 
    scheduler, 
    len(df_train)
  )

  print(f'Train loss {train_loss} accuracy {train_acc}')

  val_acc, val_loss = eval_model_gpu(
    model,
    val_data_loader,
    loss_fn, 
    device, 
    len(df_val)
  )

  print(f'Val   loss {val_loss} accuracy {val_acc}')
  print()

  history['train_acc'].append(train_acc)
  history['train_loss'].append(train_loss)
  history['val_acc'].append(val_acc)
  history['val_loss'].append(val_loss)

  if val_acc > best_accuracy:
    torch.save(model.state_dict(), 'best_model_state.bin')
    best_accuracy = val_acc
end_time = time.time()
elapsed_time = end_time - start_time
print('time:{:02d}m{:02d}s'.format(int(elapsed_time // 60), int(elapsed_time % 60)))

# plot the train and validation accuracy
train_acc = [history['train_acc'][i].tolist() for i in range(20)]
val_acc = [history['val_acc'][i].tolist() for i in range(20)]
plt.plot(train_acc, label='train accuracy')
plt.plot(val_acc, label='validation accuracy')
plt.title('Training history')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend()
plt.ylim([0, 1])
# save the fig
plt.savefig('train_val_acc.png')