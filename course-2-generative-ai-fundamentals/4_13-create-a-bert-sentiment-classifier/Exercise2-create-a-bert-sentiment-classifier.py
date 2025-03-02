#!/usr/bin/env python
# coding: utf-8

# # Exercise: Create a BERT sentiment classifier
# 
# In this exercise, you will create a BERT sentiment classifier (actually DistilBERT) using the [Hugging Face Transformers](https://huggingface.co/transformers/) library. 
# 
# You will use the [IMDB movie review dataset](https://huggingface.co/datasets/imdb) to train and evaluate your model. The IMDB dataset contains movie reviews that are labeled as either positive or negative. 

# In[1]:


# Install the required version of datasets in case you have an older version
# You will need to choose "Kernel -> Restart Kernel" from the menu after executing this cell
get_ipython().system('pip install datasets==3.2.0 huggingface_hub==0.28.1')


# In[2]:


# Import the datasets and transformers packages

from datasets import load_dataset

# Load the train and test splits of the imdb dataset
splits = ["train", "test"]
ds = {split: ds for split, ds in zip(splits, load_dataset("imdb", split=splits))}

# Thin out the dataset to make it run faster for this example
for split in splits:
    ds[split] = ds[split].shuffle(seed=42).select(range(500))

# Show the dataset
ds


# ## Pre-process datasets
# 
# Now we are going to process our datasets by converting all the text into tokens for our models. You may ask, why isn't the text converted already? Well, different models may use different tokenizers, so by converting at train time we retain more flexibility.

# In[5]:


# Replace <MASK> with your code that constructs a query to send to the LLM


from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")


def preprocess_function(examples):
    """Preprocess the imdb dataset by returning tokenized examples."""
    return tokenizer(examples["text"], padding="max_length", truncation=True)


tokenized_ds = {}
for split in splits:
    tokenized_ds[split] = ds[split].map(preprocess_function, batched=True)


# Check that we tokenized the examples properly
assert tokenized_ds["train"][0]["input_ids"][:5] == [101, 2045, 2003, 2053, 7189]

# Show the first example of the tokenized training set
print(tokenized_ds["train"][0]["input_ids"])


# ## Load and set up the model
# 
# We will now load the model and freeze most of the parameters of the model: everything except the classification head.

# In[7]:


# Replace <MASK> with your code freezes the base model

from transformers import AutoModelForSequenceClassification

model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
    num_labels=2,
    id2label={0: "NEGATIVE", 1: "POSITIVE"},  # For converting predictions to strings
    label2id={"NEGATIVE": 0, "POSITIVE": 1},
)

# Freeze all the parameters of the base model
# Hint: Check the documentation at https://huggingface.co/transformers/v4.2.2/training.html
for param in model.base_model.parameters():
    param.requires_grad = False

model.classifier


# In[8]:


print(model)


# ## Let's train it!
# 
# Now it's time to train our model. We'll use the `Trainer` class from the 🤗 Transformers library to do this. The `Trainer` class provides a high-level API that abstracts away a lot of the training loop.
# 
# First we'll define a function to compute our accuracy metreic then we make the `Trainer`.
# 
# Let's take this opportunity to learn about the `DataCollator`. According to the HuggingFace documentation:
# 
# > Data collators are objects that will form a batch by using a list of dataset elements as input. These elements are of the same type as the elements of train_dataset or eval_dataset.
# 
# > To be able to build batches, data collators may apply some processing (like padding).
# 

# In[9]:


# Replace <MASK> with your DataCollatorWithPadding argument(s)

import numpy as np
from transformers import DataCollatorWithPadding, Trainer, TrainingArguments


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return {"accuracy": (predictions == labels).mean()}


# The HuggingFace Trainer class handles the training and eval loop for PyTorch for us.
# Read more about it here https://huggingface.co/docs/transformers/main_classes/trainer
trainer = Trainer(
    model=model,
    args=TrainingArguments(
        output_dir="./data/sentiment_analysis",
        learning_rate=2e-3,
        # Reduce the batch size if you don't have enough memory
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        num_train_epochs=1,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
    ),
    train_dataset=tokenized_ds["train"],
    eval_dataset=tokenized_ds["test"],
    tokenizer=tokenizer,
    data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
    compute_metrics=compute_metrics,
)

trainer.train()


# ## Evaluate the model
# 
# Evaluating the model is as simple as calling the evaluate method on the trainer object. This will run the model on the test set and compute the metrics we specified in the compute_metrics function.

# In[10]:


# Show the performance of the model on the test set
# What do you think the evaluation accuracy will be?
trainer.evaluate()


# ### View the results
# 
# Let's look at two examples with labels and predicted values.

# In[11]:


import pandas as pd

df = pd.DataFrame(tokenized_ds["test"])
df = df[["text", "label"]]

# Replace <br /> tags in the text with spaces
df["text"] = df["text"].str.replace("<br />", " ")

# Add the model predictions to the dataframe
predictions = trainer.predict(tokenized_ds["test"])
df["predicted_label"] = np.argmax(predictions[0], axis=1)

df.head(2)


# ### Look at some of the incorrect predictions
# 
# Let's take a look at some of the incorrectly-predcted examples

# In[12]:


# Show full cell output
pd.set_option("display.max_colwidth", None)

df[df["label"] != df["predicted_label"]].head(2)


# ## End of exercise
# 
# Great work! 🤗

# In[ ]:




