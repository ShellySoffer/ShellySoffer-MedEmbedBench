

import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import torch
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import roc_curve, auc
import random
from transformers import AutoTokenizer, AutoModel, BloomModel
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm  # Import tqdm for the progress bar
import os
import gc  # Import garbage collector


os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'


import re
from fractions import Fraction


def create_chimeric_description_randomly (desc1, desc2, ratio):
    
    # Convert descriptions into lists of sentences
    sentences1 = re.split(r'(?<=[.!?]) +', desc1)
    sentences2 = re.split(r'(?<=[.!?]) +', desc2)
    
    # Convert the ratio to a fraction to determine the exact numbers to grab
    frac = Fraction(ratio).limit_denominator()
    grab_a = frac.numerator
    grab_b = frac.denominator - frac.numerator

    result = []
    current_num_a, current_num_b = 0, 0

    # Ensure to update total_a and total_b
    total_a = len(sentences1)
    total_b = len(sentences2)

    while current_num_a < total_a or current_num_b < total_b:
        # Grab sentences from desc1
        for _ in range(grab_a):
            if current_num_a < len(sentences1):
                result.append(sentences1[current_num_a])
                current_num_a += 1

        # Grab sentences from desc2
        for _ in range(grab_b):
            if current_num_b < len(sentences2):
                result.append(sentences2[current_num_b])
                current_num_b += 1

        # Check if either description has been exhausted
        if current_num_a >= len(sentences1) or current_num_b >= len(sentences2):
            break

    return ' '.join(result)


def create_chimeric_description(desc_x, desc_z, ratio):
    
    split_x = int(len(desc_x) * ratio)
    split_z = int(len(desc_z) * ratio)
    
    start_desc = desc_x[:split_x]
    
    end_desc = desc_z[split_z:]
    
    return start_desc + end_desc


class SentenceEmbedder:
    
    def __init__(self, model_name, batch_size):
        
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        
        self.model = SentenceTransformer(model_name)
        
        if torch.cuda.is_available():
            self.model = self.model.to('cuda')
            
        self.batch_size = batch_size
    
    
    def embed_sentences (self, texts):
        # List to store all embeddings
        all_embeddings = []
        
        # Process texts in batches with a tqdm progress bar
        for i in tqdm(range(0, len(texts), self.batch_size), 
                      desc="Processing batches"):

           batch_texts = texts[i:i + self.batch_size]
           
           embeddings = self.model.encode (batch_texts, 
                                            convert_to_tensor=True)
           
           all_embeddings.append(embeddings.cpu().numpy())
        
        # Concatenate all batch embeddings
        all_embeddings = np.vstack(all_embeddings)
        return all_embeddings
    
    
def run_experiment(df, model_name, task, batch_size = 4):
    
    embedder = SentenceEmbedder (model_name =  model_name, 
                                 batch_size = batch_size)
    
    # Embed all descriptions and summaries in batches
    original_descriptions = embedder.embed_sentences (df['source'].tolist())
    original_summaries = embedder.embed_sentences (df['destination'].tolist())

    data = []

    for i in range(len(df)):
        data.append((df['source'].tolist()[i], 
                     df['destination'].tolist()[i],
                     original_descriptions[i], 
                     original_summaries[i], True, 1.0))

    ratios = [0.0, 0.25, 0.5, 0.75]
    chimeric_descriptions = []
    chimeric_metadata = []

    destination_list = []
           
    for ratio in ratios:
        for i in range(len(df)):
            second_half_indices = list(range(len(df)))
            second_half_indices.remove(i)
            second_half_index = random.choice(second_half_indices)

            if task == "imaging" or task == "complaint":
                chimeric_description = create_chimeric_description(
                    df['source'][i], df['source'][second_half_index], ratio)
            else:
                chimeric_description = create_chimeric_description_randomly(
                    df['source'][i], df['source'][second_half_index], ratio)
           
            destination_list.append (df ["destination"][i])
            
            chimeric_descriptions.append(chimeric_description)
            
            chimeric_metadata.append((i, second_half_index, ratio))

    # Embed all chimeric descriptions in one batch
    chimeric_description_embeddings = embedder.embed_sentences (
        chimeric_descriptions)

    # Add chimeric description data
    for idx, embedding in enumerate (chimeric_description_embeddings):
        i, second_half_index, ratio = chimeric_metadata[idx]
        data.append((
            chimeric_descriptions [idx],
            destination_list [idx],
            embedding, 
            original_summaries[i], False, ratio))

    new_df = pd.DataFrame(
        data, columns=['source', 
                       'destination', 
                       'source_embedding', 
                       'destination_embedding', 
                       'label', 
                       'type'])
    
    new_df['cosine_similarity'] = new_df.apply(
        lambda row: cosine_similarity(row['source_embedding'].reshape(1, -1), 
                                      row['destination_embedding'].reshape(1, -1))[0][0], 
        axis=1)

    y_true = new_df['label'].astype(int).values
    y_scores = new_df['cosine_similarity'].values

    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)

    print(f"AUC: {roc_auc}")

    return new_df



# In[13]:


models = [
# 100
# "avsolatorio/NoInstruct-small-Embedding-v0",
"avsolatorio/GIST-small-Embedding-v0",
# "infgrad/stella-base-en-v2",
"BAAI/bge-small-en-v1.5",
"thenlper/gte-small",
"sentence-transformers/all-MiniLM-L12-v2",
"sentence-transformers/all-MiniLM-L6-v2",

# 100-250
#"Alibaba-NLP/gte-base-en-v1.5",
"avsolatorio/GIST-Embedding-v0",
"sentence-transformers/all-mpnet-base-v2",
"thenlper/gte-base",
#"nomic-ai/nomic-embed-text-v1",
"dwzhu/e5-base-4k",
# "jinaai/jina-embeddings-v2-base-en",
"emilyalsentzer/Bio_ClinicalBERT",
    
# 250-500
#"Alibaba-NLP/gte-large-en-v1.5",
"mixedbread-ai/mxbai-embed-large-v1",
"WhereIsAI/UAE-Large-V1",
"avsolatorio/GIST-large-Embedding-v0",
"ekorman-strive/bge-large-en-v1.5",
"w601sxs/b1ade-embed",
    
# 500-1000
"intfloat/multilingual-e5-large-instruct",
"intfloat/multilingual-e5-large",
"OrdalieTech/Solon-embeddings-large-0.1",
"manu/bge-m3-custom-fr",
"sdadas/mmlw-e5-large",

# 1000-5000
# "McGill-NLP/LLM2Vec-Sheared-LLaMA-mntp-supervised",
"hkunlp/instructor-xl",
"sentence-transformers/sentence-t5-xl",
"sentence-transformers/sentence-t5-xxl",
"sentence-transformers/gtr-t5-xxl",
"Muennighoff/SGPT-2.7B-weightedmean-msmarco-specb-bitfit",
    
#5000-10000
"Salesforce/SFR-Embedding-Mistral",
#"Alibaba-NLP/gte-Qwen1.5-7B-instruct",
# "GritLM/GritLM-7B",
# "LLM2Vec-Meta-Llama-3-8B-Instruct-mntp-supervised",
# "LLM2Vec-Mistral-7B-Instruct-v2-mntp-supervised",
# "bigscience/sgpt-bloom-7b1-msmarco",
# "izhx/udever-bloom-7b1",
'Alibaba-NLP/gte-Qwen1.5-7B-instruct',
'intfloat/e5-mistral-7b-instruct',
'Linq-AI-Research/Linq-Embed-Mistral',
'nvidia/NV-Embed-v1'
]





# In[14]:


df_summary = pd.read_excel ("summaries/summaries.xlsx")

df_summary.rename (inplace = True, 
                   columns = {"text":"source", "summary":"destination"})

df_imaging = pd.read_csv ("reports/reports.csv")

df_imaging.rename (inplace = True, columns = {
    "FINDINGS":"source", "IMPRESSION":"destination"})

df_HP = pd.read_csv ("df_HP.csv")

df_HP.rename (inplace = True, columns = {
    "ED":"source", "Clean_Text":"destination"})

df_complaint = pd.read_csv ("df_complaint.csv")

df_complaint.rename (inplace = True, columns = {"Clean_Text":"source"})

df_complaint ["destination"] = df_complaint ["ChiefComplaintName"].apply (
    lambda x: f"{x}".replace ("[", "").replace ("]", ""))


# In[16]:


def get_batch_size(model_name):
    # Model size to batch size mapping
    size_to_batch_size = {
        # Models with size in range 100
        'avsolatorio/GIST-small-Embedding-v0': 1024,
        # 'infgrad/stella-base-en-v2': 1024,
        'BAAI/bge-small-en-v1.5': 1024,
        'thenlper/gte-small': 1024,
        'sentence-transformers/all-MiniLM-L12-v2': 1024,
        'sentence-transformers/all-MiniLM-L6-v2': 1024,

        # Models with size in range 100-250
        'avsolatorio/GIST-Embedding-v0': 512,
        'BAAI/bge-base-en-v1.5': 512,
        'thenlper/gte-base': 512,
        'dwzhu/e5-base-4k': 512,
         # 'jinaai/jina-embeddings-v2-base-en': 512,
        'emilyalsentzer/Bio_ClinicalBERT': 512,
        
        # Models with size in range 250-500
        'mixedbread-ai/mxbai-embed-large-v1': 128,
        'WhereIsAI/UAE-Large-V1': 128,
        'avsolatorio/GIST-large-Embedding-v0': 128,
        'ekorman-strive/bge-large-en-v1.5': 128,
        'w601sxs/b1ade-embed': 128,

        # Models with size in range 500-1000
        'intfloat/multilingual-e5-large-instruct': 8,
        'intfloat/multilingual-e5-large': 8,
        'OrdalieTech/Solon-embeddings-large-0.1': 8,
        'manu/bge-m3-custom-fr': 8,
        'sdadas/mmlw-e5-large': 8,

        # Models with size in range 1000-5000
        'hkunlp/instructor-xl': 2,
        'izhx/udever-bloom-3b': 2,
        'sentence-transformers/sentence-t5-xxl': 2,
        'sentence-transformers/gtr-t5-xxl': 2,
        'Muennighoff/SGPT-2.7B-weightedmean-msmarco-specb-bitfit': 2,

        # Models with size over 5000
        'Salesforce/SFR-Embedding-Mistral': 1,
        'Alibaba-NLP/gte-Qwen1.5-7B-instruct': 1,
        'intfloat/e5-mistral-7b-instruct': 1,
        'Linq-AI-Research/Linq-Embed-Mistral': 1,
        'nvidia/NV-Embed-v1': 1
    }

    # Return the batch size for the given model, default to 1 if model not found
    return size_to_batch_size.get (model_name, 1) * 4



# Define the function to check file existence and load or create data accordingly
def process_data(task, df, model):
    filename = "df_temp/" + model.replace ("/", "_") + "_" + task + ".pkl"
    if os.path.exists(filename):
        print(f"Loading existing data for {model} on task {task}.")
        df_temp = pd.read_pickle(filename)
    else:
        print(f"Running new experiment for {model} on task {task}.")
        batch_size = get_batch_size(model)
        df_temp = run_experiment(df[0:2000], model_name=model, task=task, batch_size=batch_size)
        df_temp["model"] = model
        df_temp["task"] = task
        df_temp.to_pickle(filename)
    return df_temp

df_list = []

def clear_memory ():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()  # Clears the cache
        torch.cuda.ipc_collect()  # Clear any IPC handles if any are held
        torch.cuda.synchronize()  # Wait for all kernels to finish
    gc.collect()  # Explicitly runs garbage collection


# Iterate through tasks and dataframes
for task, df in zip(["summary", "imaging", "HP", "complaint"], 
                    [df_summary, df_imaging, df_HP, df_complaint]):
    for model in models:
        df_temp = process_data(task, df, model)
        df_list.append(df_temp)
        clear_memory ()  # Clear cache after processing each model
        
# df_final = pd.concat (df_list)

