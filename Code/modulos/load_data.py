import pandas as pd 
def ldata1():
    df = pd.read_csv("datasets/TheHackerNews_Dataset.csv")
    return df
def ldata2():
    df = pd.read_csv("datasets/CISA.csv")
    return df
def ldata3():
    df = pd.read_csv("datasets/CyberBERT.csv")
    return df 
def ldata4():
    splits = {'train': 'data/train-00000-of-00001.parquet', 'validation': 'data/validation-00000-of-00001.parquet', 'test': 'data/test-00000-of-00001.parquet'}
    df = pd.read_parquet("hf://datasets/OhWayTee/Cybersecurity-News/" + splits["train"])
    return df


