# -*- coding: utf-8 -*-
"""
Created on Sun Jul 11 19:45:15 2021

@author: Sabit
"""

import pandas as pd
import csv


dataset = pd.read_csv("independent-dataset-PredNTS.csv")
dataset[['Sequence']] = dataset['Sequence'].str.replace('-', '*')
dataset["fasta_sequence"] = ">" + dataset["protein_id"] + " " +"\n" + dataset["Sequence"]
fasta_dataset = dataset[["fasta_sequence"]]
fasta_dataset.to_csv("IndTestSeq.fasta", header=None, index=None, quoting=csv.QUOTE_NONE, escapechar=" ")