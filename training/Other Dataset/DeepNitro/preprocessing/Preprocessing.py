import pandas as pd

positive_dataset = pd.read_excel('training dataset.xlsx', sheet_name='Y-Nitration positive data', header=2 )
negative_dataset = pd.read_excel('training dataset.xlsx', sheet_name='Y-Nitration negative data', header=2)

pos_seq = pd.concat([positive_dataset['Peptide'], positive_dataset['Peptide.1'], 
                      positive_dataset['Peptide.2']], axis=0)
neg_seq = pd.concat([negative_dataset['Peptide'], negative_dataset['Peptide.1'], 
                      negative_dataset['Peptide.2']], axis=0)
pos_seq.dropna(inplace=True)
neg_seq.dropna(inplace=True)
pos_seq.str.replace('*', 'X')
neg_seq.str.replace('*', 'X')
pos_seq.to_csv('pos.txt',header=None, index=None)
neg_seq.to_csv('neg.txt',header=None, index=None)