"""Label-free project features using the supplied train and test feature pool."""
import sys
from pathlib import Path
import pandas as pd
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'experiments'),str(ROOT/'experiments/similarity_study')]
from similarity import features

def make(train,test):
    """Build numeric pooled features; rows and labels are never joined externally."""
    rows=train.astype(object).where(pd.notna(train),'').to_dict('records')
    future=test.astype(object).where(pd.notna(test),'').to_dict('records')
    base,extra=features(rows+future)
    result=pd.concat([base,extra['secondary']],axis=1)
    if result.columns.duplicated().any():raise ValueError('Duplicate feature names')
    return result.iloc[:len(train)].reset_index(drop=True),result.iloc[len(train):].reset_index(drop=True)
