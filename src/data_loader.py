import pandas as pd
from src.config import DATA_PATH

def load_and_prepare_data():
    print(f"Loading data from {DATA_PATH}...")
    df = pd.read_excel(DATA_PATH)
    df = df.dropna(subset=['CustomerID'])
    df = df[~df['InvoiceNo'].astype(str).str.startswith('C')]
    df['CustomerID'] = df['CustomerID'].astype(int)
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    df['TotalPrice'] = df['UnitPrice'] * df['Quantity']
    
    snapshot_date = df['InvoiceDate'].max() + pd.Timedelta(days=1)
    rfm = df.groupby('CustomerID').agg(
        recency=('InvoiceDate', lambda x: (snapshot_date - x.max()).days),
        frequency=('InvoiceNo', 'nunique'),
        monetary=('TotalPrice', 'sum')
    ).reset_index()
    return rfm
