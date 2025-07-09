import pandas as pd

def load_data(file):
    if hasattr(file, 'name'):
        filename = file.name
    else:
        filename = file

    if filename.endswith(".csv"):
        df = pd.read_csv(file, parse_dates=['InvoiceDate'], dayfirst=True, encoding='ISO-8859-1')
    elif filename.endswith(".xlsx"):
        df = pd.read_excel(file, parse_dates=['InvoiceDate'])
    else:
        raise ValueError("Unsupported file format: use .csv or .xlsx")

    df.rename(columns={
        'Invoice': 'InvoiceNo',
        'Price': 'UnitPrice',
        'Customer ID': 'CustomerID'
    }, inplace=True)

    df.dropna(inplace=True)
    df = df[df['Quantity'] > 0]

    return df
