import pandas as pd
from config import trading_asset

eua_trading_data_path = 'data/eua_trading'
ttf_trading_data_path = 'data/ttf_trading'

def load_csv_from_investingcom(path, col):
    df = pd.read_csv(path, parse_dates=['Date'])
    df.rename(columns={'Price': col}, inplace=True)
    df = df.set_index('Date').sort_index()
    return df[[col]]

def load_eua_trading_data():
    eua = load_csv_from_investingcom(eua_trading_data_path + '/eua.csv', 'eua')
    ttf = load_csv_from_investingcom(eua_trading_data_path + '/ttf.csv', 'ttf')
    stoxx = load_csv_from_investingcom(eua_trading_data_path + '/stoxx.csv', 'stoxx')
    power = load_csv_from_investingcom(eua_trading_data_path + '/power_de.csv', 'power')
    coal = load_csv_from_investingcom(eua_trading_data_path + '/coal.csv', 'coal')

    df = eua.join([ttf, stoxx, power, coal], how='outer')
    df = df.ffill().bfill()
    return df.reset_index()

def load_ttf_trading_data():
    ttf = load_csv_from_investingcom(ttf_trading_data_path + '/ttf.csv', 'ttf')
    jkm = load_csv_from_investingcom(ttf_trading_data_path + '/jkm.csv', 'jkm')

    storage = pd.read_csv(ttf_trading_data_path + '/storage.csv', sep=';', parse_dates=['Gas Day End'])
    storage.rename(columns={'Gas Day End': 'Date', 'Full (%)': 'storage_full'}, inplace=True)
    storage = storage.set_index('Date').sort_index()
    storage = storage[['storage_full']]

    df = ttf.join([jkm, storage], how='outer')
    df = df.ffill().bfill()
    return df.reset_index()

def load_data():
    if trading_asset == 'ttf':
        return load_ttf_trading_data()
    elif trading_asset == 'eua':
        return load_eua_trading_data()
    else:
        raise ValueError("trading_asset must be 'ttf' or 'eua'")