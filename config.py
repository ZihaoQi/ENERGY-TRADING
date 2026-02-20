from pathlib import Path

# ttf or eua
trading_asset = 'ttf'

lookback_period = 5  # in days
holding_period = lookback_period

training_return_threshold = 0.01
trading_threshold = 0.5
max_positions = 2
take_profit_pct = 10
stop_loss_pct = 8

training_split = 0.75

# xgboost target classes
target_classification = ['Negative', 'Neutral', 'Positive']

positive_target_class = 2 # positive
neutral_target_class = 1 # neutral
negative_target_class = 0 # negative

# output paths
out_dir = Path.cwd() / "outputs"
out_dir.mkdir(parents=True, exist_ok=True)

model_performance_path = out_dir / 'model_performance.png'
backtest_results_path = out_dir / 'backtest_results.png'
trades_log_path = out_dir / 'trades_log.csv'

test_out_dir = Path.cwd() / "outputs_test"
test_out_dir.mkdir(parents=True, exist_ok=True)

test_model_performance_path = test_out_dir / 'model_performance.png'
test_backtest_results_path = test_out_dir / 'backtest_results.png'
test_trades_log_path = test_out_dir / 'trades_log.csv'

# column names in the data df
if trading_asset == 'ttf':
    target_col = 'ttf'
    price_col='ttf'
    feature_cols = ['jkm', 'storage_full', 'ttf_lag1', 'ttf_lag2']
    comm = 1
    slippage = 0.005
elif trading_asset == 'eua':
    target_col = 'eua'
    price_col='eua'
    feature_cols = ['ttf', 'coal', 'power', 'stoxx', 'eua_lag1', 'eua_lag2']
    comm = 1
    slippage = 0.01



lstm_model_config = {
    'seq_len': 14,
    'train_split': 0.8,
    'batch_size': 32,
    'hidden_size': 64,
    'num_layers': 2,
    'epochs': 50,
    'random_seed': 42,
    'optimizer_params': {'learning_rate': 0.001}
}

xgboost_model_config = {
    'n_estimators': 200,
    'max_depth': 5,
    'learning_rate': 0.1,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'random_state': 42,
    'eval_metric': 'mlogloss'
}