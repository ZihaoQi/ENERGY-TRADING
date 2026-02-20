from features.xgb_features import *
from features.data_loader import *
from models.architectures.xgboost import XGBoostModel
from models.training import ModelTrainer
from models.evaluation import ModelEvaluator
from strategies.ml_strategies import MLPredictionStrategy
from backtest.backtester import *
from config import *


def main():
    print("="*60)
    print("XGBoost Classifier Trading Model")
    print("="*60)

    print(f"\nTRADING ASSET: {trading_asset.upper()}\n")
    
    # -------------------------------------------------------------------------
    # 1. Load data
    # -------------------------------------------------------------------------
    print(f"\n1. Loading {trading_asset} data...")

    df = load_data()

    print(f"   Data shape: {df.shape}")
    print(f"   Date range: {df['Date'].min()} to {df['Date'].max()}")
    
    # -------------------------------------------------------------------------
    # 2. Create features
    # -------------------------------------------------------------------------
    print("\n2. Creating features...")
    df = create_features(df, target_col=target_col,
                     lookback_period=lookback_period,
                     return_threshold=training_return_threshold)
    
    print(f"   Features created. Shape after feature engineering: {df.shape}")
    
    # -------------------------------------------------------------------------
    # 3. Prepare train/test split
    # -------------------------------------------------------------------------
    print("\n3. Preparing train/test split...")

    X = df[feature_cols]
    y = df['target']
    
    split_idx = int(len(df) * training_split)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    print(f"   Training set: {len(X_train)} samples")
    print(f"   Test set: {len(X_test)} samples")
    
    # -------------------------------------------------------------------------
    # 4. Train model
    # -------------------------------------------------------------------------
    print("\n4. Training model...")

    model = XGBoostModel(config=xgboost_model_config)
    # model.optimize(X_train, y_train, n_trials=30)

    trainer = ModelTrainer(model)
    trainer.train(X_train, y_train)
    y_pred_test, y_proba_test = trainer.evaluate(X_train, y_train, X_test, y_test)
    scores = trainer.cross_validate(X_train, y_train)
    
    # -------------------------------------------------------------------------
    # 5. Visualize model performance
    # -------------------------------------------------------------------------
    print("\n5. Creating model performance visualizations...")

    evaluator = ModelEvaluator(model)

    evaluator.print_classification_report(
        y_test,
        y_pred_test,
        target_names=target_classification
    )

    # Create 2x2 dashboard
    fig = evaluator.evaluate(
        X_test=X_test,
        y_test=y_test,
        y_train=y_train,
        y_pred_test=y_pred_test,
        feature_names=feature_cols,
        target_names=target_classification,
        trading_threshold=trading_threshold,
        save_path=model_performance_path
    )

    # plt.show()
    
    # -------------------------------------------------------------------------
    # 6. Create strategy
    # -------------------------------------------------------------------------
    print("\n6. Creating strategy...")
    strategy = MLPredictionStrategy(model, feature_cols=feature_cols, target_col=target_col,
                                holding_period=holding_period, threshold=trading_threshold,
                                take_profit_pct=take_profit_pct, stop_loss_pct=stop_loss_pct)
    
    # -------------------------------------------------------------------------
    # 7. Run backtest
    # -------------------------------------------------------------------------
    print("\n7. Running backtest...")

    df_backtest = df.iloc[split_idx:].reset_index(drop=True) # Use only test data for backtesting to avoid look-ahead bias
    print(f"Backtest date range: {df['Date'].min()} to {df['Date'].max()}")
    
    backtester = Backtester(df_backtest, strategy=strategy, price_col=price_col,
                               max_positions=max_positions, commission=comm, slippage=slippage)
    
    trades = backtester.run_backtest()
    
    # -------------------------------------------------------------------------
    # 8. Calculate and analyze strategy performance
    # -------------------------------------------------------------------------
    print("\n8. Calculate strategy performance...")

    analyzer = PerformanceAnalyzer(backtester.get_trades_df())
    results = analyzer.calculate_metrics()

    # -------------------------------------------------------------------------
    # 9. Visualize backtest results
    # -------------------------------------------------------------------------

    if results:
        print("\n9. Creating backtest visualizations...")
        visualize_backtest_results(results, save_path=backtest_results_path)
        
        # Save trades to CSV
        results['trades_df'].to_csv(trades_log_path, index=False)
        print("\n✓ Trades log saved to CSV")

    print("\n" + "="*60)
    print("EXECUTION COMPLETE")
    print("="*60)
    print("\nOutput files generated:")
    print("  - model_performance.png")
    print("  - backtest_results.png")
    print("  - trades_log.csv")
    print("\n")

if __name__ == "__main__":
    main()
