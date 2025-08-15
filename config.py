import os
from datetime import datetime

class Config:
    DATA_PATH = "data/"
    CURRENCY_PAIRS = ["EURUSD", "GBPUSD", "USDJPY"]
    # DateTime parsing settings
    DATETIME_FORMAT = "%d.%m.%Y %H:%M:%S.%f"  # DD.MM.YYYY format
    HAS_TIMEZONE = True  # Whether data has GMT timezone info
    SEQUENCE_LENGTH = 60  # 60 hours lookback
    PREDICTION_HORIZON = 1  # Predict next 1 hour    
    # Technical indicators
    RSI_PERIOD = 14
    MACD_FAST = 12
    MACD_SLOW = 26
    MACD_SIGNAL = 9
    # ===== MODEL ARCHITECTURE =====
    # CNN parameters
    CNN_FILTERS_1 = 32
    CNN_FILTERS_2 = 64
    CNN_KERNEL_SIZE = 3
    # LSTM parameters
    LSTM_UNITS = 64
    # Dense layers
    DENSE_UNITS = 64
    # Dropout for MC uncertainty
    DROPOUT_RATE = 0.3
    MC_SAMPLES = 50  # Number of MC samples for confidence intervals
    
    # ===== TRAINING SETTINGS =====
    BATCH_SIZE = 32
    INITIAL_EPOCHS = 100
    LEARNING_RATE = 0.00001  # ลด learning rate
    
    # Early stopping
    PATIENCE = 10  # เพิ่ม patience
    MIN_DELTA = 0.0001
    
    # Train/Validation/Test split (ตาม thesis methodology)
    TRAIN_RATIO = 0.60  # 60% for training (ประมาณ 3 ปี)
    VAL_RATIO = 0.20    # 20% for validation (ประมาณ 1 ปี)
    TEST_RATIO = 0.20   # 20% for testing (ประมาณ 1 ปี)
    
    # Time-based split settings (สำหรับ 5 ปี: 2018-2022)
    DATA_START_YEAR = 2018
    DATA_END_YEAR = 2022
    
    # Rolling window backtesting (optional - ใช้ถ้าต้องการ rolling window)
    USE_ROLLING_WINDOW = False  # Set True ถ้าต้องการใช้ rolling window
    ROLLING_WINDOW_SIZE = 0.8  # 80% for train+val, 20% for test
    STEP_SIZE = 0.1  # Move 10% each step
    
    # ===== EXPERIMENT TRACKING =====
    EXPERIMENT_TRACKING = "tensorboard"  # Options: "tensorboard", "mlflow", "both"
    
    # Experiment naming
    EXPERIMENT_BASE_DIR = "experiments/"
    
    @staticmethod
    def get_experiment_name():
        """Generate experiment name with timestamp"""
        now = datetime.now()
        return f"exp_{now.strftime('%d_%m_%Y_%H_%M')}"
    
    @staticmethod
    def get_experiment_path():
        """Get full experiment path"""
        exp_name = Config.get_experiment_name()
        exp_path = os.path.join(Config.EXPERIMENT_BASE_DIR, exp_name)
        return exp_path
    
    # ===== VISUALIZATION SETTINGS =====
    FIGURE_SIZE = (12, 8)
    DPI = 300
    STYLE = "seaborn-v0_8"
    
    # Confidence interval settings
    CONFIDENCE_LEVELS = [0.05, 0.95]  # 5% and 95% quantiles
    
    # ===== FINANCIAL METRICS =====
    # Transaction costs (spread simulation)
    TRANSACTION_COST = 0.0001  # 1 pip for major pairs
    
    # Risk-free rate for Sharpe ratio
    RISK_FREE_RATE = 0.02  # 2% annual
    
    # Position sizing
    INITIAL_CAPITAL = 100000
    POSITION_SIZE = 0.1  # 10% of capital per trade
    
    # ===== PREDICTION THRESHOLDS =====
    BUY_THRESHOLD = 0.5   # Signal > 0.6 = BUY
    SELL_THRESHOLD = 0.5  # Signal < 0.4 = SELL
    
    # ===== SYSTEM SETTINGS =====
    RANDOM_SEED = 42
    N_JOBS = -1  # Use all available cores
    VERBOSE = 1
    
    # GPU settings
    USE_GPU = True
    GPU_MEMORY_GROWTH = True
    
    # ===== FEATURE ENGINEERING =====
    # Target variable creation
    TARGET_THRESHOLD = 0.0001  # ลดลงเป็น 0.01% เพื่อได้สัญญาณมากขึ้น
    
    # Additional features (can be extended)
    USE_VOLUME_FEATURES = True
    USE_CROSS_PAIR_FEATURES = True
    
    # ===== VALIDATION =====
    @classmethod
    def validate_config(cls):
        """Validate configuration parameters"""
        assert cls.TRAIN_RATIO + cls.VAL_RATIO + cls.TEST_RATIO == 1.0, \
            "Train, validation, and test ratios must sum to 1.0"
        
        assert cls.SEQUENCE_LENGTH > 0, "Sequence length must be positive"
        assert cls.BATCH_SIZE > 0, "Batch size must be positive"
        assert 0 < cls.LEARNING_RATE < 1, "Learning rate must be between 0 and 1"
        assert cls.BUY_THRESHOLD > cls.SELL_THRESHOLD, \
            "Buy threshold must be greater than sell threshold"
        
        # Ensure data directory exists
        if not os.path.exists(cls.DATA_PATH):
            os.makedirs(cls.DATA_PATH)
        
        # Ensure experiment directory exists
        if not os.path.exists(cls.EXPERIMENT_BASE_DIR):
            os.makedirs(cls.EXPERIMENT_BASE_DIR)
        
        print("✅ Configuration validation passed!")
        return True

# Global config instance
config = Config()

# Validate on import
if __name__ == "__main__":
    config.validate_config()
    print("📁 Data path:", config.DATA_PATH)
    print("📊 Currency pairs:", config.CURRENCY_PAIRS)
    print("🎯 Sequence length:", config.SEQUENCE_LENGTH)
    print("🧠 Model: CNN({},{}) + LSTM({}) + Cross-Attention".format(
        config.CNN_FILTERS_1, config.CNN_FILTERS_2, config.LSTM_UNITS))
    print("📈 Experiment tracking:", config.EXPERIMENT_TRACKING)
    print("💾 Next experiment path:", config.get_experiment_path())