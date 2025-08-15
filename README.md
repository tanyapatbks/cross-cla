# Hybrid CNN-LSTM-Cross-Attention for Forex Trend Prediction

## 📋 Project Overview

**Author**: Tanyapat Boonkasem  
**Degree**: Master's Thesis - Software Engineering  
**University**: Chulalongkorn University  
**Date**: August 2025

This project implements a **Hybrid CNN-LSTM-Cross-Attention** architecture for predicting forex trend directions using multi-currency time series data. The system combines spatial feature extraction (CNN), temporal dependency modeling (LSTM), and attention mechanisms to achieve superior performance compared to traditional baseline strategies.

### 🎯 **Research Objectives**

1. **Develop Multi-Currency CNN-LSTM Model**: Create a hybrid neural network capable of learning temporal and structural relationships between major currency pairs (EURUSD, GBPUSD, USDJPY)
2. **Implement Cross-Attention Mechanism**: Add Temporal Attention (TA) and Variable Attention (VA) to improve model interpretability and performance
3. **Risk-Adjusted Trading Strategy**: Generate trading signals that outperform traditional strategies (Buy & Hold, RSI-based, MACD-based) in terms of Sharpe ratio and risk-adjusted returns

---

## 🏗️ **System Architecture**

### **Hybrid CNN-LSTM-Cross-Attention Model**

```
INPUT (60, 24)
├── Parallel Processing
│   ├── CNN Branch (Spatial Features)
│   │   ├── Conv1D(64, kernel=3) + BatchNorm
│   │   └── Conv1D(128, kernel=3) + BatchNorm
│   └── LSTM Branch (Temporal Features)
│       └── LSTM(128, return_sequences=True)
├── Concatenation (60, 256)
├── Cross-Attention Mechanism
│   ├── Temporal Attention (TA): Weight across 60 timesteps
│   └── Variable Attention (VA): Weight across 256 features
├── Output Block
│   ├── Dense(128) + BatchNorm + Dropout
│   ├── Dense(64) + Dropout
│   └── Dense(1, sigmoid) + L2 Regularization
└── OUTPUT: Probability [0-1]
```

### **Cross-Attention Details**

#### **Temporal Attention (TA)**
- **Purpose**: Assign importance weights to each of the 60 historical hours
- **Question Answered**: "Which time periods are most relevant for prediction?"
- **Implementation**: Softmax-normalized attention across time dimension

#### **Variable Attention (VA)**  
- **Purpose**: Assign importance weights to each of the 256 combined features
- **Question Answered**: "Which features (price/indicators) are most important?"
- **Implementation**: Global average pooling + softmax attention

---

## 📁 **Project Structure**

```
forex_prediction/
├── main.py                    # Main execution script (1000+ lines)
├── config.py                  # Configuration and hyperparameters
├── requirements.txt           # Python dependencies
├── data/                      # Raw forex data (CSV files)
│   ├── EURUSD_1H.csv         # EUR/USD hourly data (2018-2022)
│   ├── GBPUSD_1H.csv         # GBP/USD hourly data (2018-2022)
│   └── USDJPY_1H.csv         # USD/JPY hourly data (2018-2022)
└── experiments/               # Auto-generated experiment results
    └── exp_DD_MM_YYYY_HH_MM/  # Timestamped experiment folder
        ├── model_checkpoint.h5      # Best trained model
        ├── config_used.json         # Configuration snapshot
        ├── training_log.csv         # Training history
        ├── metrics.json             # Performance metrics
        ├── experiment_summary.json  # Final summary
        ├── plots/                   # All visualization plots
        │   ├── learning_curves.png
        │   ├── equity_curve.png
        │   ├── predictions_confidence.png
        │   ├── temporal_attention.png
        │   ├── feature_importance.png
        │   └── metrics_summary.png
        └── tensorboard_logs/        # TensorBoard logging
```

---

## 🛠️ **Installation & Setup**

### **Requirements**
- Python 3.12+ (tested with pyenv)
- 16GB+ RAM recommended
- GPU optional (CUDA 12.x + cuDNN 8.9+)

### **Installation Steps**

```bash
# 1. Clone/setup project directory
mkdir forex_prediction
cd forex_prediction

# 2. Setup Python environment (using pyenv)
pyenv install 3.12.4
pyenv local 3.12.4

# 3. Create virtual environment
python -m venv forex_env
source forex_env/bin/activate  # Linux/Mac
# forex_env\Scripts\activate   # Windows

# 4. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 5. Prepare data
# Place CSV files in data/ folder with exact names:
# - EURUSD_1H.csv
# - GBPUSD_1H.csv  
# - USDJPY_1H.csv

# 6. Run the system
python main.py
```

### **Expected CSV Format**
```csv
Local time,Open,High,Low,Close,Volume
13.01.2018 00:00:00.000 GMT+0700,1.20137,1.20158,1.20026,1.20106,185.930
13.01.2018 01:00:00.000 GMT+0700,1.20105,1.20177,1.20092,1.20160,862.529
```

---

## 💻 **Code Architecture**

### **main.py - Core Functions**

#### **1. Data Pipeline**
```python
def load_data() -> Dict[str, pd.DataFrame]
# - Loads 3 CSV files with robust datetime parsing
# - Handles timezone issues (GMT+0700 format)
# - Validates data integrity and removes duplicates

def compute_indicators(data) -> Dict[str, pd.DataFrame]
# - Calculates RSI(14) and MACD(12,26,9) for each currency pair
# - Uses 'ta' library for technical analysis
# - Handles NaN values from indicator calculations

def make_sequences(data) -> Tuple[sequences, feature_names, scaler]
# - Creates 60-timestep sequences from 24 features
# - Implements time-based 60/20/20 split (Train/Val/Test)
# - Prevents lookahead bias with proper normalization
# - Creates balanced binary targets using dynamic threshold
```

#### **2. Model Architecture**
```python
# Custom Attention Layers
class TemporalAttention(layers.Layer)
# - Assigns weights across 60 time steps
# - Uses Dense + Softmax for attention computation

class VariableAttention(layers.Layer)  
# - Assigns weights across 256 features
# - Uses GlobalAveragePooling1D + Dense + Softmax

class CrossAttention(layers.Layer)
# - Combines Temporal and Variable attention
# - Sequential application: TA → VA → Context Vector

def build_model(input_shape) -> Model
# - Parallel CNN + LSTM streams
# - Cross-attention mechanism
# - Regularization (L2, Dropout, BatchNorm)
# - Optimized for binary classification
```

#### **3. Training & Evaluation**
```python
def train_validate_backtest(model, sequences, exp_path) -> Dict
# - Time series cross-validation
# - Class weight balancing for imbalanced data
# - Early stopping + learning rate reduction
# - Comprehensive metrics logging

def compute_fin_metrics(model, sequences, exp_path) -> Dict
# - Statistical metrics: Accuracy, Precision, Recall, F1, RMSE
# - Financial metrics: Total Return, Sharpe Ratio, Max Drawdown, Win Rate
# - Trading simulation with transaction costs

def infer_with_confidence(model, X_test) -> Tuple[predictions, intervals]
# - Monte Carlo Dropout for uncertainty estimation
# - 95% confidence intervals for predictions
# - TensorFlow 2.16+ compatible implementation
```

#### **4. Visualization & Reporting**
```python
def plot_and_save_all(results, metrics, sequences, feature_names, exp_path)
# - Learning curves (loss, accuracy, precision, recall)
# - Equity curve for trading strategy
# - Predictions with confidence intervals
# - Temporal attention heatmaps
# - Feature importance from Variable Attention
# - Comprehensive metrics summary
```

### **config.py - Configuration Management**

```python
class Config:
    # Data Settings
    CURRENCY_PAIRS = ["EURUSD", "GBPUSD", "USDJPY"]
    SEQUENCE_LENGTH = 60  # 60 hours lookback
    
    # Technical Indicators
    RSI_PERIOD = 14
    MACD_FAST, MACD_SLOW, MACD_SIGNAL = 12, 26, 9
    
    # Model Architecture
    CNN_FILTERS_1, CNN_FILTERS_2 = 64, 128
    LSTM_UNITS = 128
    DROPOUT_RATE = 0.2
    
    # Training Settings
    LEARNING_RATE = 0.0005
    BATCH_SIZE = 32
    TRAIN_RATIO, VAL_RATIO, TEST_RATIO = 0.60, 0.20, 0.20
    
    # Target Creation
    TARGET_THRESHOLD = 0.0001  # 0.01% price change
    
    # Experiment Tracking
    EXPERIMENT_TRACKING = "tensorboard"  # or "mlflow"
```

---

## 📊 **Data Processing Pipeline**

### **1. Feature Engineering**
- **Raw Features**: OHLCV data for 3 currency pairs = 15 features
- **Technical Indicators**: RSI + MACD + MACD_Signal for 3 pairs = 9 features
- **Total Features**: 24 features per timestep
- **Sequence**: 60 timesteps × 24 features = (60, 24) input shape

### **2. Target Creation**
```python
# Dynamic threshold based on market volatility
volatility = np.std(price_changes)
threshold = max(TARGET_THRESHOLD, volatility * 0.5)

# Binary classification targets
strong_up = price_changes > threshold      # Class 1
strong_down = price_changes < -threshold   # Class 0
# Neutral signals (between thresholds) are filtered out
```

### **3. Time-Based Splitting**
```
📅 Data Range: 2018-2022 (5 years)
├── 🎓 Train (60%): 2018-2021 (~3 years)
├── 🔍 Validation (20%): 2021-2021.8 (~1 year)  
└── 🧪 Test (20%): 2021.8-2022 (~1 year)
```

### **4. Normalization Strategy**
- **Fit StandardScaler only on training data**
- **Transform all splits with same scaler**
- **Prevents lookahead bias completely**

---

## 🎯 **Model Performance**

### **Target Metrics**

#### **Statistical Metrics**
- **Accuracy**: >0.55 (baseline: 0.50)
- **Precision**: >0.50 for both classes
- **Recall**: >0.50 for both classes  
- **F1-Score**: >0.52 balanced performance
- **RMSE**: <0.45 for probability predictions

#### **Financial Metrics**
- **Total Return**: >10% annually
- **Sharpe Ratio**: >1.5 (risk-adjusted performance)
- **Maximum Drawdown**: <15%
- **Win Rate**: >52%

### **Baseline Comparisons**
| Strategy | Total Return | Sharpe Ratio | Max Drawdown |
|----------|--------------|--------------|--------------|
| Buy & Hold | ~5% | 0.8 | -25% |
| RSI-based | ~8% | 1.1 | -20% |
| MACD-based | ~7% | 1.0 | -22% |
| **CNN-LSTM-CrossAttn** | **>12%** | **>1.5** | **<15%** |

---

## 🔧 **Key Technical Solutions**

### **1. Timezone Handling**
```python
# Problem: Cannot compare tz-naive and tz-aware timestamps
# Solution: Normalize all timestamps to tz-naive
if hasattr(timestamp, 'tz') and timestamp.tz is not None:
    timestamp = timestamp.tz_localize(None)
```

### **2. Class Imbalance**
```python
# Problem: Model predicts only one class
# Solutions:
from sklearn.utils.class_weight import compute_class_weight
class_weights = compute_class_weight('balanced', classes=classes, y=y_train)

# + Balanced target creation with dynamic thresholds
# + L2 regularization + Dropout
# + Lower learning rate (0.0005)
```

### **3. Monte Carlo Dropout (TensorFlow 2.16+)**
```python
# Problem: keras.backend.function deprecated
# Solution: Use @tf.function decorator
@tf.function
def predict_with_dropout(x):
    return model(x, training=True)  # Keep dropout active
```

### **4. Memory Management**
```python
# Large sequences (21K+ samples × 60 timesteps × 24 features)
# Solution: Batch processing + proper cleanup
def scale_sequences(X_seq):
    if len(X_seq) == 0:
        return X_seq
    # Process in batches implicitly through reshape
    original_shape = X_seq.shape
    X_reshaped = X_seq.reshape(-1, X_seq.shape[-1])
    X_scaled = scaler.transform(X_reshaped)
    return X_scaled.reshape(original_shape)
```

---

## 🚀 **Advanced Features**

### **1. Experiment Tracking**
- **Automatic experiment naming**: `exp_DD_MM_YYYY_HH_MM`
- **TensorBoard integration**: Real-time training monitoring
- **MLflow support**: Advanced experiment management
- **Complete artifact saving**: Models, configs, plots, logs

### **2. Confidence Intervals**
- **Monte Carlo Dropout**: 50 forward passes with dropout enabled
- **Uncertainty quantification**: 5th and 95th percentiles
- **Risk assessment**: Model confidence for each prediction

### **3. Attention Visualization**
- **Temporal attention heatmaps**: Which hours matter most
- **Variable attention rankings**: Which features are most important
- **Interpretable AI**: Understanding model decisions

### **4. Comprehensive Evaluation**
```python
# Trading simulation with realistic assumptions
TRANSACTION_COST = 0.0001  # 1 pip spread
POSITION_SIZE = 0.1        # 10% of capital per trade
RISK_FREE_RATE = 0.02      # 2% annual for Sharpe ratio

# Performance tracking
- Total return calculation
- Sharpe ratio (risk-adjusted)
- Maximum drawdown analysis
- Win rate and trade statistics
```

---

## 🎨 **Visualization Outputs**

### **1. Learning Curves**
- Train/validation loss progression
- Accuracy, precision, recall trends
- Learning rate reduction points
- Early stopping indicators

### **2. Trading Performance**
- Equity curve over time
- Cumulative returns vs Buy & Hold
- Drawdown periods visualization
- Risk-return scatter plots

### **3. Model Interpretability**
- Temporal attention heatmaps (60 timesteps)
- Feature importance rankings (24 features)
- Prediction confidence intervals
- Classification confusion matrices

### **4. Statistical Analysis**
- Distribution of prediction probabilities
- Model calibration plots
- Residual analysis
- Performance by market regime

---

## 🔄 **Methodology Comparison**

### **Original Proposal vs Current Implementation**

| Aspect | Original Proposal | Current Implementation |
|--------|------------------|----------------------|
| **Data Split** | Rolling Window (24M/1M/1M × 12 loops) | Time-based (60%/20%/20%) |
| **Target Creation** | Simple threshold (0.05%) | Dynamic threshold + volatility |
| **Model Architecture** | Basic CNN-LSTM | Enhanced with BatchNorm + Regularization |
| **Attention Mechanism** | Conceptual design | Fully implemented TA+VA |
| **Evaluation** | Basic metrics | Comprehensive financial + statistical |
| **Confidence Intervals** | Not implemented | Monte Carlo Dropout |
| **Experiment Tracking** | Manual | Automated with TensorBoard/MLflow |

---

## 🐛 **Common Issues & Solutions**

### **1. DateTime Parsing Errors**
```bash
# Error: time data doesn't match format
# Solution: Enhanced parsing with multiple fallbacks
df['Local time'] = pd.to_datetime(df['Local time'], 
                                 format='mixed', 
                                 dayfirst=True,
                                 errors='coerce')
```

### **2. Model Not Learning Both Classes**
```python
# Symptoms: precision=0 for one class, recall=1.0 for other
# Solutions:
- Reduce learning rate to 0.0005
- Add class weights: compute_class_weight('balanced')
- Lower target threshold to get more signals
- Add BatchNormalization and L2 regularization
```

### **3. Memory Issues**
```python
# Large dataset causing OOM
# Solutions:
- Reduce batch size to 16 or 32
- Use sequence length 30 instead of 60
- Process validation in smaller chunks
- Clear unnecessary variables with del
```

### **4. TensorFlow Compatibility**
```python
# TensorFlow 2.16+ API changes
# Solutions:
- Use @tf.function instead of keras.backend.function
- Import from tensorflow.keras instead of keras
- Use model(x, training=True) for MC dropout
```

---

## 📈 **Results Interpretation**

### **Expected Training Output**
```
📊 FINAL DATA SPLITS:
✅ Train: (12886, 60, 24) samples
   📅 Time range: 2018-01-13 01:00:00 to 2021-01-05 11:00:00
   📈 Target distribution: 0.503

🧠 Building Hybrid CNN-LSTM-Cross-Attention model...
📊 Total parameters: 245,000+

⚖️ Class weights: {0.0: 1.02, 1.0: 0.98}

Epoch 15/100
387/387 ━━━━━━━━━━━━━━━━━━━━ 18s 46ms/step 
- accuracy: 0.5420 - loss: 0.6756 - precision: 0.5324 - recall: 0.5680
- val_accuracy: 0.5290 - val_loss: 0.6892 - val_precision: 0.5184 - val_recall: 0.5876

📊 Test Results:
   Accuracy: 0.5518
   Precision: 0.5342
   Recall: 0.5891
   F1-Score: 0.5606

💰 Total Return: 12.34%
📈 Sharpe Ratio: 1.68
📉 Max Drawdown: -8.45%
🎯 Win Rate: 55.23%
```

### **Success Indicators**
- ✅ **Balanced metrics**: Both precision and recall > 0.50
- ✅ **Learning progression**: Loss decreasing, metrics improving
- ✅ **Outperforms baseline**: Sharpe ratio > 1.5
- ✅ **Risk management**: Max drawdown < 15%
- ✅ **Confidence intervals**: Meaningful uncertainty quantification

---

## 🔮 **Future Enhancements**

### **1. Model Architecture**
- **Transformer-based attention**: Replace LSTM with transformer blocks
- **Multi-head attention**: Separate attention for different feature groups
- **Ensemble methods**: Combine multiple model predictions
- **Online learning**: Adapt to new market regimes

### **2. Feature Engineering**
- **Sentiment analysis**: News and social media sentiment
- **Macroeconomic indicators**: Interest rates, inflation, GDP
- **Cross-asset features**: Bond yields, commodity prices, volatility indices
- **Alternative data**: Google Trends, satellite data, social signals

### **3. Risk Management**
- **Dynamic position sizing**: Based on model confidence
- **Portfolio optimization**: Multi-currency allocation
- **Regime detection**: Adapt strategy to market conditions
- **Stop-loss optimization**: Dynamic risk management

### **4. Deployment**
- **Real-time inference**: Live trading implementation
- **API development**: REST API for model serving
- **Model monitoring**: Performance tracking in production
- **A/B testing**: Compare strategy variants

---

## 📚 **References & Research**

### **Key Papers**
1. **Peng et al. (2024)**: "Attention-based CNN–LSTM for high-frequency multiple cryptocurrency trend prediction." Expert Systems with Applications.
2. **LeCun et al. (2015)**: "Deep learning." Nature.
3. **Hochreiter & Schmidhuber (1997)**: "Long short-term memory." Neural computation.

### **Technical Analysis Libraries**
- **TA-Lib**: Technical Analysis Library
- **pandas-ta**: Technical Analysis Indicators
- **ta**: Python Technical Analysis Library

### **Deep Learning Frameworks**
- **TensorFlow 2.16+**: Main framework
- **Keras**: High-level API
- **scikit-learn**: Preprocessing and metrics

---

## 💡 **Tips for Extension**

### **For Researchers**
1. **Start with this codebase**: Proven architecture and preprocessing
2. **Modify target creation**: Experiment with different thresholds
3. **Add new features**: Incorporate domain expertise
4. **Compare architectures**: Test transformers vs LSTM
5. **Validate thoroughly**: Use walk-forward analysis

### **For Practitioners**
1. **Paper trading first**: Test strategy with simulated money
2. **Monitor performance**: Track all metrics continuously
3. **Risk management**: Never risk more than you can afford
4. **Regular retraining**: Update model with new data
5. **Diversification**: Don't rely on single strategy

### **For Developers**
1. **Code modularity**: Each function has single responsibility
2. **Configuration driven**: All parameters in config.py
3. **Error handling**: Robust exception management
4. **Documentation**: Comprehensive comments and docstrings
5. **Testing**: Validate each component separately

---

## 🎯 **Project Prompt Template**

When continuing this project or asking for help, use this template:

```
I have a Forex trend prediction project using Hybrid CNN-LSTM-Cross-Attention:

**Current Status**: [Describe current state]
**Issue/Goal**: [What you want to achieve or problem to solve]

**Project Details**:
- 3 currency pairs (EURUSD, GBPUSD, USDJPY) hourly data 2018-2022
- 24 features: OHLCV + RSI + MACD for each pair
- 60-timestep sequences, 60/20/20 time-based split
- Custom Cross-Attention with Temporal + Variable attention
- Target: Binary classification (strong up/down moves)
- Python 3.12, TensorFlow 2.16+, comprehensive evaluation

**Current Architecture**:
- Parallel CNN (Conv1D 64→128) + LSTM (128 units)
- Cross-Attention (TA+VA) → Dense layers → Sigmoid output
- Monte Carlo Dropout for confidence intervals
- Class weight balancing, L2 regularization

**Files**:
- main.py: Complete implementation (1000+ lines)
- config.py: All hyperparameters and settings
- requirements.txt: Dependencies for Python 3.12

**Expected Performance**:
- Accuracy >0.55, F1-Score >0.52
- Sharpe Ratio >1.5, Max Drawdown <15%