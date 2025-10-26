# STIS Adaptive Neural Core - Self-Learning Intelligence System
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Conv1D, MaxPooling1D, Flatten
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import json
import os
from datetime import datetime, timedelta
import logging

class AdaptiveNeuralCore:
    """
    Advanced Neural Network with CNN + Reinforcement Learning for Adaptive Trading Intelligence
    """
    
    def __init__(self, config):
        self.config = config
        self.model = None
        self.scaler = MinMaxScaler()
        self.memory = []
        self.learning_rate = config.LEARNING_RATE
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        self.confidence_threshold = config.MIN_CONFIDENCE_THRESHOLD
        
        # Performance tracking
        self.trades_history = []
        self.prediction_accuracy = []
        self.adaptation_count = 0
        
        # Initialize logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger('AdaptiveNeuralCore')
        
    def build_cnn_lstm_model(self, input_shape):
        """
        Build Hybrid CNN-LSTM model for pattern recognition and temporal analysis
        """
        model = Sequential()
        
        # CNN layers for pattern extraction
        model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=input_shape))
        model.add(MaxPooling1D(pool_size=2))
        model.add(Dropout(0.2))
        
        model.add(Conv1D(filters=128, kernel_size=3, activation='relu'))
        model.add(MaxPooling1D(pool_size=2))
        model.add(Dropout(0.2))
        
        # LSTM layers for temporal sequence analysis
        model.add(LSTM(units=100, return_sequences=True))
        model.add(Dropout(0.3))
        
        model.add(LSTM(units=50, return_sequences=False))
        model.add(Dropout(0.3))
        
        # Dense layers for decision making
        model.add(Dense(units=25, activation='relu'))
        model.add(Dropout(0.2))
        
        # Output layers
        model.add(Dense(units=3, activation='softmax'))  # [Buy, Sell, Hold]
        model.add(Dense(units=1, activation='sigmoid'))  # Confidence score
        
        model.compile(
            optimizer=Adam(learning_rate=self.learning_rate),
            loss=['categorical_crossentropy', 'binary_crossentropy'],
            metrics=['accuracy']
        )
        
        return model
    
    def preprocess_data(self, market_data):
        """
        Preprocess market data for neural network input
        """
        features = [
            'close', 'volume', 'rsi', 'macd', 'signal', 
            'bollinger_upper', 'bollinger_lower', 'ema_short', 
            'ema_long', 'order_flow', 'sentiment_score'
        ]
        
        df = pd.DataFrame(market_data)
        df = df[features]
        
        # Handle missing values
        df = df.fillna(method='ffill').fillna(method='bfill')
        
        # Normalize data
        scaled_data = self.scaler.fit_transform(df)
        
        # Create sequences
        sequences = []
        for i in range(len(scaled_data) - self.config.NEURAL_INPUT_SIZE + 1):
            sequences.append(scaled_data[i:i + self.config.NEURAL_INPUT_SIZE])
        
        return np.array(sequences)
    
    def predict_market_direction(self, current_data):
        """
        Generate real-time market prediction with confidence score
        """
        try:
            # Preprocess current data
            processed_data = self.preprocess_data(current_data)
            
            if self.model is None:
                self.model = self.build_cnn_lstm_model(
                    (self.config.NEURAL_INPUT_SIZE, processed_data.shape[2])
                )
                self.load_model_weights()
            
            # Make prediction
            predictions = self.model.predict(processed_data[-1:], verbose=0)
            direction_pred = predictions[0][0]
            confidence_pred = predictions[1][0][0]
            
            # Interpret prediction
            direction = np.argmax(direction_pred)
            direction_map = {0: 'HOLD', 1: 'BUY', 2: 'SELL'}
            
            # Calculate risk/reward ratio
            current_price = current_data[-1]['close']
            volatility = self.calculate_volatility(current_data)
            risk_reward = self.calculate_risk_reward(direction, current_price, volatility)
            
            prediction = {
                'direction': direction_map[direction],
                'confidence': float(confidence_pred * 100),
                'risk_reward_ratio': risk_reward,
                'timestamp': datetime.now().isoformat(),
                'current_price': current_price,
                'volatility': volatility
            }
            
            self.logger.info(f"Prediction: {direction_map[direction]} with {confidence_pred[0]*100:.2f}% confidence")
            
            return prediction
            
        except Exception as e:
            self.logger.error(f"Prediction error: {str(e)}")
            return None
    
    def calculate_volatility(self, data):
        """
        Calculate market volatility for risk assessment
        """
        prices = [d['close'] for d in data[-20:]]
        returns = np.diff(np.log(prices))
        return float(np.std(returns) * np.sqrt(252))
    
    def calculate_risk_reward(self, direction, current_price, volatility):
        """
        Calculate Risk/Reward ratio based on prediction and market conditions
        """
        if direction == 1:  # BUY
            potential_profit = current_price * (1 + volatility * 2)
            potential_loss = current_price * (1 - volatility)
        elif direction == 2:  # SELL
            potential_profit = current_price * (1 - volatility * 2)
            potential_loss = current_price * (1 + volatility)
        else:  # HOLD
            return 1.0
        
        risk = abs(current_price - potential_loss)
        reward = abs(potential_profit - current_price)
        
        return reward / risk if risk > 0 else 1.0
    
    def learn_from_trade(self, trade_result):
        """
        Reinforcement Learning: Learn from trade outcomes
        """
        # Store trade in memory
        self.memory.append(trade_result)
        
        # Update prediction accuracy
        if 'prediction' in trade_result and 'actual_outcome' in trade_result:
            accuracy = 1 if trade_result['prediction'] == trade_result['actual_outcome'] else 0
            self.prediction_accuracy.append(accuracy)
        
        # Adaptive learning: Adjust strategy based on performance
        if len(self.prediction_accuracy) >= 10:
            recent_accuracy = np.mean(self.prediction_accuracy[-10:])
            
            if recent_accuracy < 0.6:  # If accuracy is low
                self.adapt_strategy()
            
            # Update exploration rate
            if recent_accuracy > 0.7:
                self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
    
    def adapt_strategy(self):
        """
        Self-adaptation: Modify neural network architecture and parameters
        """
        self.logger.info("Initiating strategy adaptation...")
        
        # Adjust learning rate
        self.learning_rate *= 0.9
        
        # Retrain with recent data
        if len(self.memory) >= 50:
            self.retrain_model()
        
        self.adaptation_count += 1
        self.logger.info(f"Strategy adaptation #{self.adaptation_count} completed")
    
    def retrain_model(self):
        """
        Retrain neural network with latest data
        """
        # Prepare training data from memory
        # Implementation would extract features and labels from trade history
        self.logger.info("Retraining model with latest data...")
        
        # Placeholder for actual training logic
        pass
    
    def save_model_weights(self):
        """
        Save current model state
        """
        if self.model:
            model_path = f"data/models/neural_core_{datetime.now().strftime('%Y%m%d')}.h5"
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            self.model.save_weights(model_path)
            self.logger.info(f"Model weights saved to {model_path}")
    
    def load_model_weights(self):
        """
        Load latest model weights
        """
        try:
            model_dir = "data/models"
            if os.path.exists(model_dir):
                models = [f for f in os.listdir(model_dir) if f.endswith('.h5')]
                if models:
                    latest_model = max(models)
                    self.model.load_weights(f"{model_dir}/{latest_model}")
                    self.logger.info(f"Loaded model weights: {latest_model}")
        except Exception as e:
            self.logger.warning(f"Could not load model weights: {str(e)}")
    
    def get_performance_metrics(self):
        """
        Get current performance metrics
        """
        if not self.prediction_accuracy:
            return {}
        
        return {
            'total_trades': len(self.trades_history),
            'accuracy': np.mean(self.prediction_accuracy),
            'adaptations': self.adaptation_count,
            'epsilon': self.epsilon,
            'learning_rate': self.learning_rate
        }