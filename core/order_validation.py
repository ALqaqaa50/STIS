# STIS Smart Order Validation System
import numpy as np
from datetime import datetime, timedelta
import logging

class OrderValidator:
    """
    Intelligent order validation system with multi-layer security checks
    """
    
    def __init__(self, config):
        self.config = config
        self.active_orders = []
        self.daily_trades = []
        self.blacklist_periods = []
        
        # Initialize logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger('OrderValidator')
        
        # Risk metrics
        self.daily_loss_limit = 0.05  # 5% daily loss limit
        self.consecutive_loss_limit = 3
        self.consecutive_losses = 0
        
    def validate_order(self, order, market_data, prediction):
        """
        Comprehensive order validation with multiple checks
        """
        validation_result = {
            'valid': True,
            'reason': '',
            'adjusted_params': {}
        }
        
        try:
            # 1. Basic validation checks
            if not self._basic_validation(order):
                validation_result['valid'] = False
                validation_result['reason'] = 'Basic validation failed'
                return validation_result
            
            # 2. Risk management validation
            risk_result = self._validate_risk_parameters(order, market_data, prediction)
            if not risk_result['valid']:
                validation_result['valid'] = False
                validation_result['reason'] = risk_result['reason']
                return validation_result
            
            # 3. Market conditions validation
            market_result = self._validate_market_conditions(order, market_data)
            if not market_result['valid']:
                validation_result['valid'] = False
                validation_result['reason'] = market_result['reason']
                return validation_result
            
            # 4. Position sizing validation
            sizing_result = self._validate_position_size(order, market_data)
            validation_result['adjusted_params'].update(sizing_result)
            
            # 5. Time-based validation
            time_result = self._validate_timing(order)
            if not time_result['valid']:
                validation_result['valid'] = False
                validation_result['reason'] = time_result['reason']
                return validation_result
            
            # 6. Correlation validation
            correlation_result = self._validate_correlation(order)
            if not correlation_result['valid']:
                validation_result['valid'] = False
                validation_result['reason'] = correlation_result['reason']
                return validation_result
            
            self.logger.info(f"Order validation passed: {order['type']} {order['symbol']}")
            return validation_result
            
        except Exception as e:
            self.logger.error(f"Order validation error: {str(e)}")
            validation_result['valid'] = False
            validation_result['reason'] = f'Validation error: {str(e)}'
            return validation_result
    
    def _basic_validation(self, order):
        """
        Basic order structure and parameter validation
        """
        required_fields = ['type', 'symbol', 'side', 'amount', 'price']
        
        for field in required_fields:
            if field not in order:
                self.logger.error(f"Missing required field: {field}")
                return False
        
        # Validate order type
        if order['type'] not in ['market', 'limit', 'stop', 'stop_limit']:
            self.logger.error(f"Invalid order type: {order['type']}")
            return False
        
        # Validate side
        if order['side'] not in ['buy', 'sell']:
            self.logger.error(f"Invalid order side: {order['side']}")
            return False
        
        # Validate amount
        if order['amount'] <= 0:
            self.logger.error("Order amount must be positive")
            return False
        
        return True
    
    def _validate_risk_parameters(self, order, market_data, prediction):
        """
        Validate risk management parameters
        """
        result = {'valid': True, 'reason': ''}
        
        # Check confidence threshold
        if prediction and prediction.get('confidence', 0) < self.config.MIN_CONFIDENCE_THRESHOLD * 100:
            result['valid'] = False
            result['reason'] = f"Low confidence: {prediction['confidence']:.2f}% < {self.config.MIN_CONFIDENCE_THRESHOLD * 100}%"
            return result
        
        # Check risk/reward ratio
        if prediction and prediction.get('risk_reward_ratio', 0) < 1.5:
            result['valid'] = False
            result['reason'] = f"Low R:R ratio: {prediction['risk_reward_ratio']:.2f} < 1.5"
            return result
        
        # Check daily loss limit
        if self._check_daily_loss_limit():
            result['valid'] = False
            result['reason'] = "Daily loss limit reached"
            return result
        
        # Check consecutive losses
        if self.consecutive_losses >= self.consecutive_loss_limit:
            result['valid'] = False
            result['reason'] = f"Too many consecutive losses: {self.consecutive_losses}"
            return result
        
        # Validate stop loss and take profit
        if 'stop_loss' not in order:
            order['stop_loss'] = self._calculate_default_stop_loss(order, market_data)
        
        if 'take_profit' not in order:
            order['take_profit'] = self._calculate_default_take_profit(order, market_data)
        
        return result
    
    def _validate_market_conditions(self, order, market_data):
        """
        Validate if market conditions are suitable for trading
        """
        result = {'valid': True, 'reason': ''}
        
        # Check market volatility
        volatility = self._calculate_volatility(market_data)
        if volatility > 0.05:  # 5% volatility threshold
            result['valid'] = False
            result['reason'] = f"High volatility: {volatility:.2%}"
            return result
        
        # Check market liquidity
        if self._check_liquidity(market_data) < 100000:  # $100k minimum liquidity
            result['valid'] = False
            result['reason'] = "Low market liquidity"
            return result
        
        # Check spread
        spread = self._calculate_spread(market_data)
        if spread > 0.001:  # 0.1% spread threshold
            result['valid'] = False
            result['reason'] = f"Wide spread: {spread:.2%}"
            return result
        
        return result
    
    def _validate_position_size(self, order, market_data):
        """
        Validate and adjust position size based on risk
        """
        adjusted_params = {}
        
        # Calculate maximum position size based on risk
        max_size = self._calculate_max_position_size(market_data)
        
        if order['amount'] > max_size:
            adjusted_params['amount'] = max_size
            self.logger.info(f"Position size adjusted from {order['amount']} to {max_size}")
        
        # Adjust based on confidence
        if 'prediction' in order:
            confidence_factor = order['prediction'].get('confidence', 100) / 100
            adjusted_size = adjusted_params.get('amount', order['amount']) * confidence_factor
            adjusted_params['amount'] = adjusted_size
        
        return adjusted_params
    
    def _validate_timing(self, order):
        """
        Validate if timing is appropriate for trading
        """
        result = {'valid': True, 'reason': ''}
        
        current_time = datetime.now()
        
        # Check if within trading hours (if applicable)
        # For crypto markets, this might not be necessary
        
        # Check cooldown periods after losses
        if self.consecutive_losses > 0:
            last_loss_time = self._get_last_loss_time()
            if last_loss_time:
                cooldown_period = timedelta(minutes=30 * self.consecutive_losses)
                if current_time - last_loss_time < cooldown_period:
                    result['valid'] = False
                    result['reason'] = f"In cooldown period after losses"
                    return result
        
        # Check daily trade limit
        today_trades = [t for t in self.daily_trades if t.date() == current_time.date()]
        if len(today_trades) >= self.config.MAX_DAILY_TRADES:
            result['valid'] = False
            result['reason'] = "Daily trade limit reached"
            return result
        
        return result
    
    def _validate_correlation(self, order):
        """
        Validate correlation with existing positions
        """
        result = {'valid': True, 'reason': ''}
        
        # Check if similar position already exists
        for active_order in self.active_orders:
            if (active_order['symbol'] == order['symbol'] and 
                active_order['side'] == order['side']):
                result['valid'] = False
                result['reason'] = f"Similar position already active"
                return result
        
        # Check maximum concurrent positions
        if len(self.active_orders) >= self.config.MAX_CONCURRENT_POSITIONS:
            result['valid'] = False
            result['reason'] = "Maximum concurrent positions reached"
            return result
        
        return result
    
    def _calculate_default_stop_loss(self, order, market_data):
        """
        Calculate default stop loss based on market conditions
        """
        current_price = market_data[-1]['close']
        volatility = self._calculate_volatility(market_data)
        
        if order['side'] == 'buy':
            return current_price * (1 - max(self.config.DEFAULT_STOP_LOSS, volatility * 2))
        else:
            return current_price * (1 + max(self.config.DEFAULT_STOP_LOSS, volatility * 2))
    
    def _calculate_default_take_profit(self, order, market_data):
        """
        Calculate default take profit based on market conditions
        """
        current_price = market_data[-1]['close']
        volatility = self._calculate_volatility(market_data)
        
        if order['side'] == 'buy':
            return current_price * (1 + max(self.config.DEFAULT_TAKE_PROFIT, volatility * 3))
        else:
            return current_price * (1 - max(self.config.DEFAULT_TAKE_PROFIT, volatility * 3))
    
    def _calculate_volatility(self, market_data):
        """
        Calculate market volatility
        """
        if len(market_data) < 20:
            return 0.02  # Default volatility
        
        prices = [d['close'] for d in market_data[-20:]]
        returns = np.diff(np.log(prices))
        return np.std(returns)
    
    def _check_liquidity(self, market_data):
        """
        Check market liquidity
        """
        # This would typically use order book data
        # For now, return a default value
        return 500000  # $500k default liquidity
    
    def _calculate_spread(self, market_data):
        """
        Calculate bid-ask spread
        """
        # This would typically use order book data
        # For now, return a default spread
        return 0.0005  # 0.05% default spread
    
    def _calculate_max_position_size(self, market_data):
        """
        Calculate maximum position size based on risk management
        """
        # Base position size
        base_size = self.config.MAX_POSITION_SIZE
        
        # Adjust based on volatility
        volatility = self._calculate_volatility(market_data)
        volatility_adjustment = 1 / (1 + volatility * 10)
        
        # Adjust based on consecutive losses
        loss_adjustment = max(0.5, 1 - (self.consecutive_losses * 0.2))
        
        return base_size * volatility_adjustment * loss_adjustment
    
    def _check_daily_loss_limit(self):
        """
        Check if daily loss limit has been reached
        """
        today = datetime.now().date()
        today_losses = [t for t in self.daily_trades 
                       if t.date() == today and t['pnl'] < 0]
        
        total_loss = sum(t['pnl'] for t in today_losses)
        return abs(total_loss) >= self.daily_loss_limit
    
    def _get_last_loss_time(self):
        """
        Get timestamp of last losing trade
        """
        losses = [t for t in self.daily_trades if t['pnl'] < 0]
        if losses:
            return max(t['timestamp'] for t in losses)
        return None
    
    def update_trade_result(self, trade_result):
        """
        Update system with trade results for learning
        """
        self.daily_trades.append(trade_result)
        
        if trade_result['pnl'] < 0:
            self.consecutive_losses += 1
        else:
            self.consecutive_losses = 0
        
        # Remove from active orders
        self.active_orders = [o for o in self.active_orders 
                             if o['id'] != trade_result['order_id']]
        
        # Clean old trades (keep last 30 days)
        cutoff_date = datetime.now() - timedelta(days=30)
        self.daily_trades = [t for t in self.daily_trades 
                            if t['timestamp'] > cutoff_date]
    
    def add_active_order(self, order):
        """
        Add order to active orders list
        """
        self.active_orders.append(order)
    
    def get_validation_stats(self):
        """
        Get validation statistics
        """
        return {
            'active_orders': len(self.active_orders),
            'daily_trades': len([t for t in self.daily_trades 
                               if t['timestamp'].date() == datetime.now().date()]),
            'consecutive_losses': self.consecutive_losses,
            'blacklist_periods': len(self.blacklist_periods)
        }