# STIS Main Orchestrator - SuperNinja Trading Intelligence System
import asyncio
import time
import logging
from datetime import datetime, timedelta
import signal
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config.settings import Config
from core.brain import AdaptiveNeuralCore
from core.order_validation import OrderValidator
from core.execute_trade import TradeExecutor
from analysis.technical_analyzer import TechnicalAnalyzer
from analysis.sentiment_analyzer import SentimentAnalyzer
from api.okx_client import OKXClient
from monitoring.telegram_bot import TelegramBot
from monitoring.performance_tracker import PerformanceTracker

class SuperNinjaTradingSystem:
    """
    Main orchestrator for the SuperNinja Trading Intelligence System
    """
    
    def __init__(self):
        self.config = Config()
        self.running = False
        self.last_analysis_time = None
        self.analysis_interval = 60  # seconds
        
        # Initialize components
        self.brain = None
        self.validator = None
        self.executor = None
        self.technical_analyzer = None
        self.sentiment_analyzer = None
        self.okx_client = None
        self.telegram_bot = None
        self.performance_tracker = None
        
        # Setup logging
        self.setup_logging()
        self.logger = logging.getLogger('SuperNinjaTradingSystem')
        
        # Register signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
    
    def setup_logging(self):
        """
        Setup comprehensive logging system
        """
        os.makedirs('logs', exist_ok=True)
        
        # Create file handler
        file_handler = logging.FileHandler('logs/stis.log')
        file_handler.setLevel(logging.INFO)
        
        # Create console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # Add handlers to root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.INFO)
        root_logger.addHandler(file_handler)
        root_logger.addHandler(console_handler)
    
    async def initialize(self):
        """
        Initialize all system components
        """
        self.logger.info("Initializing SuperNinja Trading Intelligence System...")
        
        try:
            # Initialize core components
            self.brain = AdaptiveNeuralCore(self.config)
            self.validator = OrderValidator(self.config)
            self.executor = TradeExecutor(self.config)
            
            # Initialize analysis components
            self.technical_analyzer = TechnicalAnalyzer(self.config)
            self.sentiment_analyzer = SentimentAnalyzer(self.config)
            
            # Initialize API client
            self.okx_client = OKXClient(self.config)
            await self.okx_client.initialize()
            
            # Initialize monitoring components
            self.telegram_bot = TelegramBot(self.config)
            self.performance_tracker = PerformanceTracker(self.config)
            
            # Initialize executor session
            await self.executor.initialize()
            
            # Load historical data for initial analysis
            await self.load_historical_data()
            
            self.logger.info("✅ All components initialized successfully")
            await self.send_telegram_message("🚀 SuperNinja Trading System initialized and ready!")
            
        except Exception as e:
            self.logger.error(f"❌ Initialization failed: {str(e)}")
            raise
    
    async def load_historical_data(self):
        """
        Load historical data for neural network training
        """
        self.logger.info("Loading historical market data...")
        
        try:
            # Get historical data for the last 30 days
            end_time = datetime.now()
            start_time = end_time - timedelta(days=30)
            
            historical_data = await self.okx_client.get_historical_klines(
                self.config.TRADING_PAIR,
                self.config.TIMEFRAME,
                start_time,
                end_time
            )
            
            if historical_data:
                self.logger.info(f"Loaded {len(historical_data)} historical data points")
                
                # Pre-train the neural network with historical data
                self.logger.info("Pre-training neural network...")
                # self.brain.pretrain_with_historical_data(historical_data)
                
            else:
                self.logger.warning("No historical data loaded")
                
        except Exception as e:
            self.logger.error(f"Error loading historical data: {str(e)}")
    
    async def start(self):
        """
        Start the main trading loop
        """
        self.logger.info("🎯 Starting SuperNinja Trading System...")
        
        await self.initialize()
        self.running = True
        
        # Send startup notification
        await self.send_telegram_message(
            f"🤖 SuperNinja Trading System Started\\n"
            f"📊 Trading Pair: {self.config.TRADING_PAIR}\\n"
            f"⏱️ Timeframe: {self.config.TIMEFRAME}\\n"
            f"🧠 Neural Core: Active\\n"
            f"🔒 Risk Management: Enabled"
        )
        
        # Main trading loop
        while self.running:
            try:
                await self.trading_loop()
                await asyncio.sleep(self.analysis_interval)
                
            except Exception as e:
                self.logger.error(f"Error in trading loop: {str(e)}")
                await asyncio.sleep(10)  # Wait before retrying
    
    async def trading_loop(self):
        """
        Main trading logic loop
        """
        current_time = datetime.now()
        
        try:
            # 1. Get current market data
            market_data = await self.get_current_market_data()
            if not market_data:
                self.logger.warning("No market data available")
                return
            
            # 2. Perform technical analysis
            technical_signals = await self.technical_analyzer.analyze(market_data)
            
            # 3. Perform sentiment analysis
            sentiment_signals = await self.sentiment_analyzer.analyze()
            
            # 4. Generate neural network prediction
            combined_data = self.combine_analysis_data(market_data, technical_signals, sentiment_signals)
            prediction = self.brain.predict_market_direction(combined_data)
            
            if not prediction:
                self.logger.warning("No prediction generated")
                return
            
            # 5. Evaluate trading opportunity
            trading_decision = await self.evaluate_trading_opportunity(prediction, technical_signals, sentiment_signals)
            
            # 6. Execute trade if opportunity exists
            if trading_decision['should_trade']:
                await self.execute_trading_decision(trading_decision, prediction)
            
            # 7. Update performance tracking
            await self.performance_tracker.update_performance(market_data, prediction)
            
            # 8. Send periodic updates
            if self.should_send_update(current_time):
                await self.send_periodic_update(prediction, technical_signals)
            
            self.last_analysis_time = current_time
            
        except Exception as e:
            self.logger.error(f"Error in trading loop: {str(e)}")
            await self.send_telegram_message(f"⚠️ Trading loop error: {str(e)}")
    
    async def get_current_market_data(self):
        """
        Get current market data from OKX
        """
        try:
            # Get recent klines
            klines = await self.okx_client.get_recent_klines(
                self.config.TRADING_PAIR,
                self.config.TIMEFRAME,
                limit=100
            )
            
            # Get order book
            orderbook = await self.okx_client.get_orderbook(self.config.TRADING_PAIR, 20)
            
            # Get ticker data
            ticker = await self.okx_client.get_ticker(self.config.TRADING_PAIR)
            
            return {
                'klines': klines,
                'orderbook': orderbook,
                'ticker': ticker,
                'timestamp': datetime.now()
            }
            
        except Exception as e:
            self.logger.error(f"Error getting market data: {str(e)}")
            return None
    
    def combine_analysis_data(self, market_data, technical_signals, sentiment_signals):
        """
        Combine all analysis data for neural network input
        """
        # Extract price data from klines
        if market_data and market_data.get('klines'):
            latest_kline = market_data['klines'][-1]
            
            combined_data = {
                'timestamp': datetime.now().isoformat(),
                'close': float(latest_kline[4]),  # Close price
                'volume': float(latest_kline[5]),  # Volume
                'high': float(latest_kline[2]),    # High price
                'low': float(latest_kline[3]),     # Low price
                'open': float(latest_kline[1]),    # Open price
            }
            
            # Add technical indicators
            if technical_signals:
                combined_data.update(technical_signals)
            
            # Add sentiment score
            if sentiment_signals:
                combined_data['sentiment_score'] = sentiment_signals.get('overall_sentiment', 0.5)
            
            return combined_data
        
        return None
    
    async def evaluate_trading_opportunity(self, prediction, technical_signals, sentiment_signals):
        """
        Evaluate if a trading opportunity exists
        """
        decision = {
            'should_trade': False,
            'reason': '',
            'confidence': prediction.get('confidence', 0),
            'direction': prediction.get('direction', 'HOLD'),
            'risk_reward': prediction.get('risk_reward_ratio', 1.0)
        }
        
        # Check if confidence meets threshold
        if prediction.get('confidence', 0) < self.config.MIN_CONFIDENCE_THRESHOLD * 100:
            decision['reason'] = f"Low confidence: {prediction['confidence']:.2f}%"
            return decision
        
        # Check if risk/reward is acceptable
        if prediction.get('risk_reward_ratio', 0) < 1.5:
            decision['reason'] = f"Low R:R ratio: {prediction['risk_reward_ratio']:.2f}"
            return decision
        
        # Check if prediction is not HOLD
        if prediction.get('direction') == 'HOLD':
            decision['reason'] = "Neutral signal - no trading opportunity"
            return decision
        
        # Check technical indicators alignment
        if technical_signals:
            # Add technical validation logic here
            pass
        
        # Check sentiment alignment
        if sentiment_signals:
            # Add sentiment validation logic here
            pass
        
        # All checks passed - trading opportunity exists
        decision['should_trade'] = True
        decision['reason'] = "High confidence trading opportunity detected"
        
        return decision
    
    async def execute_trading_decision(self, trading_decision, prediction):
        """
        Execute trading decision
        """
        try:
            # Get current price
            ticker = await self.okx_client.get_ticker(self.config.TRADING_PAIR)
            current_price = float(ticker[0]['last'])
            
            # Calculate position size
            position_size = self.calculate_position_size(prediction, current_price)
            
            # Create order
            order = {
                'type': 'market',  # Use market orders for fast execution
                'symbol': self.config.TRADING_PAIR,
                'side': trading_decision['direction'].lower(),
                'amount': position_size,
                'price': current_price,
                'prediction': prediction
            }
            
            # Add stop loss and take profit
            if trading_decision['direction'] == 'BUY':
                order['stop_loss'] = current_price * (1 - self.config.DEFAULT_STOP_LOSS)
                order['take_profit'] = current_price * (1 + self.config.DEFAULT_TAKE_PROFIT)
            else:
                order['stop_loss'] = current_price * (1 + self.config.DEFAULT_STOP_LOSS)
                order['take_profit'] = current_price * (1 - self.config.DEFAULT_TAKE_PROFIT)
            
            # Validate order
            validation_result = self.validator.validate_order(
                order, 
                await self.get_historical_context(), 
                prediction
            )
            
            if not validation_result['valid']:
                self.logger.warning(f"Order validation failed: {validation_result['reason']}")
                await self.send_telegram_message(f"❌ Order rejected: {validation_result['reason']}")
                return
            
            # Apply adjusted parameters from validation
            order.update(validation_result['adjusted_params'])
            
            # Execute order
            execution_result = await self.executor.execute_order(order, prediction)
            
            if execution_result['success']:
                self.logger.info(f"✅ Order executed successfully: {execution_result['order_id']}")
                
                # Add to active orders in validator
                self.validator.add_active_order({
                    'id': execution_result['order_id'],
                    'symbol': order['symbol'],
                    'side': order['side'],
                    'amount': order['amount'],
                    'timestamp': datetime.now()
                })
                
                # Send notification
                await self.send_telegram_message(
                    f"🎯 Trade Executed\\n"
                    f"📈 Direction: {trading_decision['direction']}\\n"
                    f"💰 Amount: {order['amount']:.6f}\\n"
                    f"🎚️ Entry: ${current_price:.2f}\\n"
                    f"🛡️ SL: ${order['stop_loss']:.2f}\\n"
                    f"🎯 TP: ${order['take_profit']:.2f}\\n"
                    f"🧠 Confidence: {prediction['confidence']:.2f}%\\n"
                    f"⚖️ R:R: {prediction['risk_reward_ratio']:.2f}"
                )
                
                # Learn from this trade
                self.brain.learn_from_trade({
                    'order_id': execution_result['order_id'],
                    'prediction': trading_decision['direction'],
                    'confidence': prediction['confidence'],
                    'timestamp': datetime.now()
                })
                
            else:
                self.logger.error(f"❌ Order execution failed: {execution_result['reason']}")
                await self.send_telegram_message(f"❌ Order failed: {execution_result['reason']}")
                
        except Exception as e:
            self.logger.error(f"Error executing trading decision: {str(e)}")
            await self.send_telegram_message(f"⚠️ Execution error: {str(e)}")
    
    def calculate_position_size(self, prediction, current_price):
        """
        Calculate optimal position size based on confidence and risk parameters
        """
        base_size = self.config.MAX_POSITION_SIZE
        
        # Adjust based on confidence
        confidence_factor = prediction.get('confidence', 50) / 100
        
        # Adjust based on risk/reward ratio
        rr_factor = min(prediction.get('risk_reward_ratio', 1.0) / 2.0, 1.0)
        
        # Calculate final position size
        position_size = base_size * confidence_factor * rr_factor
        
        # Ensure minimum position size
        position_size = max(position_size, 0.001)  # Minimum 0.001 BTC
        
        return position_size
    
    async def get_historical_context(self):
        """
        Get historical data for context in validation
        """
        # This would return recent market data for validation
        return []
    
    def should_send_update(self, current_time):
        """
        Check if periodic update should be sent
        """
        if not self.last_analysis_time:
            return True
        
        # Send update every hour
        return (current_time - self.last_analysis_time).total_seconds() >= 3600
    
    async def send_periodic_update(self, prediction, technical_signals):
        """
        Send periodic status update
        """
        try:
            performance = self.performance_tracker.get_current_performance()
            
            message = (
                f"📊 SuperNinja Status Update\\n"
                f"🧠 Current Signal: {prediction.get('direction', 'HOLD')}\\n"
                f"🎚️ Confidence: {prediction.get('confidence', 0):.2f}%\\n"
                f"⚖️ R:R Ratio: {prediction.get('risk_reward_ratio', 0):.2f}\\n"
                f"💹 Today's P&L: {performance.get('daily_pnl', 0):.2f}%\\n"
                f"🎯 Win Rate: {performance.get('win_rate', 0):.2f}%\\n"
                f"📈 Total Trades: {performance.get('total_trades', 0)}"
            )
            
            await self.send_telegram_message(message)
            
        except Exception as e:
            self.logger.error(f"Error sending periodic update: {str(e)}")
    
    async def send_telegram_message(self, message):
        """
        Send message via Telegram bot
        """
        try:
            if self.telegram_bot:
                await self.telegram_bot.send_message(message)
        except Exception as e:
            self.logger.error(f"Error sending Telegram message: {str(e)}")
    
    def signal_handler(self, signum, frame):
        """
        Handle shutdown signals
        """
        self.logger.info(f"Received signal {signum}, shutting down gracefully...")
        self.running = False
    
    async def shutdown(self):
        """
        Graceful shutdown of all components
        """
        self.logger.info("Shutting down SuperNinja Trading System...")
        
        try:
            # Save neural network weights
            if self.brain:
                self.brain.save_model_weights()
            
            # Close API sessions
            if self.executor:
                await self.executor.close_session()
            
            if self.okx_client:
                await self.okx_client.close_session()
            
            # Send shutdown notification
            await self.send_telegram_message("🛑 SuperNinja Trading System shutdown complete")
            
            self.logger.info("✅ Shutdown complete")
            
        except Exception as e:
            self.logger.error(f"Error during shutdown: {str(e)}")

async def main():
    """
    Main entry point
    """
    stis = SuperNinjaTradingSystem()
    
    try:
        await stis.start()
    except KeyboardInterrupt:
        print("\\n🛑 Received keyboard interrupt...")
    except Exception as e:
        print(f"❌ Fatal error: {str(e)}")
    finally:
        await stis.shutdown()

if __name__ == "__main__":
    asyncio.run(main())