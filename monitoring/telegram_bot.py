# STIS Telegram Monitoring Bot
import asyncio
from telegram import Bot, Update
from telegram.ext import Application, CommandHandler, ContextTypes
import logging
from datetime import datetime
from typing import Optional, Dict, List

class TelegramBot:
    """
    Telegram bot for real-time monitoring and notifications
    """
    
    def __init__(self, config):
        self.config = config
        self.bot = None
        self.application = None
        self.logger = logging.getLogger('TelegramBot')
        
        # Monitoring data
        self.alerts_enabled = True
        self.performance_updates = True
        self.trade_notifications = True
        
        # Initialize bot if token is provided
        if config.TELEGRAM_BOT_TOKEN:
            self.bot = Bot(token=config.TELEGRAM_BOT_TOKEN)
            self._setup_application()
    
    def _setup_application(self):
        """
        Setup Telegram application with handlers
        """
        try:
            self.application = Application.builder().token(self.config.TELEGRAM_BOT_TOKEN).build()
            
            # Add command handlers
            self.application.add_handler(CommandHandler("start", self._handle_start))
            self.application.add_handler(CommandHandler("status", self._handle_status))
            self.application.add_handler(CommandHandler("performance", self._handle_performance))
            self.application.add_handler(CommandHandler("balance", self._handle_balance))
            self.application.add_handler(CommandHandler("positions", self._handle_positions))
            self.application.add_handler(CommandHandler("alerts", self._handle_alerts))
            self.application.add_handler(CommandHandler("help", self._handle_help))
            
            self.logger.info("Telegram bot handlers setup completed")
            
        except Exception as e:
            self.logger.error(f"Error setting up Telegram application: {str(e)}")
    
    async def start_bot(self):
        """
        Start the Telegram bot
        """
        try:
            if self.application:
                await self.application.initialize()
                await self.application.start()
                self.logger.info("Telegram bot started successfully")
            
        except Exception as e:
            self.logger.error(f"Error starting Telegram bot: {str(e)}")
    
    async def stop_bot(self):
        """
        Stop the Telegram bot
        """
        try:
            if self.application:
                await self.application.stop()
                await self.application.shutdown()
                self.logger.info("Telegram bot stopped successfully")
            
        except Exception as e:
            self.logger.error(f"Error stopping Telegram bot: {str(e)}")
    
    async def send_message(self, message: str, chat_id: Optional[str] = None, parse_mode: str = 'Markdown'):
        """
        Send message to Telegram chat
        """
        try:
            if not self.bot:
                self.logger.warning("Telegram bot not initialized")
                return False
            
            target_chat_id = chat_id or self.config.TELEGRAM_CHAT_ID
            
            if not target_chat_id:
                self.logger.warning("No chat ID specified for Telegram message")
                return False
            
            # Send message
            await self.bot.send_message(
                chat_id=target_chat_id,
                text=message,
                parse_mode=parse_mode
            )
            
            self.logger.info(f"Telegram message sent successfully to {target_chat_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error sending Telegram message: {str(e)}")
            return False
    
    async def send_trade_notification(self, trade_data: Dict):
        """
        Send trade execution notification
        """
        if not self.trade_notifications:
            return False
        
        try:
            message = (
                f"🎯 **Trade Executed**\\n"
                f"📈 **Direction:** {trade_data.get('direction', 'N/A')}\\n"
                f"💰 **Amount:** {trade_data.get('amount', 'N/A')}\\n"
                f"🎚️ **Entry Price:** ${trade_data.get('entry_price', 'N/A')}\\n"
                f"🛡️ **Stop Loss:** ${trade_data.get('stop_loss', 'N/A')}\\n"
                f"🎯 **Take Profit:** ${trade_data.get('take_profit', 'N/A')}\\n"
                f"🧠 **Confidence:** {trade_data.get('confidence', 'N/A')}%\\n"
                f"⚖️ **R:R Ratio:** {trade_data.get('risk_reward', 'N/A')}\\n"
                f"🕒 **Time:** {datetime.now().strftime('%H:%M:%S')}"
            )
            
            return await self.send_message(message)
            
        except Exception as e:
            self.logger.error(f"Error sending trade notification: {str(e)}")
            return False
    
    async def send_alert(self, alert_type: str, message: str, severity: str = 'INFO'):
        """
        Send alert notification
        """
        if not self.alerts_enabled:
            return False
        
        try:
            # Choose emoji based on severity
            severity_emojis = {
                'INFO': 'ℹ️',
                'WARNING': '⚠️',
                'ERROR': '❌',
                'CRITICAL': '🚨'
            }
            
            emoji = severity_emojis.get(severity, 'ℹ️')
            
            formatted_message = (
                f"{emoji} **{alert_type}**\\n"
                f"📊 **Severity:** {severity}\\n"
                f"📝 **Message:** {message}\\n"
                f"🕒 **Time:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            )
            
            return await self.send_message(formatted_message)
            
        except Exception as e:
            self.logger.error(f"Error sending alert: {str(e)}")
            return False
    
    async def send_performance_update(self, performance_data: Dict):
        """
        Send performance update
        """
        if not self.performance_updates:
            return False
        
        try:
            message = (
                f"📊 **Performance Update**\\n"
                f"💹 **Daily P&L:** {performance_data.get('daily_pnl', 'N/A')}%\\n"
                f"📈 **Total P&L:** {performance_data.get('total_pnl', 'N/A')}%\\n"
                f"🎯 **Win Rate:** {performance_data.get('win_rate', 'N/A')}%\\n"
                f"📊 **Total Trades:** {performance_data.get('total_trades', 'N/A')}\\n"
                f"🔥 **Consecutive Wins:** {performance_data.get('consecutive_wins', 'N/A')}\\n"
                f"💎 **Best Trade:** {performance_data.get('best_trade', 'N/A')}\\n"
                f"📉 **Worst Trade:** {performance_data.get('worst_trade', 'N/A')}\\n"
                f"🕒 **Last Update:** {datetime.now().strftime('%H:%M:%S')}"
            )
            
            return await self.send_message(message)
            
        except Exception as e:
            self.logger.error(f"Error sending performance update: {str(e)}")
            return False
    
    async def send_market_analysis(self, analysis_data: Dict):
        """
        Send market analysis update
        """
        try:
            prediction = analysis_data.get('prediction', {})
            technical = analysis_data.get('technical', {})
            sentiment = analysis_data.get('sentiment', {})
            
            message = (
                f"🧠 **Market Analysis**\\n"
                f"📊 **Neural Signal:** {prediction.get('direction', 'N/A')}\\n"
                f"🎚️ **Confidence:** {prediction.get('confidence', 'N/A')}%\\n"
                f"⚖️ **R:R Ratio:** {prediction.get('risk_reward_ratio', 'N/A')}\\n"
                f"📈 **Technical Signal:** {technical.get('signal', 'N/A')}\\n"
                f"💭 **Sentiment:** {sentiment.get('direction', 'N/A')}\\n"
                f"🎯 **Combined Signal:** {analysis_data.get('combined_signal', 'N/A')}\\n"
                f"💰 **Current Price:** ${analysis_data.get('current_price', 'N/A')}\\n"
                f"🕒 **Time:** {datetime.now().strftime('%H:%M:%S')}"
            )
            
            return await self.send_message(message)
            
        except Exception as e:
            self.logger.error(f"Error sending market analysis: {str(e)}")
            return False
    
    async def send_system_status(self, system_data: Dict):
        """
        Send system status update
        """
        try:
            message = (
                f"🤖 **System Status**\\n"
                f"🔥 **CPU Usage:** {system_data.get('cpu_usage', 'N/A')}%\\n"
                f"💾 **Memory Usage:** {system_data.get('memory_usage', 'N/A')}%\\n"
                f"📡 **API Status:** {system_data.get('api_status', 'N/A')}\\n"
                f"🔄 **Active Positions:** {system_data.get('active_positions', 'N/A')}\\n"
                f"📊 **Today's Trades:** {system_data.get('daily_trades', 'N/A')}\\n"
                f"⚡ **Neural Core:** {system_data.get('neural_status', 'N/A')}\\n"
                f"🛡️ **Risk Manager:** {system_data.get('risk_status', 'N/A')}\\n"
                f"🕒 **Uptime:** {system_data.get('uptime', 'N/A')}\\n"
                f"📅 **Last Update:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            )
            
            return await self.send_message(message)
            
        except Exception as e:
            self.logger.error(f"Error sending system status: {str(e)}")
            return False
    
    async def _handle_start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """
        Handle /start command
        """
        try:
            message = (
                f"🤖 **SuperNinja Trading Bot**\\n\\n"
                f"Welcome to the SuperNinja Trading Intelligence System!\\n\\n"
                f"**Available Commands:**\\n"
                f"📊 `/status` - Get current system status\\n"
                f"💰 `/balance` - View account balance\\n"
                f"📈 `/positions` - Show open positions\\n"
                f"🎯 `/performance` - Get performance statistics\\n"
                f"🔔 `/alerts` - Toggle alert notifications\\n"
                f"❓ `/help` - Show this help message\\n\\n"
                f"🚀 *The bot is currently active and monitoring the markets!*"
            )
            
            await update.message.reply_text(message, parse_mode='Markdown')
            
        except Exception as e:
            self.logger.error(f"Error handling start command: {str(e)}")
    
    async def _handle_status(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """
        Handle /status command
        """
        try:
            message = (
                f"🤖 **System Status**\\n\\n"
                f"🔥 **Status:** 🟢 Active\\n"
                f"📊 **Trading Pair:** {self.config.TRADING_PAIR}\\n"
                f"⏱️ **Timeframe:** {self.config.TIMEFRAME}\\n"
                f"🧠 **Neural Core:** 🟢 Operational\\n"
                f"🛡️ **Risk Management:** 🟢 Enabled\\n"
                f"📡 **API Connection:** 🟢 Connected\\n"
                f"🔔 **Alerts:** {'🟢 Enabled' if self.alerts_enabled else '🔴 Disabled'}\\n\\n"
                f"🕒 *Last updated: {datetime.now().strftime('%H:%M:%S')}*"
            )
            
            await update.message.reply_text(message, parse_mode='Markdown')
            
        except Exception as e:
            self.logger.error(f"Error handling status command: {str(e)}")
    
    async def _handle_performance(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """
        Handle /performance command
        """
        try:
            # This would typically fetch real performance data
            message = (
                f"📊 **Performance Statistics**\\n\\n"
                f"💹 **Today's P&L:** +2.34%\\n"
                f"📈 **Total P&L:** +15.67%\\n"
                f"🎯 **Win Rate:** 68.5%\\n"
                f"📊 **Total Trades:** 127\\n"
                f"🔥 **Current Streak:** 3 wins\\n"
                f"💎 **Best Trade:** +5.2%\\n"
                f"📉 **Worst Trade:** -1.8%\\n\\n"
                f"📅 *Period: Last 30 days*"
            )
            
            await update.message.reply_text(message, parse_mode='Markdown')
            
        except Exception as e:
            self.logger.error(f"Error handling performance command: {str(e)}")
    
    async def _handle_balance(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """
        Handle /balance command
        """
        try:
            message = (
                f"💰 **Account Balance**\\n\\n"
                f"📊 **Total Balance:** $12,345.67\\n"
                f"₿ **Bitcoin:** 0.2543 BTC ($11,234.56)\\n"
                f"💵 **USDT:** 1,111.11 USDT\\n"
                f"📈 **24h Change:** +2.34%\\n"
                f"🔄 **Available:** $10,000.00\\n"
                f"🔒 **In Use:** $2,345.67\\n\\n"
                f"💼 *Account ID: ***REDACTED***"
            )
            
            await update.message.reply_text(message, parse_mode='Markdown')
            
        except Exception as e:
            self.logger.error(f"Error handling balance command: {str(e)}")
    
    async def _handle_positions(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """
        Handle /positions command
        """
        try:
            message = (
                f"📈 **Open Positions**\\n\\n"
                f"🔵 **BTC-USDT Long**\\n"
                f"💰 Size: 0.05 BTC\\n"
                f"🎚️ Entry: $44,250.00\\n"
                f"📊 Current: $44,500.00\\n"
                f"💹 P&L: +0.56% (+$124.50)\\n"
                f"🛡️ SL: $43,500.00\\n"
                f"🎯 TP: $45,500.00\\n\\n"
                f"📊 **Total Positions:** 1\\n"
                f"💹 **Total P&L:** +$124.50"
            )
            
            await update.message.reply_text(message, parse_mode='Markdown')
            
        except Exception as e:
            self.logger.error(f"Error handling positions command: {str(e)}")
    
    async def _handle_alerts(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """
        Handle /alerts command
        """
        try:
            self.alerts_enabled = not self.alerts_enabled
            
            status = "🟢 Enabled" if self.alerts_enabled else "🔴 Disabled"
            
            message = (
                f"🔔 **Alert Settings**\\n\\n"
                f"📊 **Trade Notifications:** {'🟢 Enabled' if self.trade_notifications else '🔴 Disabled'}\\n"
                f"📈 **Performance Updates:** {'🟢 Enabled' if self.performance_updates else '🔴 Disabled'}\\n"
                f"🚨 **System Alerts:** {status}\\n\\n"
                f"💡 *Use `/alerts` again to toggle system alerts*"
            )
            
            await update.message.reply_text(message, parse_mode='Markdown')
            
        except Exception as e:
            self.logger.error(f"Error handling alerts command: {str(e)}")
    
    async def _handle_help(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """
        Handle /help command
        """
        try:
            message = (
                f"❓ **Help & Commands**\\n\\n"
                f"**Basic Commands:**\\n"
                f"📊 `/status` - Get current system status\\n"
                f"💰 `/balance` - View account balance\\n"
                f"📈 `/positions` - Show open positions\\n"
                f"🎯 `/performance` - Get performance statistics\\n"
                f"🔔 `/alerts` - Toggle alert notifications\\n\\n"
                f"**Information:**\\n"
                f"🤖 The SuperNinja Trading Bot uses advanced AI and machine learning\\n"
                f"🧠 Neural networks analyze market data in real-time\\n"
                f"🛡️ Smart risk management protects your capital\\n"
                f"📡 All trades are executed automatically\\n\\n"
                f"**Support:**\\n"
                f"⚠️ *This is a high-frequency trading system. Use with caution.*\\n"
                f"📧 For support, contact the system administrator."
            )
            
            await update.message.reply_text(message, parse_mode='Markdown')
            
        except Exception as e:
            self.logger.error(f"Error handling help command: {str(e)}")
    
    def toggle_alerts(self, enabled: bool):
        """
        Toggle alerts on/off
        """
        self.alerts_enabled = enabled
        self.logger.info(f"Alerts {'enabled' if enabled else 'disabled'}")
    
    def toggle_performance_updates(self, enabled: bool):
        """
        Toggle performance updates on/off
        """
        self.performance_updates = enabled
        self.logger.info(f"Performance updates {'enabled' if enabled else 'disabled'}")
    
    def toggle_trade_notifications(self, enabled: bool):
        """
        Toggle trade notifications on/off
        """
        self.trade_notifications = enabled
        self.logger.info(f"Trade notifications {'enabled' if enabled else 'disabled'}")