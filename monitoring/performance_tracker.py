# STIS Performance Tracker
import json
import os
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional

class PerformanceTracker:
    """
    Comprehensive performance tracking and analysis system
    """
    
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger('PerformanceTracker')
        
        # Performance data storage
        self.trades_history = []
        self.daily_performance = {}
        self.performance_metrics = {}
        
        # File paths
        self.data_dir = 'data/performance'
        self.trades_file = f'{self.data_dir}/trades.json'
        self.daily_file = f'{self.data_dir}/daily_performance.json'
        self.metrics_file = f'{self.data_dir}/metrics.json'
        
        # Ensure data directory exists
        os.makedirs(self.data_dir, exist_ok=True)
        
        # Load existing data
        self.load_performance_data()
    
    def load_performance_data(self):
        """
        Load existing performance data from files
        """
        try:
            # Load trades history
            if os.path.exists(self.trades_file):
                with open(self.trades_file, 'r') as f:
                    self.trades_history = json.load(f)
                self.logger.info(f"Loaded {len(self.trades_history)} trades from file")
            
            # Load daily performance
            if os.path.exists(self.daily_file):
                with open(self.daily_file, 'r') as f:
                    self.daily_performance = json.load(f)
                self.logger.info(f"Loaded daily performance data")
            
            # Load metrics
            if os.path.exists(self.metrics_file):
                with open(self.metrics_file, 'r') as f:
                    self.performance_metrics = json.load(f)
                self.logger.info(f"Loaded performance metrics")
                
        except Exception as e:
            self.logger.error(f"Error loading performance data: {str(e)}")
    
    def save_performance_data(self):
        """
        Save performance data to files
        """
        try:
            # Save trades history
            with open(self.trades_file, 'w') as f:
                json.dump(self.trades_history, f, indent=2, default=str)
            
            # Save daily performance
            with open(self.daily_file, 'w') as f:
                json.dump(self.daily_performance, f, indent=2, default=str)
            
            # Save metrics
            with open(self.metrics_file, 'w') as f:
                json.dump(self.performance_metrics, f, indent=2, default=str)
            
            self.logger.info("Performance data saved successfully")
            
        except Exception as e:
            self.logger.error(f"Error saving performance data: {str(e)}")
    
    async def update_performance(self, market_data, prediction):
        """
        Update performance metrics with current data
        """
        try:
            current_time = datetime.now()
            today = current_time.date().isoformat()
            
            # Update daily performance
            if today not in self.daily_performance:
                self.daily_performance[today] = {
                    'date': today,
                    'trades': [],
                    'pnl': 0.0,
                    'win_rate': 0.0,
                    'max_drawdown': 0.0,
                    'sharpe_ratio': 0.0,
                    'predictions': [],
                    'confidence_avg': 0.0
                }
            
            # Add prediction to daily data
            self.daily_performance[today]['predictions'].append({
                'timestamp': current_time.isoformat(),
                'direction': prediction.get('direction', 'HOLD'),
                'confidence': prediction.get('confidence', 0),
                'risk_reward': prediction.get('risk_reward_ratio', 1.0)
            })
            
            # Update average confidence
            predictions = self.daily_performance[today]['predictions']
            if predictions:
                avg_confidence = np.mean([p['confidence'] for p in predictions])
                self.daily_performance[today]['confidence_avg'] = avg_confidence
            
            # Calculate overall metrics
            self.calculate_performance_metrics()
            
            # Save data
            self.save_performance_data()
            
        except Exception as e:
            self.logger.error(f"Error updating performance: {str(e)}")
    
    def record_trade(self, trade_data):
        """
        Record a completed trade
        """
        try:
            trade = {
                'id': trade_data.get('id', str(len(self.trades_history) + 1)),
                'timestamp': trade_data.get('timestamp', datetime.now().isoformat()),
                'symbol': trade_data.get('symbol', self.config.TRADING_PAIR),
                'direction': trade_data.get('direction', 'UNKNOWN'),
                'entry_price': trade_data.get('entry_price', 0),
                'exit_price': trade_data.get('exit_price', 0),
                'amount': trade_data.get('amount', 0),
                'pnl': trade_data.get('pnl', 0),
                'pnl_percent': trade_data.get('pnl_percent', 0),
                'duration': trade_data.get('duration', 0),
                'confidence': trade_data.get('confidence', 0),
                'risk_reward': trade_data.get('risk_reward', 1.0),
                'stop_loss': trade_data.get('stop_loss', 0),
                'take_profit': trade_data.get('take_profit', 0),
                'exit_reason': trade_data.get('exit_reason', 'UNKNOWN')
            }
            
            self.trades_history.append(trade)
            
            # Update daily performance
            trade_date = datetime.fromisoformat(trade['timestamp']).date().isoformat()
            if trade_date not in self.daily_performance:
                self.daily_performance[trade_date] = {
                    'date': trade_date,
                    'trades': [],
                    'pnl': 0.0,
                    'win_rate': 0.0,
                    'max_drawdown': 0.0,
                    'sharpe_ratio': 0.0
                }
            
            self.daily_performance[trade_date]['trades'].append(trade)
            
            # Recalculate daily metrics
            self.calculate_daily_metrics(trade_date)
            
            # Calculate overall metrics
            self.calculate_performance_metrics()
            
            # Save data
            self.save_performance_data()
            
            self.logger.info(f"Trade recorded: {trade['direction']} {trade['symbol']} P&L: {trade['pnl_percent']:.2f}%")
            
        except Exception as e:
            self.logger.error(f"Error recording trade: {str(e)}")
    
    def calculate_daily_metrics(self, date):
        """
        Calculate metrics for a specific day
        """
        try:
            if date not in self.daily_performance:
                return
            
            daily_data = self.daily_performance[date]
            trades = daily_data['trades']
            
            if not trades:
                return
            
            # Calculate total P&L
            total_pnl = sum(trade['pnl'] for trade in trades)
            daily_data['pnl'] = total_pnl
            
            # Calculate win rate
            winning_trades = [t for t in trades if t['pnl'] > 0]
            win_rate = len(winning_trades) / len(trades) if trades else 0
            daily_data['win_rate'] = win_rate
            
            # Calculate average trade metrics
            if trades:
                avg_pnl = np.mean([t['pnl'] for t in trades])
                avg_pnl_percent = np.mean([t['pnl_percent'] for t in trades])
                best_trade = max(trades, key=lambda x: x['pnl'])
                worst_trade = min(trades, key=lambda x: x['pnl'])
                
                daily_data.update({
                    'avg_pnl': avg_pnl,
                    'avg_pnl_percent': avg_pnl_percent,
                    'best_trade': best_trade['pnl_percent'],
                    'worst_trade': worst_trade['pnl_percent'],
                    'num_trades': len(trades)
                })
            
            # Calculate maximum drawdown for the day
            daily_data['max_drawdown'] = self.calculate_max_drawdown(trades)
            
            # Calculate Sharpe ratio
            daily_data['sharpe_ratio'] = self.calculate_sharpe_ratio(trades)
            
        except Exception as e:
            self.logger.error(f"Error calculating daily metrics for {date}: {str(e)}")
    
    def calculate_performance_metrics(self):
        """
        Calculate overall performance metrics
        """
        try:
            if not self.trades_history:
                return
            
            # Convert to DataFrame for easier analysis
            df = pd.DataFrame(self.trades_history)
            
            # Basic metrics
            total_trades = len(df)
            winning_trades = len(df[df['pnl'] > 0])
            losing_trades = len(df[df['pnl'] < 0])
            
            # Win rate
            win_rate = winning_trades / total_trades if total_trades > 0 else 0
            
            # Total P&L
            total_pnl = df['pnl'].sum()
            total_pnl_percent = df['pnl_percent'].sum()
            
            # Average metrics
            avg_win = df[df['pnl'] > 0]['pnl_percent'].mean() if winning_trades > 0 else 0
            avg_loss = df[df['pnl'] < 0]['pnl_percent'].mean() if losing_trades > 0 else 0
            
            # Profit factor
            total_wins = df[df['pnl'] > 0]['pnl'].sum()
            total_losses = abs(df[df['pnl'] < 0]['pnl'].sum())
            profit_factor = total_wins / total_losses if total_losses > 0 else float('inf')
            
            # Maximum consecutive wins/losses
            consecutive_wins = self.calculate_consecutive_streaks(df, 'wins')
            consecutive_losses = self.calculate_consecutive_streaks(df, 'losses')
            
            # Maximum drawdown
            max_drawdown = self.calculate_max_drawdown(self.trades_history)
            
            # Sharpe ratio
            sharpe_ratio = self.calculate_sharpe_ratio(self.trades_history)
            
            # Sortino ratio
            sortino_ratio = self.calculate_sortino_ratio(self.trades_history)
            
            # Calmar ratio
            calmar_ratio = self.calculate_calmar_ratio(self.trades_history)
            
            # Update performance metrics
            self.performance_metrics = {
                'total_trades': total_trades,
                'winning_trades': winning_trades,
                'losing_trades': losing_trades,
                'win_rate': win_rate,
                'total_pnl': total_pnl,
                'total_pnl_percent': total_pnl_percent,
                'avg_win_percent': avg_win,
                'avg_loss_percent': avg_loss,
                'profit_factor': profit_factor,
                'max_consecutive_wins': consecutive_wins,
                'max_consecutive_losses': consecutive_losses,
                'max_drawdown': max_drawdown,
                'sharpe_ratio': sharpe_ratio,
                'sortino_ratio': sortino_ratio,
                'calmar_ratio': calmar_ratio,
                'last_updated': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating performance metrics: {str(e)}")
    
    def calculate_max_drawdown(self, trades):
        """
        Calculate maximum drawdown
        """
        try:
            if not trades:
                return 0
            
            # Sort trades by timestamp
            sorted_trades = sorted(trades, key=lambda x: x['timestamp'])
            
            # Calculate cumulative P&L
            cumulative_pnl = []
            running_total = 0
            
            for trade in sorted_trades:
                running_total += trade['pnl']
                cumulative_pnl.append(running_total)
            
            if not cumulative_pnl:
                return 0
            
            # Calculate drawdown
            peak = cumulative_pnl[0]
            max_drawdown = 0
            
            for pnl in cumulative_pnl:
                if pnl > peak:
                    peak = pnl
                
                drawdown = (peak - pnl) / peak if peak != 0 else 0
                max_drawdown = max(max_drawdown, drawdown)
            
            return max_drawdown
            
        except Exception as e:
            self.logger.error(f"Error calculating max drawdown: {str(e)}")
            return 0
    
    def calculate_sharpe_ratio(self, trades, risk_free_rate=0.02):
        """
        Calculate Sharpe ratio
        """
        try:
            if not trades or len(trades) < 2:
                return 0
            
            # Get daily returns
            returns = [trade['pnl_percent'] for trade in trades]
            
            if not returns:
                return 0
            
            # Calculate excess returns
            excess_returns = [r - (risk_free_rate / 252) for r in returns]
            
            # Calculate Sharpe ratio
            mean_excess_return = np.mean(excess_returns)
            std_excess_return = np.std(excess_returns)
            
            if std_excess_return == 0:
                return 0
            
            sharpe_ratio = mean_excess_return / std_excess_return * np.sqrt(252)
            
            return sharpe_ratio
            
        except Exception as e:
            self.logger.error(f"Error calculating Sharpe ratio: {str(e)}")
            return 0
    
    def calculate_sortino_ratio(self, trades, risk_free_rate=0.02):
        """
        Calculate Sortino ratio
        """
        try:
            if not trades or len(trades) < 2:
                return 0
            
            # Get daily returns
            returns = [trade['pnl_percent'] for trade in trades]
            
            if not returns:
                return 0
            
            # Calculate excess returns
            excess_returns = [r - (risk_free_rate / 252) for r in returns]
            
            # Calculate downside deviation
            negative_returns = [r for r in excess_returns if r < 0]
            
            if not negative_returns:
                return float('inf') if np.mean(excess_returns) > 0 else 0
            
            downside_deviation = np.std(negative_returns)
            
            if downside_deviation == 0:
                return 0
            
            mean_excess_return = np.mean(excess_returns)
            sortino_ratio = mean_excess_return / downside_deviation * np.sqrt(252)
            
            return sortino_ratio
            
        except Exception as e:
            self.logger.error(f"Error calculating Sortino ratio: {str(e)}")
            return 0
    
    def calculate_calmar_ratio(self, trades):
        """
        Calculate Calmar ratio (Annual Return / Maximum Drawdown)
        """
        try:
            if not trades:
                return 0
            
            # Calculate annual return
            total_pnl_percent = sum(trade['pnl_percent'] for trade in trades)
            
            # Estimate annual return (assuming current period is representative)
            if trades:
                first_trade = min(trades, key=lambda x: x['timestamp'])
                last_trade = max(trades, key=lambda x: x['timestamp'])
                
                first_date = datetime.fromisoformat(first_trade['timestamp'])
                last_date = datetime.fromisoformat(last_trade['timestamp'])
                days = (last_date - first_date).days or 1
                
                annual_return = total_pnl_percent * (365 / days)
            else:
                annual_return = 0
            
            # Calculate maximum drawdown
            max_drawdown = self.calculate_max_drawdown(trades)
            
            if max_drawdown == 0:
                return float('inf') if annual_return > 0 else 0
            
            calmar_ratio = annual_return / max_drawdown
            
            return calmar_ratio
            
        except Exception as e:
            self.logger.error(f"Error calculating Calmar ratio: {str(e)}")
            return 0
    
    def calculate_consecutive_streaks(self, df, streak_type):
        """
        Calculate maximum consecutive wins or losses
        """
        try:
            if df.empty:
                return 0
            
            # Create binary outcome column
            df['outcome'] = np.where(df['pnl'] > 0, 1, -1)
            
            # Calculate consecutive streaks
            if streak_type == 'wins':
                streaks = (df['outcome'] != 1).cumsum()
                max_streak = df[df['outcome'] == 1].groupby(streaks).size().max()
            else:  # losses
                streaks = (df['outcome'] != -1).cumsum()
                max_streak = df[df['outcome'] == -1].groupby(streaks).size().max()
            
            return max_streak if not pd.isna(max_streak) else 0
            
        except Exception as e:
            self.logger.error(f"Error calculating consecutive streaks: {str(e)}")
            return 0
    
    def get_current_performance(self):
        """
        Get current performance summary
        """
        try:
            today = datetime.now().date().isoformat()
            
            # Get today's performance
            today_performance = self.daily_performance.get(today, {})
            
            # Get recent trades
            recent_trades = [t for t in self.trades_history 
                           if datetime.fromisoformat(t['timestamp']).date() == datetime.now().date()]
            
            # Calculate today's metrics
            if recent_trades:
                daily_pnl = sum(t['pnl_percent'] for t in recent_trades)
                daily_wins = len([t for t in recent_trades if t['pnl'] > 0])
                daily_win_rate = daily_wins / len(recent_trades) if recent_trades else 0
            else:
                daily_pnl = 0
                daily_win_rate = 0
            
            return {
                'daily_pnl': daily_pnl,
                'total_pnl': self.performance_metrics.get('total_pnl_percent', 0),
                'win_rate': self.performance_metrics.get('win_rate', 0),
                'daily_win_rate': daily_win_rate,
                'total_trades': self.performance_metrics.get('total_trades', 0),
                'daily_trades': len(recent_trades),
                'consecutive_wins': self.performance_metrics.get('max_consecutive_wins', 0),
                'max_drawdown': self.performance_metrics.get('max_drawdown', 0),
                'sharpe_ratio': self.performance_metrics.get('sharpe_ratio', 0),
                'profit_factor': self.performance_metrics.get('profit_factor', 0),
                'avg_win': self.performance_metrics.get('avg_win_percent', 0),
                'avg_loss': self.performance_metrics.get('avg_loss_percent', 0)
            }
            
        except Exception as e:
            self.logger.error(f"Error getting current performance: {str(e)}")
            return {}
    
    def get_performance_report(self, days=30):
        """
        Generate detailed performance report
        """
        try:
            # Filter trades for the specified period
            cutoff_date = datetime.now() - timedelta(days=days)
            recent_trades = [t for t in self.trades_history 
                           if datetime.fromisoformat(t['timestamp']) > cutoff_date]
            
            if not recent_trades:
                return {"error": "No trades found in the specified period"}
            
            # Create DataFrame for analysis
            df = pd.DataFrame(recent_trades)
            
            # Calculate metrics
            report = {
                'period': f"Last {days} days",
                'total_trades': len(df),
                'winning_trades': len(df[df['pnl'] > 0]),
                'losing_trades': len(df[df['pnl'] < 0]),
                'win_rate': len(df[df['pnl'] > 0]) / len(df),
                'total_pnl': df['pnl'].sum(),
                'total_pnl_percent': df['pnl_percent'].sum(),
                'avg_pnl_percent': df['pnl_percent'].mean(),
                'std_pnl_percent': df['pnl_percent'].std(),
                'max_win_percent': df['pnl_percent'].max(),
                'max_loss_percent': df['pnl_percent'].min(),
                'avg_win_percent': df[df['pnl'] > 0]['pnl_percent'].mean(),
                'avg_loss_percent': df[df['pnl'] < 0]['pnl_percent'].mean(),
                'profit_factor': df[df['pnl'] > 0]['pnl'].sum() / abs(df[df['pnl'] < 0]['pnl'].sum()),
                'sharpe_ratio': self.calculate_sharpe_ratio(recent_trades),
                'max_drawdown': self.calculate_max_drawdown(recent_trades),
                'best_trade': df.loc[df['pnl_percent'].idxmax()].to_dict() if not df.empty else None,
                'worst_trade': df.loc[df['pnl_percent'].idxmin()].to_dict() if not df.empty else None
            }
            
            return report
            
        except Exception as e:
            self.logger.error(f"Error generating performance report: {str(e)}")
            return {"error": str(e)}