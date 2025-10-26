# STIS Technical Analysis Module
import numpy as np
import pandas as pd
import talib
from typing import Dict, List, Optional
import logging

class TechnicalAnalyzer:
    """
    Advanced technical analysis with multiple indicators
    """
    
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger('TechnicalAnalyzer')
    
    async def analyze(self, market_data):
        """
        Perform comprehensive technical analysis
        """
        try:
            if not market_data or 'klines' not in market_data:
                return None
            
            # Convert klines to DataFrame
            df = self._klines_to_dataframe(market_data['klines'])
            
            if df.empty or len(df) < 50:
                self.logger.warning("Insufficient data for technical analysis")
                return None
            
            # Calculate all indicators
            indicators = {}
            
            # 1. Bollinger Bands
            bb_indicators = self._calculate_bollinger_bands(df)
            indicators.update(bb_indicators)
            
            # 2. Order Flow Analysis
            order_flow = self._analyze_order_flow(market_data.get('orderbook', []))
            indicators.update(order_flow)
            
            # 3. MACD
            macd_indicators = self._calculate_macd(df)
            indicators.update(macd_indicators)
            
            # 4. EMA
            ema_indicators = self._calculate_ema(df)
            indicators.update(ema_indicators)
            
            # 5. RSI
            rsi_indicators = self._calculate_rsi(df)
            indicators.update(rsi_indicators)
            
            # 6. Volume Profile
            volume_profile = self._calculate_volume_profile(df)
            indicators.update(volume_profile)
            
            # 7. Additional indicators
            additional_indicators = self._calculate_additional_indicators(df)
            indicators.update(additional_indicators)
            
            # Generate overall technical signal
            technical_signal = self._generate_technical_signal(indicators)
            indicators['technical_signal'] = technical_signal
            
            self.logger.info(f"Technical analysis completed. Signal: {technical_signal['direction']}")
            
            return indicators
            
        except Exception as e:
            self.logger.error(f"Technical analysis error: {str(e)}")
            return None
    
    def _klines_to_dataframe(self, klines):
        """
        Convert klines data to pandas DataFrame
        """
        if not klines:
            return pd.DataFrame()
        
        data = []
        for kline in klines:
            data.append({
                'timestamp': pd.to_datetime(int(kline[0]), unit='ms'),
                'open': float(kline[1]),
                'high': float(kline[2]),
                'low': float(kline[3]),
                'close': float(kline[4]),
                'volume': float(kline[5])
            })
        
        df = pd.DataFrame(data)
        df.set_index('timestamp', inplace=True)
        return df
    
    def _calculate_bollinger_bands(self, df):
        """
        Calculate Bollinger Bands
        """
        try:
            # Calculate Bollinger Bands using TA-Lib
            upper, middle, lower = talib.BBANDS(
                df['close'].values,
                timeperiod=self.config.BOLLINGER_PERIOD,
                nbdevup=self.config.BOLLINGER_STD,
                nbdevdn=self.config.BOLLINGER_STD
            )
            
            current_price = df['close'].iloc[-1]
            current_upper = upper[-1]
            current_middle = middle[-1]
            current_lower = lower[-1]
            
            # Calculate Bollinger Band position
            bb_position = (current_price - current_lower) / (current_upper - current_lower)
            
            # Calculate Bollinger Band width
            bb_width = (current_upper - current_lower) / current_middle
            
            # Generate Bollinger Band signal
            if bb_position < 0.1:  # Near lower band
                bb_signal = 'BUY'
                bb_strength = (0.1 - bb_position) * 10
            elif bb_position > 0.9:  # Near upper band
                bb_signal = 'SELL'
                bb_strength = (bb_position - 0.9) * 10
            else:
                bb_signal = 'HOLD'
                bb_strength = 0
            
            return {
                'bb_upper': current_upper,
                'bb_middle': current_middle,
                'bb_lower': current_lower,
                'bb_position': bb_position,
                'bb_width': bb_width,
                'bb_signal': bb_signal,
                'bb_strength': bb_strength
            }
            
        except Exception as e:
            self.logger.error(f"Bollinger Bands calculation error: {str(e)}")
            return {}
    
    def _analyze_order_flow(self, orderbook):
        """
        Analyze order book flow
        """
        try:
            if not orderbook:
                return {'order_flow_signal': 'HOLD', 'order_flow_strength': 0}
            
            bids = orderbook.get('bids', [])
            asks = orderbook.get('asks', [])
            
            if not bids or not asks:
                return {'order_flow_signal': 'HOLD', 'order_flow_strength': 0}
            
            # Calculate bid/ask imbalance
            total_bid_volume = sum(float(bid[1]) for bid in bids[:10])
            total_ask_volume = sum(float(ask[1]) for ask in asks[:10])
            
            imbalance = (total_bid_volume - total_ask_volume) / (total_bid_volume + total_ask_volume)
            
            # Calculate spread
            best_bid = float(bids[0][0])
            best_ask = float(asks[0][0])
            spread = (best_ask - best_bid) / best_bid
            
            # Generate order flow signal
            if imbalance > 0.2:  # Strong buying pressure
                flow_signal = 'BUY'
                flow_strength = min(imbalance * 2, 1.0)
            elif imbalance < -0.2:  # Strong selling pressure
                flow_signal = 'SELL'
                flow_strength = min(abs(imbalance) * 2, 1.0)
            else:
                flow_signal = 'HOLD'
                flow_strength = 0
            
            return {
                'order_flow_signal': flow_signal,
                'order_flow_strength': flow_strength,
                'bid_ask_imbalance': imbalance,
                'spread': spread,
                'total_bid_volume': total_bid_volume,
                'total_ask_volume': total_ask_volume
            }
            
        except Exception as e:
            self.logger.error(f"Order flow analysis error: {str(e)}")
            return {'order_flow_signal': 'HOLD', 'order_flow_strength': 0}
    
    def _calculate_macd(self, df):
        """
        Calculate MACD indicator
        """
        try:
            # Calculate MACD using TA-Lib
            macd, signal, histogram = talib.MACD(
                df['close'].values,
                fastperiod=self.config.MACD_FAST,
                slowperiod=self.config.MACD_SLOW,
                signalperiod=self.config.MACD_SIGNAL
            )
            
            current_macd = macd[-1]
            current_signal = signal[-1]
            current_histogram = histogram[-1]
            
            # Generate MACD signal
            if current_histogram > 0 and current_histogram > histogram[-2]:
                macd_signal = 'BUY'
                macd_strength = min(current_histogram / abs(current_histogram), 1.0)
            elif current_histogram < 0 and current_histogram < histogram[-2]:
                macd_signal = 'SELL'
                macd_strength = min(abs(current_histogram) / abs(current_histogram), 1.0)
            else:
                macd_signal = 'HOLD'
                macd_strength = 0
            
            # Check for MACD crossover
            crossover = self._check_macd_crossover(macd, signal)
            
            return {
                'macd': current_macd,
                'macd_signal_line': current_signal,
                'macd_histogram': current_histogram,
                'macd_signal': macd_signal,
                'macd_strength': macd_strength,
                'macd_crossover': crossover
            }
            
        except Exception as e:
            self.logger.error(f"MACD calculation error: {str(e)}")
            return {}
    
    def _check_macd_crossover(self, macd, signal):
        """
        Check for MACD crossover
        """
        try:
            if len(macd) < 2 or len(signal) < 2:
                return None
            
            # Check for bullish crossover
            if macd[-2] <= signal[-2] and macd[-1] > signal[-1]:
                return 'BULLISH'
            # Check for bearish crossover
            elif macd[-2] >= signal[-2] and macd[-1] < signal[-1]:
                return 'BEARISH'
            else:
                return None
                
        except Exception as e:
            self.logger.error(f"MACD crossover check error: {str(e)}")
            return None
    
    def _calculate_ema(self, df):
        """
        Calculate Exponential Moving Averages
        """
        try:
            # Calculate EMAs using TA-Lib
            ema_short = talib.EMA(df['close'].values, timeperiod=self.config.EMA_SHORT)
            ema_long = talib.EMA(df['close'].values, timeperiod=self.config.EMA_LONG)
            
            current_ema_short = ema_short[-1]
            current_ema_long = ema_long[-1]
            current_price = df['close'].iloc[-1]
            
            # Calculate EMA position
            ema_position_short = (current_price - current_ema_short) / current_ema_short
            ema_position_long = (current_price - current_ema_long) / current_ema_long
            
            # Generate EMA signal
            if current_price > current_ema_short > current_ema_long:
                ema_signal = 'BUY'
                ema_strength = min(ema_position_short * 5, 1.0)
            elif current_price < current_ema_short < current_ema_long:
                ema_signal = 'SELL'
                ema_strength = min(abs(ema_position_short) * 5, 1.0)
            else:
                ema_signal = 'HOLD'
                ema_strength = 0
            
            # Check for EMA crossover
            crossover = self._check_ema_crossover(ema_short, ema_long)
            
            return {
                'ema_short': current_ema_short,
                'ema_long': current_ema_long,
                'ema_signal': ema_signal,
                'ema_strength': ema_strength,
                'ema_crossover': crossover,
                'ema_position_short': ema_position_short,
                'ema_position_long': ema_position_long
            }
            
        except Exception as e:
            self.logger.error(f"EMA calculation error: {str(e)}")
            return {}
    
    def _check_ema_crossover(self, ema_short, ema_long):
        """
        Check for EMA crossover
        """
        try:
            if len(ema_short) < 2 or len(ema_long) < 2:
                return None
            
            # Check for bullish crossover
            if ema_short[-2] <= ema_long[-2] and ema_short[-1] > ema_long[-1]:
                return 'BULLISH'
            # Check for bearish crossover
            elif ema_short[-2] >= ema_long[-2] and ema_short[-1] < ema_long[-1]:
                return 'BEARISH'
            else:
                return None
                
        except Exception as e:
            self.logger.error(f"EMA crossover check error: {str(e)}")
            return None
    
    def _calculate_rsi(self, df):
        """
        Calculate Relative Strength Index
        """
        try:
            # Calculate RSI using TA-Lib
            rsi = talib.RSI(df['close'].values, timeperiod=self.config.RSI_PERIOD)
            current_rsi = rsi[-1]
            
            # Generate RSI signal
            if current_rsi < 30:  # Oversold
                rsi_signal = 'BUY'
                rsi_strength = (30 - current_rsi) / 30
            elif current_rsi > 70:  # Overbought
                rsi_signal = 'SELL'
                rsi_strength = (current_rsi - 70) / 30
            else:
                rsi_signal = 'HOLD'
                rsi_strength = 0
            
            # Check for RSI divergence
            divergence = self._check_rsi_divergence(df, rsi)
            
            return {
                'rsi': current_rsi,
                'rsi_signal': rsi_signal,
                'rsi_strength': rsi_strength,
                'rsi_divergence': divergence
            }
            
        except Exception as e:
            self.logger.error(f"RSI calculation error: {str(e)}")
            return {}
    
    def _check_rsi_divergence(self, df, rsi):
        """
        Check for RSI divergence
        """
        try:
            if len(rsi) < 20:
                return None
            
            # Get recent price highs and RSI highs
            recent_prices = df['close'].iloc[-20:].values
            recent_rsi = rsi[-20:]
            
            # Find peaks
            price_peaks = self._find_peaks(recent_prices)
            rsi_peaks = self._find_peaks(recent_rsi)
            
            # Check for bearish divergence (price makes higher high, RSI makes lower high)
            if (len(price_peaks) >= 2 and len(rsi_peaks) >= 2 and
                price_peaks[-1] > price_peaks[-2] and rsi_peaks[-1] < rsi_peaks[-2]):
                return 'BEARISH'
            
            # Check for bullish divergence (price makes lower low, RSI makes higher low)
            price_troughs = self._find_troughs(recent_prices)
            rsi_troughs = self._find_troughs(recent_rsi)
            
            if (len(price_troughs) >= 2 and len(rsi_troughs) >= 2 and
                price_troughs[-1] < price_troughs[-2] and rsi_troughs[-1] > rsi_troughs[-2]):
                return 'BULLISH'
            
            return None
            
        except Exception as e:
            self.logger.error(f"RSI divergence check error: {str(e)}")
            return None
    
    def _find_peaks(self, data):
        """
        Find peaks in data
        """
        peaks = []
        for i in range(1, len(data) - 1):
            if data[i] > data[i-1] and data[i] > data[i+1]:
                peaks.append(data[i])
        return peaks
    
    def _find_troughs(self, data):
        """
        Find troughs in data
        """
        troughs = []
        for i in range(1, len(data) - 1):
            if data[i] < data[i-1] and data[i] < data[i+1]:
                troughs.append(data[i])
        return troughs
    
    def _calculate_volume_profile(self, df):
        """
        Calculate Volume Profile
        """
        try:
            # Group volume by price levels
            price_levels = pd.cut(df['close'], bins=20)
            volume_by_price = df.groupby(price_levels)['volume'].sum()
            
            # Find Point of Control (POC) - price with highest volume
            poc = volume_by_price.idxmax()
            poc_volume = volume_by_price.max()
            
            # Calculate Value Area (70% of volume)
            total_volume = volume_by_price.sum()
            value_area_volume = total_volume * 0.7
            cumulative_volume = volume_by_price.sort_values(ascending=False).cumsum()
            value_area = cumulative_volume[cumulative_volume <= value_area_volume]
            
            if len(value_area) > 0:
                va_high = value_area.index.max().right
                va_low = value_area.index.min().left
            else:
                va_high = df['close'].max()
                va_low = df['close'].min()
            
            current_price = df['close'].iloc[-1]
            
            # Generate volume profile signal
            if current_price < va_low:
                vp_signal = 'BUY'
                vp_strength = (va_low - current_price) / va_low
            elif current_price > va_high:
                vp_signal = 'SELL'
                vp_strength = (current_price - va_high) / va_high
            else:
                vp_signal = 'HOLD'
                vp_strength = 0
            
            return {
                'volume_profile_signal': vp_signal,
                'volume_profile_strength': vp_strength,
                'poc': float(poc.mid) if hasattr(poc, 'mid') else float(poc),
                'poc_volume': float(poc_volume),
                'va_high': float(va_high),
                'va_low': float(va_low)
            }
            
        except Exception as e:
            self.logger.error(f"Volume profile calculation error: {str(e)}")
            return {}
    
    def _calculate_additional_indicators(self, df):
        """
        Calculate additional technical indicators
        """
        try:
            indicators = {}
            
            # Stochastic Oscillator
            slowk, slowd = talib.STOCH(
                df['high'].values,
                df['low'].values,
                df['close'].values,
                fastk_period=14,
                slowk_period=3,
                slowd_period=3
            )
            
            indicators['stoch_k'] = slowk[-1]
            indicators['stoch_d'] = slowd[-1]
            
            # ADX (Average Directional Index)
            adx = talib.ADX(df['high'].values, df['low'].values, df['close'].values, timeperiod=14)
            indicators['adx'] = adx[-1]
            
            # ATR (Average True Range)
            atr = talib.ATR(df['high'].values, df['low'].values, df['close'].values, timeperiod=14)
            indicators['atr'] = atr[-1]
            
            # Williams %R
            willr = talib.WILLR(df['high'].values, df['low'].values, df['close'].values, timeperiod=14)
            indicators['williams_r'] = willr[-1]
            
            # CCI (Commodity Channel Index)
            cci = talib.CCI(df['high'].values, df['low'].values, df['close'].values, timeperiod=14)
            indicators['cci'] = cci[-1]
            
            return indicators
            
        except Exception as e:
            self.logger.error(f"Additional indicators calculation error: {str(e)}")
            return {}
    
    def _generate_technical_signal(self, indicators):
        """
        Generate overall technical signal from all indicators
        """
        try:
            # Collect signals from all indicators
            signals = []
            weights = []
            
            # Bollinger Bands
            if 'bb_signal' in indicators and indicators['bb_signal'] != 'HOLD':
                signals.append(indicators['bb_signal'])
                weights.append(indicators.get('bb_strength', 0.5))
            
            # Order Flow
            if 'order_flow_signal' in indicators and indicators['order_flow_signal'] != 'HOLD':
                signals.append(indicators['order_flow_signal'])
                weights.append(indicators.get('order_flow_strength', 0.5))
            
            # MACD
            if 'macd_signal' in indicators and indicators['macd_signal'] != 'HOLD':
                signals.append(indicators['macd_signal'])
                weights.append(indicators.get('macd_strength', 0.5))
            
            # EMA
            if 'ema_signal' in indicators and indicators['ema_signal'] != 'HOLD':
                signals.append(indicators['ema_signal'])
                weights.append(indicators.get('ema_strength', 0.5))
            
            # RSI
            if 'rsi_signal' in indicators and indicators['rsi_signal'] != 'HOLD':
                signals.append(indicators['rsi_signal'])
                weights.append(indicators.get('rsi_strength', 0.5))
            
            # Volume Profile
            if 'volume_profile_signal' in indicators and indicators['volume_profile_signal'] != 'HOLD':
                signals.append(indicators['volume_profile_signal'])
                weights.append(indicators.get('volume_profile_strength', 0.5))
            
            if not signals:
                return {'direction': 'HOLD', 'strength': 0, 'confidence': 0}
            
            # Weighted voting
            buy_weight = sum(w for s, w in zip(signals, weights) if s == 'BUY')
            sell_weight = sum(w for s, w in zip(signals, weights) if s == 'SELL')
            
            total_weight = buy_weight + sell_weight
            
            if total_weight == 0:
                return {'direction': 'HOLD', 'strength': 0, 'confidence': 0}
            
            # Determine final signal
            if buy_weight > sell_weight:
                direction = 'BUY'
                strength = buy_weight / total_weight
            elif sell_weight > buy_weight:
                direction = 'SELL'
                strength = sell_weight / total_weight
            else:
                direction = 'HOLD'
                strength = 0
            
            # Calculate confidence based on consensus
            consensus_signals = len([s for s in signals if s == direction])
            confidence = consensus_signals / len(signals) if signals else 0
            
            return {
                'direction': direction,
                'strength': strength,
                'confidence': confidence,
                'buy_weight': buy_weight,
                'sell_weight': sell_weight,
                'total_indicators': len(signals)
            }
            
        except Exception as e:
            self.logger.error(f"Technical signal generation error: {str(e)}")
            return {'direction': 'HOLD', 'strength': 0, 'confidence': 0}