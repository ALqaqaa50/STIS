# STIS OKX API Client
import aiohttp
import hashlib
import hmac
import base64
import json
import time
from datetime import datetime, timezone
import logging
from typing import Dict, List, Optional

class OKXClient:
    """
    OKX API client for market data and trading operations
    """
    
    def __init__(self, config):
        self.config = config
        self.base_url = "https://www.okx.com" if not config.OKX_SANDBOX else "https://www.okx.com"
        self.session = None
        self.logger = logging.getLogger('OKXClient')
    
    async def initialize(self):
        """
        Initialize aiohttp session
        """
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=10),
            headers={'Content-Type': 'application/json'}
        )
        self.logger.info("OKX client initialized")
    
    async def get_recent_klines(self, symbol, timeframe, limit=100):
        """
        Get recent kline data
        """
        try:
            endpoint = f'/api/v5/market/candles'
            params = {
                'instId': symbol,
                'bar': timeframe,
                'limit': str(limit)
            }
            
            result = await self._send_public_request('GET', endpoint, params)
            
            if result['success'] and result['data']:
                return result['data']
            else:
                self.logger.error(f"Failed to get klines: {result.get('reason', 'Unknown error')}")
                return []
                
        except Exception as e:
            self.logger.error(f"Error getting recent klines: {str(e)}")
            return []
    
    async def get_historical_klines(self, symbol, timeframe, start_time, end_time):
        """
        Get historical kline data
        """
        try:
            endpoint = f'/api/v5/market/history-candles'
            params = {
                'instId': symbol,
                'bar': timeframe,
                'before': str(int(start_time.timestamp() * 1000)),
                'after': str(int(end_time.timestamp() * 1000))
            }
            
            result = await self._send_public_request('GET', endpoint, params)
            
            if result['success'] and result['data']:
                return result['data']
            else:
                self.logger.error(f"Failed to get historical klines: {result.get('reason', 'Unknown error')}")
                return []
                
        except Exception as e:
            self.logger.error(f"Error getting historical klines: {str(e)}")
            return []
    
    async def get_ticker(self, symbol):
        """
        Get ticker information
        """
        try:
            endpoint = f'/api/v5/market/ticker'
            params = {'instId': symbol}
            
            result = await self._send_public_request('GET', endpoint, params)
            
            if result['success'] and result['data']:
                return result['data']
            else:
                self.logger.error(f"Failed to get ticker: {result.get('reason', 'Unknown error')}")
                return []
                
        except Exception as e:
            self.logger.error(f"Error getting ticker: {str(e)}")
            return []
    
    async def get_orderbook(self, symbol, depth=20):
        """
        Get order book
        """
        try:
            endpoint = f'/api/v5/market/books'
            params = {
                'instId': symbol,
                'sz': str(depth)
            }
            
            result = await self._send_public_request('GET', endpoint, params)
            
            if result['success'] and result['data']:
                orderbook = result['data'][0]
                return {
                    'bids': [[float(bid[0]), float(bid[1])] for bid in orderbook['bids']],
                    'asks': [[float(ask[0]), float(ask[1])] for ask in orderbook['asks']],
                    'timestamp': orderbook['ts']
                }
            else:
                self.logger.error(f"Failed to get orderbook: {result.get('reason', 'Unknown error')}")
                return {}
                
        except Exception as e:
            self.logger.error(f"Error getting orderbook: {str(e)}")
            return {}
    
    async def get_account_balance(self):
        """
        Get account balance
        """
        try:
            endpoint = '/api/v5/account/balance'
            result = await self._send_private_request('GET', endpoint, None)
            
            if result['success'] and result['data']:
                return result['data']
            else:
                self.logger.error(f"Failed to get account balance: {result.get('reason', 'Unknown error')}")
                return []
                
        except Exception as e:
            self.logger.error(f"Error getting account balance: {str(e)}")
            return []
    
    async def get_positions(self):
        """
        Get open positions
        """
        try:
            endpoint = '/api/v5/account/positions'
            result = await self._send_private_request('GET', endpoint, None)
            
            if result['success'] and result['data']:
                return result['data']
            else:
                self.logger.error(f"Failed to get positions: {result.get('reason', 'Unknown error')}")
                return []
                
        except Exception as e:
            self.logger.error(f"Error getting positions: {str(e)}")
            return []
    
    async def place_order(self, order_data):
        """
        Place order
        """
        try:
            endpoint = '/api/v5/trade/order'
            result = await self._send_private_request('POST', endpoint, order_data)
            
            if result['success'] and result['data']:
                return {
                    'success': True,
                    'order_id': result['data'][0]['ordId'],
                    'client_order_id': result['data'][0]['clOrdId'],
                    'raw_response': result
                }
            else:
                return {
                    'success': False,
                    'reason': result.get('reason', 'Unknown error'),
                    'raw_response': result
                }
                
        except Exception as e:
            self.logger.error(f"Error placing order: {str(e)}")
            return {'success': False, 'reason': f'Error: {str(e)}'}
    
    async def cancel_order(self, order_id):
        """
        Cancel order
        """
        try:
            endpoint = '/api/v5/trade/cancel-order'
            data = {'ordId': order_id}
            result = await self._send_private_request('POST', endpoint, data)
            
            if result['success'] and result['data']:
                return {
                    'success': True,
                    'order_id': result['data'][0]['ordId'],
                    'raw_response': result
                }
            else:
                return {
                    'success': False,
                    'reason': result.get('reason', 'Unknown error'),
                    'raw_response': result
                }
                
        except Exception as e:
            self.logger.error(f"Error canceling order: {str(e)}")
            return {'success': False, 'reason': f'Error: {str(e)}'}
    
    async def get_order_status(self, order_id):
        """
        Get order status
        """
        try:
            endpoint = f'/api/v5/trade/order'
            params = {'ordId': order_id}
            result = await self._send_private_request('GET', endpoint, params)
            
            if result['success'] and result['data']:
                order_data = result['data'][0]
                return {
                    'success': True,
                    'status': order_data['state'],
                    'filled_size': float(order_data['fillSz']),
                    'avg_price': float(order_data['avgPx']) if order_data['avgPx'] else 0,
                    'raw_response': result
                }
            else:
                return {
                    'success': False,
                    'reason': result.get('reason', 'Unknown error'),
                    'raw_response': result
                }
                
        except Exception as e:
            self.logger.error(f"Error getting order status: {str(e)}")
            return {'success': False, 'reason': f'Error: {str(e)}'}
    
    async def get_open_orders(self, symbol=None):
        """
        Get open orders
        """
        try:
            endpoint = '/api/v5/trade/orders-pending'
            params = {}
            if symbol:
                params['instId'] = symbol
            
            result = await self._send_private_request('GET', endpoint, params)
            
            if result['success'] and result['data']:
                return result['data']
            else:
                self.logger.error(f"Failed to get open orders: {result.get('reason', 'Unknown error')}")
                return []
                
        except Exception as e:
            self.logger.error(f"Error getting open orders: {str(e)}")
            return []
    
    async def get_funding_rate(self, symbol):
        """
        Get funding rate
        """
        try:
            endpoint = '/api/v5/public/funding-rate'
            params = {'instId': symbol}
            result = await self._send_public_request('GET', endpoint, params)
            
            if result['success'] and result['data']:
                return result['data']
            else:
                self.logger.error(f"Failed to get funding rate: {result.get('reason', 'Unknown error')}")
                return []
                
        except Exception as e:
            self.logger.error(f"Error getting funding rate: {str(e)}")
            return []
    
    async def _send_public_request(self, method, endpoint, params=None):
        """
        Send public API request
        """
        if not self.session:
            await self.initialize()
        
        url = self.base_url + endpoint
        
        try:
            async with self.session.request(method, url, params=params) as response:
                result = await response.json()
                
                if response.status == 200:
                    if result.get('code') == '0':
                        return {'success': True, 'data': result.get('data', [])}
                    else:
                        return {'success': False, 'reason': result.get('msg', 'Unknown API error')}
                else:
                    return {'success': False, 'reason': f'HTTP {response.status}: {result}'}
                    
        except Exception as e:
            self.logger.error(f"Public request error: {str(e)}")
            return {'success': False, 'reason': f'Request error: {str(e)}'}
    
    async def _send_private_request(self, method, endpoint, data=None):
        """
        Send private API request with authentication
        """
        if not self.session:
            await self.initialize()
        
        timestamp = str(int(time.time()))
        body = json.dumps(data) if data else ''
        
        # Create signature
        message = timestamp + method.upper() + endpoint + body
        signature = base64.b64encode(
            hmac.new(
                self.config.OKX_SECRET_KEY.encode('utf-8'),
                message.encode('utf-8'),
                hashlib.sha256
            ).digest()
        )
        
        headers = {
            'OK-ACCESS-KEY': self.config.OKX_API_KEY,
            'OK-ACCESS-SIGN': signature.decode('utf-8'),
            'OK-ACCESS-TIMESTAMP': timestamp,
            'OK-ACCESS-PASSPHRASE': self.config.OKX_PASSPHRASE,
            'Content-Type': 'application/json'
        }
        
        url = self.base_url + endpoint
        
        try:
            async with self.session.request(
                method, 
                url, 
                headers=headers, 
                data=body
            ) as response:
                result = await response.json()
                
                if response.status == 200:
                    if result.get('code') == '0':
                        return {'success': True, 'data': result.get('data', [])}
                    else:
                        return {'success': False, 'reason': result.get('msg', 'Unknown API error')}
                else:
                    return {'success': False, 'reason': f'HTTP {response.status}: {result}'}
                    
        except Exception as e:
            self.logger.error(f"Private request error: {str(e)}")
            return {'success': False, 'reason': f'Request error: {str(e)}'}
    
    async def close_session(self):
        """
        Close aiohttp session
        """
        if self.session:
            await self.session.close()
            self.logger.info("OKX client session closed")
    
    async def get_server_time(self):
        """
        Get server time
        """
        try:
            endpoint = '/api/v5/public/time'
            result = await self._send_public_request('GET', endpoint, None)
            
            if result['success'] and result['data']:
                return result['data'][0]['ts']
            else:
                return None
                
        except Exception as e:
            self.logger.error(f"Error getting server time: {str(e)}")
            return None
    
    def format_symbol(self, base, quote):
        """
        Format trading symbol for OKX
        """
        return f"{base.upper()}-{quote.upper()}"
    
    def parse_kline_data(self, kline):
        """
        Parse kline data from OKX format
        """
        if not kline:
            return None
        
        return {
            'timestamp': int(kline[0]),
            'open': float(kline[1]),
            'high': float(kline[2]),
            'low': float(kline[3]),
            'close': float(kline[4]),
            'volume': float(kline[5]),
            'volume_currency': float(kline[6]) if len(kline) > 6 else 0,
            'volume_currency_quote': float(kline[7]) if len(kline) > 7 else 0
        }