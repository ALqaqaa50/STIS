# STIS Secure Trade Execution Engine
import asyncio
import aiohttp
import hashlib
import hmac
import base64
import json
from datetime import datetime, timezone
import logging
import time
from typing import Dict, Optional

class TradeExecutor:
    """
    Secure, high-performance trade execution system with dual-layer security
    """
    
    def __init__(self, config):
        self.config = config
        self.base_url = "https://www.okx.com" if not config.OKX_SANDBOX else "https://www.okx.com"
        self.session = None
        self.order_cache = {}
        self.execution_history = []
        
        # Security parameters
        self.max_retries = 3
        self.retry_delay = 1
        self.timeout = 10
        
        # Initialize logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger('TradeExecutor')
        
    async def initialize(self):
        """
        Initialize aiohttp session
        """
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=self.timeout),
            headers={'Content-Type': 'application/json'}
        )
    
    async def execute_order(self, validated_order, prediction=None):
        """
        Execute validated order with dual-layer security
        """
        try:
            # Security Layer 1: Pre-execution validation
            security_check = self._pre_execution_security_check(validated_order)
            if not security_check['valid']:
                self.logger.error(f"Pre-execution security check failed: {security_check['reason']}")
                return {'success': False, 'reason': security_check['reason']}
            
            # Security Layer 2: Order integrity verification
            integrity_check = self._verify_order_integrity(validated_order)
            if not integrity_check['valid']:
                self.logger.error(f"Order integrity check failed: {integrity_check['reason']}")
                return {'success': False, 'reason': integrity_check['reason']}
            
            # Execute order based on type
            if validated_order['type'] == 'market':
                result = await self._execute_market_order(validated_order)
            elif validated_order['type'] == 'limit':
                result = await self._execute_limit_order(validated_order)
            elif validated_order['type'] == 'stop':
                result = await self._execute_stop_order(validated_order)
            else:
                result = await self._execute_stop_limit_order(validated_order)
            
            # Post-execution verification
            if result['success']:
                post_check = await self._post_execution_verification(result)
                if not post_check['valid']:
                    self.logger.warning(f"Post-execution verification failed: {post_check['reason']}")
                    # Attempt to cancel or modify the order
                    await self._handle_execution_failure(result, post_check['reason'])
            
            # Log execution
            self._log_execution(validated_order, result, prediction)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Trade execution error: {str(e)}")
            return {'success': False, 'reason': f'Execution error: {str(e)}'}
    
    async def _execute_market_order(self, order):
        """
        Execute market order
        """
        endpoint = '/api/v5/trade/order'
        
        order_data = {
            'instId': order['symbol'],
            'tdMode': 'cross',  # Cross margin mode
            'side': order['side'].upper(),
            'ordType': 'market',
            'sz': str(order['amount'])
        }
        
        return await self._send_signed_request('POST', endpoint, order_data)
    
    async def _execute_limit_order(self, order):
        """
        Execute limit order
        """
        endpoint = '/api/v5/trade/order'
        
        order_data = {
            'instId': order['symbol'],
            'tdMode': 'cross',
            'side': order['side'].upper(),
            'ordType': 'limit',
            'sz': str(order['amount']),
            'px': str(order['price'])
        }
        
        # Add stop loss and take profit if specified
        if 'stop_loss' in order:
            order_data['tpTriggerPx'] = str(order['stop_loss'])
            order_data['tpOrdPx'] = str(order['stop_loss'] * 1.01)  # Slightly above stop loss
        
        if 'take_profit' in order:
            order_data['slTriggerPx'] = str(order['take_profit'])
            order_data['slOrdPx'] = str(order['take_profit'] * 0.99)  # Slightly below take profit
        
        return await self._send_signed_request('POST', endpoint, order_data)
    
    async def _execute_stop_order(self, order):
        """
        Execute stop order
        """
        endpoint = '/api/v5/trade/order'
        
        order_data = {
            'instId': order['symbol'],
            'tdMode': 'cross',
            'side': order['side'].upper(),
            'ordType': 'conditional',
            'sz': str(order['amount'],
            'triggerPx': str(order['stop_price']),
            'ordPx': str(order.get('price', 'market'))
        }
        
        return await self._send_signed_request('POST', endpoint, order_data)
    
    async def _execute_stop_limit_order(self, order):
        """
        Execute stop-limit order
        """
        endpoint = '/api/v5/trade/order'
        
        order_data = {
            'instId': order['symbol'],
            'tdMode': 'cross',
            'side': order['side'].upper(),
            'ordType': 'conditional',
            'sz': str(order['amount']),
            'triggerPx': str(order['stop_price']),
            'ordPx': str(order['price'])
        }
        
        return await self._send_signed_request('POST', endpoint, order_data)
    
    async def _send_signed_request(self, method, endpoint, data):
        """
        Send signed request to OKX API
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
                        return {
                            'success': True,
                            'order_id': result['data'][0]['ordId'],
                            'client_order_id': result['data'][0]['clOrdId'],
                            'raw_response': result
                        }
                    else:
                        return {
                            'success': False,
                            'reason': result.get('msg', 'Unknown error'),
                            'raw_response': result
                        }
                else:
                    return {
                        'success': False,
                        'reason': f'HTTP {response.status}: {result}',
                        'raw_response': result
                    }
                    
        except Exception as e:
            self.logger.error(f"API request error: {str(e)}")
            return {
                'success': False,
                'reason': f'API request error: {str(e)}'
            }
    
    def _pre_execution_security_check(self, order):
        """
        Security Layer 1: Pre-execution validation
        """
        result = {'valid': True, 'reason': ''}
        
        # Check for duplicate orders
        order_hash = self._generate_order_hash(order)
        if order_hash in self.order_cache:
            if time.time() - self.order_cache[order_hash] < 60:  # 1 minute cooldown
                result['valid'] = False
                result['reason'] = 'Duplicate order detected'
                return result
        
        # Check order size limits
        if order['amount'] > self.config.MAX_POSITION_SIZE:
            result['valid'] = False
            result['reason'] = f'Order size exceeds maximum: {order["amount"]} > {self.config.MAX_POSITION_SIZE}'
            return result
        
        # Check rate limiting
        recent_orders = [t for t in self.execution_history 
                        if time.time() - t['timestamp'] < 60]
        if len(recent_orders) > 10:  # Max 10 orders per minute
            result['valid'] = False
            result['reason'] = 'Rate limit exceeded'
            return result
        
        # Cache order hash
        self.order_cache[order_hash] = time.time()
        
        return result
    
    def _verify_order_integrity(self, order):
        """
        Security Layer 2: Order integrity verification
        """
        result = {'valid': True, 'reason': ''}
        
        # Verify required fields
        required_fields = ['type', 'symbol', 'side', 'amount']
        for field in required_fields:
            if field not in order:
                result['valid'] = False
                result['reason'] = f'Missing required field: {field}'
                return result
        
        # Verify field values
        if order['side'] not in ['buy', 'sell']:
            result['valid'] = False
            result['reason'] = f'Invalid side: {order["side"]}'
            return result
        
        if order['amount'] <= 0:
            result['valid'] = False
            result['reason'] = 'Invalid amount'
            return result
        
        # Verify price for limit orders
        if order['type'] in ['limit', 'stop_limit'] and 'price' not in order:
            result['valid'] = False
            result['reason'] = f'Missing price for {order["type"]} order'
            return result
        
        return result
    
    async def _post_execution_verification(self, execution_result):
        """
        Post-execution verification to ensure order was placed correctly
        """
        result = {'valid': True, 'reason': ''}
        
        try:
            # Get order status
            order_id = execution_result['order_id']
            status_result = await self._get_order_status(order_id)
            
            if not status_result['success']:
                result['valid'] = False
                result['reason'] = f'Cannot verify order status: {status_result["reason"]}'
                return result
            
            # Verify order is active or filled
            order_status = status_result['status']
            if order_status not in ['live', 'partially_filled', 'filled']:
                result['valid'] = False
                result['reason'] = f'Order in unexpected status: {order_status}'
                return result
            
        except Exception as e:
            result['valid'] = False
            result['reason'] = f'Verification error: {str(e)}'
        
        return result
    
    async def _get_order_status(self, order_id):
        """
        Get order status from exchange
        """
        endpoint = f'/api/v5/trade/order?ordId={order_id}'
        result = await self._send_signed_request('GET', endpoint, None)
        
        if result['success'] and result['raw_response']['data']:
            return {
                'success': True,
                'status': result['raw_response']['data'][0]['state'],
                'filled_size': float(result['raw_response']['data'][0]['fillSz']),
                'avg_price': float(result['raw_response']['data'][0]['avgPx'])
            }
        
        return result
    
    async def _handle_execution_failure(self, execution_result, reason):
        """
        Handle execution failures and attempt recovery
        """
        self.logger.warning(f"Handling execution failure: {reason}")
        
        try:
            # Attempt to cancel the order if it's still active
            order_id = execution_result['order_id']
            cancel_result = await self._cancel_order(order_id)
            
            if cancel_result['success']:
                self.logger.info(f"Successfully cancelled order {order_id}")
            else:
                self.logger.error(f"Failed to cancel order {order_id}: {cancel_result['reason']}")
                
        except Exception as e:
            self.logger.error(f"Error handling execution failure: {str(e)}")
    
    async def _cancel_order(self, order_id):
        """
        Cancel order
        """
        endpoint = '/api/v5/trade/cancel-order'
        data = {'ordId': order_id}
        return await self._send_signed_request('POST', endpoint, data)
    
    def _generate_order_hash(self, order):
        """
        Generate unique hash for order to detect duplicates
        """
        order_str = f"{order['symbol']}-{order['side']}-{order['amount']}-{order.get('price', 0)}"
        return hashlib.md5(order_str.encode()).hexdigest()
    
    def _log_execution(self, order, result, prediction):
        """
        Log execution details
        """
        log_entry = {
            'timestamp': datetime.now(timezone.utc),
            'order': order,
            'result': result,
            'prediction': prediction
        }
        
        self.execution_history.append(log_entry)
        
        # Keep only last 1000 executions
        if len(self.execution_history) > 1000:
            self.execution_history = self.execution_history[-1000:]
        
        if result['success']:
            self.logger.info(f"Order executed successfully: {result['order_id']}")
        else:
            self.logger.error(f"Order execution failed: {result['reason']}")
    
    async def get_account_balance(self):
        """
        Get account balance
        """
        endpoint = '/api/v5/account/balance'
        result = await self._send_signed_request('GET', endpoint, None)
        return result
    
    async def get_open_orders(self, symbol=None):
        """
        Get open orders
        """
        endpoint = '/api/v5/trade/orders-pending'
        if symbol:
            endpoint += f'?instId={symbol}'
        
        result = await self._send_signed_request('GET', endpoint, None)
        return result
    
    async def close_session(self):
        """
        Close aiohttp session
        """
        if self.session:
            await self.session.close()
    
    def get_execution_stats(self):
        """
        Get execution statistics
        """
        total_executions = len(self.execution_history)
        successful_executions = len([e for e in self.execution_history if e['result']['success']])
        
        return {
            'total_executions': total_executions,
            'successful_executions': successful_executions,
            'success_rate': successful_executions / total_executions if total_executions > 0 else 0,
            'cached_orders': len(self.order_cache)
        }