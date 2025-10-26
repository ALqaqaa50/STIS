# STIS Sentiment Analysis Module
import asyncio
import aiohttp
import json
from datetime import datetime, timedelta
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import re
import logging
from typing import Dict, List, Optional

class SentimentAnalyzer:
    """
    Advanced sentiment analysis for crypto market news and social media
    """
    
    def __init__(self, config):
        self.config = config
        self.vader_analyzer = SentimentIntensityAnalyzer()
        self.logger = logging.getLogger('SentimentAnalyzer')
        
        # News sources and social media APIs
        self.news_sources = [
            'coindesk.com',
            'cointelegraph.com',
            'cryptonews.com',
            'bitcoin.com',
            'coinbase.com/blog'
        ]
        
        # Keywords for crypto sentiment analysis
        self.bullish_keywords = [
            'bullish', 'surge', 'rally', 'boom', 'breakthrough', 'adoption',
            'growth', 'positive', 'optimistic', 'moon', 'pump', 'accumulate',
            'hodl', 'diamond hands', 'buy the dip', 'to the moon'
        ]
        
        self.bearish_keywords = [
            'bearish', 'crash', 'dump', 'collapse', 'fud', 'panic sell',
            'negative', 'pessimistic', 'correction', 'recession', 'liquidation',
            'bear market', 'paper hands', 'fear', 'uncertainty'
        ]
        
        self.neutral_keywords = [
            'stable', 'sideways', 'consolidation', 'holding', 'waiting',
            'uncertain', 'mixed', 'neutral', 'steady', 'flat'
        ]
    
    async def analyze(self):
        """
        Perform comprehensive sentiment analysis
        """
        try:
            self.logger.info("Starting sentiment analysis...")
            
            # Gather sentiment from multiple sources
            news_sentiment = await self._analyze_news_sentiment()
            twitter_sentiment = await self._analyze_twitter_sentiment()
            reddit_sentiment = await self._analyze_reddit_sentiment()
            fear_greed = await self._get_fear_greed_index()
            
            # Combine all sentiment sources
            combined_sentiment = self._combine_sentiment_sources(
                news_sentiment,
                twitter_sentiment,
                reddit_sentiment,
                fear_greed
            )
            
            # Generate sentiment signal
            sentiment_signal = self._generate_sentiment_signal(combined_sentiment)
            
            self.logger.info(f"Sentiment analysis completed. Signal: {sentiment_signal['direction']}")
            
            return {
                'overall_sentiment': combined_sentiment['overall_score'],
                'sentiment_signal': sentiment_signal,
                'news_sentiment': news_sentiment,
                'twitter_sentiment': twitter_sentiment,
                'reddit_sentiment': reddit_sentiment,
                'fear_greed_index': fear_greed,
                'confidence': combined_sentiment['confidence'],
                'sources_analyzed': combined_sentiment['source_count']
            }
            
        except Exception as e:
            self.logger.error(f"Sentiment analysis error: {str(e)}")
            return {
                'overall_sentiment': 0.5,
                'sentiment_signal': {'direction': 'HOLD', 'strength': 0, 'confidence': 0},
                'confidence': 0,
                'error': str(e)
            }
    
    async def _analyze_news_sentiment(self):
        """
        Analyze sentiment from crypto news sources
        """
        try:
            news_articles = []
            total_sentiment = 0
            article_count = 0
            
            # Fetch recent news articles (simulated for now)
            # In production, this would use news APIs
            headlines = [
                "Bitcoin surges past $45,000 as institutional adoption accelerates",
                "Crypto market shows signs of recovery after recent correction",
                "Major companies announce Bitcoin integration plans",
                "Regulatory clarity boosts cryptocurrency confidence",
                "Bitcoin mining difficulty reaches new all-time high"
            ]
            
            for headline in headlines:
                # Analyze sentiment using multiple methods
                textblob_sentiment = self._analyze_with_textblob(headline)
                vader_sentiment = self._analyze_with_vader(headline)
                keyword_sentiment = self._analyze_keywords(headline)
                
                # Combine sentiment scores
                combined_score = (
                    textblob_sentiment * 0.3 +
                    vader_sentiment * 0.4 +
                    keyword_sentiment * 0.3
                )
                
                news_articles.append({
                    'headline': headline,
                    'sentiment': combined_score,
                    'timestamp': datetime.now()
                })
                
                total_sentiment += combined_score
                article_count += 1
            
            average_sentiment = total_sentiment / article_count if article_count > 0 else 0.5
            
            return {
                'average_sentiment': average_sentiment,
                'article_count': article_count,
                'articles': news_articles,
                'source': 'crypto_news'
            }
            
        except Exception as e:
            self.logger.error(f"News sentiment analysis error: {str(e)}")
            return {'average_sentiment': 0.5, 'article_count': 0, 'source': 'crypto_news'}
    
    async def _analyze_twitter_sentiment(self):
        """
        Analyze sentiment from Twitter (simulated)
        """
        try:
            tweets = []
            total_sentiment = 0
            tweet_count = 0
            
            # Simulated Twitter data
            sample_tweets = [
                "Just bought more BTC! 🚀 To the moon! 🌙 #Bitcoin #Crypto",
                "Bitcoin is looking strong today, expecting a breakout soon",
                "The crypto market is really heating up! Bullish on BTC",
                "HODLing strong through the dip, diamond hands 💎🙌",
                "Bitcoin adoption is growing faster than ever! 📈",
                "Feeling bullish about the upcoming Bitcoin halving",
                "The technical analysis points to a Bitcoin rally",
                "Bitcoin fundamentals are stronger than ever"
            ]
            
            for tweet in sample_tweets:
                # Clean tweet text
                cleaned_tweet = self._clean_text(tweet)
                
                # Analyze sentiment
                textblob_sentiment = self._analyze_with_textblob(cleaned_tweet)
                vader_sentiment = self._analyze_with_vader(cleaned_tweet)
                keyword_sentiment = self._analyze_keywords(cleaned_tweet)
                
                # Combine sentiment scores
                combined_score = (
                    textblob_sentiment * 0.3 +
                    vader_sentiment * 0.4 +
                    keyword_sentiment * 0.3
                )
                
                tweets.append({
                    'text': tweet,
                    'sentiment': combined_score,
                    'timestamp': datetime.now()
                })
                
                total_sentiment += combined_score
                tweet_count += 1
            
            average_sentiment = total_sentiment / tweet_count if tweet_count > 0 else 0.5
            
            return {
                'average_sentiment': average_sentiment,
                'tweet_count': tweet_count,
                'tweets': tweets,
                'source': 'twitter'
            }
            
        except Exception as e:
            self.logger.error(f"Twitter sentiment analysis error: {str(e)}")
            return {'average_sentiment': 0.5, 'tweet_count': 0, 'source': 'twitter'}
    
    async def _analyze_reddit_sentiment(self):
        """
        Analyze sentiment from Reddit crypto communities
        """
        try:
            reddit_posts = []
            total_sentiment = 0
            post_count = 0
            
            # Simulated Reddit data
            sample_posts = [
                "Bitcoin is showing incredible strength despite market uncertainty",
                "Technical indicators suggest a major Bitcoin breakout is imminent",
                "Institutional FOMO is real - Bitcoin accumulation continues",
                "The Bitcoin network has never been more secure and decentralized",
                "Long-term Bitcoin holders are increasing their positions",
                "Bitcoin's price action suggests a new bull market is beginning",
                "The hash rate is at all-time highs, showing network confidence",
                "Bitcoin is becoming the digital gold we always knew it would be"
            ]
            
            for post in sample_posts:
                # Clean post text
                cleaned_post = self._clean_text(post)
                
                # Analyze sentiment
                textblob_sentiment = self._analyze_with_textblob(cleaned_post)
                vader_sentiment = self._analyze_with_vader(cleaned_post)
                keyword_sentiment = self._analyze_keywords(cleaned_post)
                
                # Combine sentiment scores
                combined_score = (
                    textblob_sentiment * 0.3 +
                    vader_sentiment * 0.4 +
                    keyword_sentiment * 0.3
                )
                
                reddit_posts.append({
                    'text': post,
                    'sentiment': combined_score,
                    'timestamp': datetime.now()
                })
                
                total_sentiment += combined_score
                post_count += 1
            
            average_sentiment = total_sentiment / post_count if post_count > 0 else 0.5
            
            return {
                'average_sentiment': average_sentiment,
                'post_count': post_count,
                'posts': reddit_posts,
                'source': 'reddit'
            }
            
        except Exception as e:
            self.logger.error(f"Reddit sentiment analysis error: {str(e)}")
            return {'average_sentiment': 0.5, 'post_count': 0, 'source': 'reddit'}
    
    async def _get_fear_greed_index(self):
        """
        Get Fear & Greed Index (simulated)
        """
        try:
            # Simulated Fear & Greed Index
            # In production, this would fetch from alternative.me API
            fear_greed_value = 65  # Slightly greedy
            fear_greed_classification = "Greed"
            
            # Convert to sentiment score (0-1, where 0.5 is neutral)
            sentiment_score = fear_greed_value / 100
            
            return {
                'value': fear_greed_value,
                'classification': fear_greed_classification,
                'sentiment_score': sentiment_score,
                'source': 'fear_greed_index'
            }
            
        except Exception as e:
            self.logger.error(f"Fear & Greed Index error: {str(e)}")
            return {'value': 50, 'classification': 'Neutral', 'sentiment_score': 0.5}
    
    def _analyze_with_textblob(self, text):
        """
        Analyze sentiment using TextBlob
        """
        try:
            blob = TextBlob(text)
            polarity = blob.sentiment.polarity
            
            # Convert polarity from [-1, 1] to [0, 1]
            return (polarity + 1) / 2
            
        except Exception as e:
            self.logger.error(f"TextBlob analysis error: {str(e)}")
            return 0.5
    
    def _analyze_with_vader(self, text):
        """
        Analyze sentiment using VADER
        """
        try:
            scores = self.vader_analyzer.polarity_scores(text)
            compound_score = scores['compound']
            
            # Convert compound score from [-1, 1] to [0, 1]
            return (compound_score + 1) / 2
            
        except Exception as e:
            self.logger.error(f"VADER analysis error: {str(e)}")
            return 0.5
    
    def _analyze_keywords(self, text):
        """
        Analyze sentiment based on keyword presence
        """
        try:
            text_lower = text.lower()
            
            bullish_count = sum(1 for keyword in self.bullish_keywords if keyword in text_lower)
            bearish_count = sum(1 for keyword in self.bearish_keywords if keyword in text_lower)
            neutral_count = sum(1 for keyword in self.neutral_keywords if keyword in text_lower)
            
            total_keywords = bullish_count + bearish_count + neutral_count
            
            if total_keywords == 0:
                return 0.5
            
            # Calculate sentiment score
            bullish_weight = 1.0
            bearish_weight = 0.0
            neutral_weight = 0.5
            
            sentiment_score = (
                (bullish_count * bullish_weight) +
                (bearish_count * bearish_weight) +
                (neutral_count * neutral_weight)
            ) / total_keywords
            
            return sentiment_score
            
        except Exception as e:
            self.logger.error(f"Keyword analysis error: {str(e)}")
            return 0.5
    
    def _clean_text(self, text):
        """
        Clean text for analysis
        """
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove user mentions and hashtags
        text = re.sub(r'@\w+|#\w+', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def _combine_sentiment_sources(self, news_sentiment, twitter_sentiment, reddit_sentiment, fear_greed):
        """
        Combine sentiment from all sources
        """
        try:
            sources = []
            weights = []
            
            # Add news sentiment
            if news_sentiment.get('article_count', 0) > 0:
                sources.append(news_sentiment['average_sentiment'])
                weights.append(0.3)  # News has high weight
            
            # Add Twitter sentiment
            if twitter_sentiment.get('tweet_count', 0) > 0:
                sources.append(twitter_sentiment['average_sentiment'])
                weights.append(0.25)  # Twitter has medium-high weight
            
            # Add Reddit sentiment
            if reddit_sentiment.get('post_count', 0) > 0:
                sources.append(reddit_sentiment['average_sentiment'])
                weights.append(0.25)  # Reddit has medium-high weight
            
            # Add Fear & Greed Index
            if fear_greed.get('value') is not None:
                sources.append(fear_greed['sentiment_score'])
                weights.append(0.2)  # Fear & Greed has lower weight
            
            if not sources:
                return {'overall_score': 0.5, 'confidence': 0, 'source_count': 0}
            
            # Calculate weighted average
            weighted_sum = sum(s * w for s, w in zip(sources, weights))
            total_weight = sum(weights)
            overall_score = weighted_sum / total_weight
            
            # Calculate confidence based on number of sources
            source_count = len(sources)
            confidence = min(source_count / 4.0, 1.0)  # Max confidence with 4 sources
            
            return {
                'overall_score': overall_score,
                'confidence': confidence,
                'source_count': source_count,
                'weights': weights,
                'individual_scores': sources
            }
            
        except Exception as e:
            self.logger.error(f"Sentiment combination error: {str(e)}")
            return {'overall_score': 0.5, 'confidence': 0, 'source_count': 0}
    
    def _generate_sentiment_signal(self, combined_sentiment):
        """
        Generate trading signal from sentiment analysis
        """
        try:
            overall_score = combined_sentiment.get('overall_score', 0.5)
            confidence = combined_sentiment.get('confidence', 0)
            
            # Determine signal based on sentiment score
            if overall_score > 0.65:  # Strong bullish
                direction = 'BUY'
                strength = min((overall_score - 0.65) * 2, 1.0)
            elif overall_score < 0.35:  # Strong bearish
                direction = 'SELL'
                strength = min((0.35 - overall_score) * 2, 1.0)
            else:  # Neutral
                direction = 'HOLD'
                strength = 0
            
            # Adjust strength based on confidence
            adjusted_strength = strength * confidence
            
            return {
                'direction': direction,
                'strength': adjusted_strength,
                'confidence': confidence,
                'raw_score': overall_score
            }
            
        except Exception as e:
            self.logger.error(f"Sentiment signal generation error: {str(e)}")
            return {'direction': 'HOLD', 'strength': 0, 'confidence': 0}
    
    async def get_sentiment_trends(self, hours=24):
        """
        Get sentiment trends over time
        """
        try:
            # This would analyze historical sentiment data
            # For now, return simulated trend data
            trends = []
            current_time = datetime.now()
            
            for i in range(hours):
                timestamp = current_time - timedelta(hours=i)
                # Simulate sentiment with some randomness
                base_sentiment = 0.6
                noise = (hash(str(timestamp)) % 100 - 50) / 500  # -0.1 to 0.1
                sentiment_value = max(0, min(1, base_sentiment + noise))
                
                trends.append({
                    'timestamp': timestamp,
                    'sentiment': sentiment_value
                })
            
            return trends[::-1]  # Return in chronological order
            
        except Exception as e:
            self.logger.error(f"Sentiment trends error: {str(e)}")
            return []
    
    def get_sentiment_summary(self, sentiment_analysis):
        """
        Get a summary of sentiment analysis
        """
        try:
            if not sentiment_analysis:
                return "No sentiment data available"
            
            overall_sentiment = sentiment_analysis.get('overall_sentiment', 0.5)
            sentiment_signal = sentiment_analysis.get('sentiment_signal', {})
            sources_analyzed = sentiment_analysis.get('sources_analyzed', 0)
            
            direction = sentiment_signal.get('direction', 'HOLD')
            strength = sentiment_signal.get('strength', 0)
            confidence = sentiment_analysis.get('confidence', 0)
            
            # Create human-readable summary
            if direction == 'BUY':
                sentiment_desc = "bullish"
            elif direction == 'SELL':
                sentiment_desc = "bearish"
            else:
                sentiment_desc = "neutral"
            
            summary = (
                f"Market sentiment is {sentiment_desc} with {strength:.2f} strength "
                f"and {confidence:.2f} confidence. "
                f"Analyzed {sources_analyzed} sources. "
                f"Overall sentiment score: {overall_sentiment:.2f}"
            )
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Sentiment summary error: {str(e)}")
            return "Error generating sentiment summary"