#!/usr/bin/env python3
"""
SuperNinja Trading Intelligence System - Launcher Script
"""

import os
import sys
import subprocess
import logging
from pathlib import Path

def setup_logging():
    """Setup logging for the launcher"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger('STIS-Launcher')

def check_dependencies():
    """Check if all required dependencies are installed"""
    logger = logging.getLogger('STIS-Launcher')
    
    try:
        import numpy
        import pandas
        import tensorflow
        import asyncio
        import aiohttp
        logger.info("✅ Core dependencies verified")
        return True
    except ImportError as e:
        logger.error(f"❌ Missing dependency: {e}")
        logger.info("Please install dependencies: pip install -r requirements.txt")
        return False

def check_config():
    """Check configuration files"""
    logger = logging.getLogger('STIS-Launcher')
    
    config_file = Path('config/settings.py')
    if not config_file.exists():
        logger.error("❌ Configuration file not found")
        return False
    
    logger.info("✅ Configuration file found")
    return True

def create_directories():
    """Create necessary directories"""
    logger = logging.getLogger('STIS-Launcher')
    
    directories = [
        'logs',
        'data/performance',
        'data/models',
        'data/backups'
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        logger.info(f"✅ Directory created/verified: {directory}")

def check_environment():
    """Check environment variables"""
    logger = logging.getLogger('STIS-Launcher')
    
    required_vars = [
        'OKX_API_KEY',
        'OKX_SECRET_KEY',
        'OKX_PASSPHRASE'
    ]
    
    optional_vars = [
        'TELEGRAM_BOT_TOKEN',
        'TELEGRAM_CHAT_ID'
    ]
    
    missing_required = []
    missing_optional = []
    
    for var in required_vars:
        if not os.getenv(var):
            missing_required.append(var)
    
    for var in optional_vars:
        if not os.getenv(var):
            missing_optional.append(var)
    
    if missing_required:
        logger.error("❌ Missing required environment variables:")
        for var in missing_required:
            logger.error(f"   - {var}")
        logger.info("Please set the required environment variables before running the system")
        return False
    
    if missing_optional:
        logger.warning("⚠️  Missing optional environment variables:")
        for var in missing_optional:
            logger.warning(f"   - {var}")
        logger.info("Some features may not be available without these variables")
    
    logger.info("✅ Environment variables checked")
    return True

def run_system():
    """Run the main trading system"""
    logger = logging.getLogger('STIS-Launcher')
    
    try:
        logger.info("🚀 Starting SuperNinja Trading Intelligence System...")
        
        # Run the main system
        subprocess.run([sys.executable, 'main.py'], check=True)
        
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ System exited with error code {e.returncode}")
        return False
    except KeyboardInterrupt:
        logger.info("🛑 System stopped by user")
        return True
    except Exception as e:
        logger.error(f"❌ Unexpected error: {e}")
        return False

def main():
    """Main launcher function"""
    print("""
    ╔═══════════════════════════════════════════════════════════════╗
    ║          SuperNinja Trading Intelligence System              ║
    ║                    🤖 AI-Powered Trading Bot                   ║
    ╚═══════════════════════════════════════════════════════════════╝
    """)
    
    logger = setup_logging()
    
    logger.info("🔍 Initializing system checks...")
    
    # Run all checks
    checks = [
        ("Dependencies", check_dependencies),
        ("Configuration", check_config),
        ("Environment", check_environment)
    ]
    
    all_passed = True
    for check_name, check_func in checks:
        logger.info(f"Checking {check_name}...")
        if not check_func():
            all_passed = False
            break
    
    if not all_passed:
        logger.error("❌ System checks failed. Please resolve the issues above.")
        sys.exit(1)
    
    # Create necessary directories
    logger.info("📁 Creating necessary directories...")
    create_directories()
    
    # Run the system
    success = run_system()
    
    if success:
        logger.info("✅ System shutdown completed")
    else:
        logger.error("❌ System shutdown with errors")
        sys.exit(1)

if __name__ == "__main__":
    main()