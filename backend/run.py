"""
Run Script for Perspective API
===============================

Entry point for running the Flask application.
"""

import os
import sys

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# Load .env file from project root
from dotenv import load_dotenv
load_dotenv(os.path.join(project_root, '.env'))

from backend.app import create_app


def main():
    """Run the Flask development server."""
    # Get configuration from environment
    config_name = os.getenv('FLASK_ENV', 'development')
    port = int(os.getenv('PORT', 5000))
    host = os.getenv('HOST', '0.0.0.0')
    use_reloader = os.getenv('FLASK_USE_RELOADER', 'false').lower() == 'true'
    
    # Create application
    app = create_app(config_name)
    
    print(f"""
    ╔═══════════════════════════════════════════╗
    ║         PERSPECTIVE API SERVER            ║
    ╠═══════════════════════════════════════════╣
    ║  Running on: http://{host}:{port}          
    ║  Environment: {config_name}                
    ║  Debug: {app.debug}                        
    ╚═══════════════════════════════════════════╝
    """)
    
    # Run server
    app.run(
        host=host,
        port=port,
        debug=app.debug,
        use_reloader=use_reloader
    )


if __name__ == '__main__':
    main()
