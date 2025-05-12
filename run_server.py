import os
import argparse
import subprocess
import logging
import signal
import sys


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Start MLflow tracking server')
    
    parser.add_argument('--host', type=str, default='127.0.0.1',
                       help='The network address to listen on')
    parser.add_argument('--port', type=int, default=5000,
                       help='The port to listen on')
    parser.add_argument('--backend-store-uri', type=str, default='sqlite:///mlflow.db',
                       help='The SQLAlchemy database URI for the backend store')
    parser.add_argument('--default-artifact-root', type=str, default='mlruns',
                       help='Local or S3 URI to store artifacts')
    
    
    return parser.parse_args()

def start_mlflow_server(args):
    """
    Start MLflow tracking server with specified configuration.
    
    Args:
        args: Command line arguments
    """
    # Create directories if they don't exist
    os.makedirs(args.default_artifact_root, exist_ok=True)
    
    # Construct command
    cmd = [
        "python", "-m", "mlflow", "server",
        "--host", args.host,
        "--port", str(args.port),
        "--backend-store-uri", args.backend_store_uri,
        "--default-artifact-root", args.default_artifact_root
    ]
    
    logger.info(f"Starting MLflow server with command: {' '.join(cmd)}")
    
    # Start server
    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1
        )
        
        # Set up signal handler for graceful shutdown
        def signal_handler(sig, frame):
            logger.info("Shutting down MLflow server...")
            process.terminate()
            process.wait(timeout=5)
            logger.info("MLflow server shutdown complete.")
            sys.exit(0)
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        # Wait for server to start
        server_url = f"http://{args.host}:{args.port}"
        logger.info(f"Server starting up. MLflow UI will be available at {server_url}")
        
        # Monitor server output
        while True:
            line = process.stdout.readline()
            if not line:
                break
            logger.info(line.strip())
        
        # Check if process is still running
        if process.poll() is not None:
            logger.error(f"MLflow server exited with code {process.returncode}")
            return process.returncode
        
        # Keep program running
        process.wait()
        
    except KeyboardInterrupt:
        logger.info("Interrupted by user. Shutting down...")
        process.terminate()
        process.wait(timeout=5)
    except Exception as e:
        logger.error(f"Error starting MLflow server: {e}")
        return 1
    
    return 0

def main():
    """Main function to start MLflow server."""
    args = parse_args()
    return start_mlflow_server(args)

if __name__ == "__main__":
    sys.exit(main()) 