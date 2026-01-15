"""
HTTP Client for QModelTrainer server endpoints

This client handles all communication between the QGIS plugin and the server.
It ALWAYS uses the current server URL from QGeoAIClient (dynamic port).
"""

import json
import logging
from typing import Optional, Dict, Any
from pathlib import Path
import sys

import requests
from qgis.PyQt.QtCore import QObject, pyqtSignal

# Import QGeoAI client
qgeoai_path = str(Path.home() / '.qgeoai')
if qgeoai_path not in sys.path:
    sys.path.insert(0, qgeoai_path)

from qgeoai_client import QGeoAIClient

logger = logging.getLogger(__name__)


class TrainingMonitor(QObject):
    """
    Monitor training progress via SSE (Server-Sent Events).
    
    Runs in a separate QThread to avoid blocking the UI.
    """
    
    # Signals
    log_message = pyqtSignal(str)
    progress_update = pyqtSignal(int, int)  # (current_epoch, total_epochs)
    epoch_metrics = pyqtSignal(dict)
    training_finished = pyqtSignal(bool, str)  # (success, message)
    connection_error = pyqtSignal(str)
    
    def __init__(self, qgeoai_client: QGeoAIClient, job_id: str):
        """
        Initialize monitor.
        
        Args:
            qgeoai_client: QGeoAI client (for dynamic URL and headers)
            job_id: Job ID to monitor
        """
        super().__init__()
        self.qgeoai_client = qgeoai_client
        self.job_id = job_id
        self.should_stop = False
    
    def run(self):
        """Connect to SSE stream and emit signals for events."""
        # Get URL dynamically
        server_url = self.qgeoai_client.base_url
        url = f"{server_url}/qmodeltrainer/training/{self.job_id}/stream"
        
        try:
            with requests.get(url, headers=self.qgeoai_client.headers, stream=True, timeout=None) as response:
                response.raise_for_status()
                
                event_type = None
                for line in response.iter_lines():
                    if self.should_stop:
                        break
                    
                    if not line:
                        continue
                    
                    line = line.decode('utf-8')
                    
                    # Parse SSE format
                    if line.startswith('event:'):
                        event_type = line.split(':', 1)[1].strip()
                    elif line.startswith('data:'):
                        data_str = line.split(':', 1)[1].strip()
                        data = json.loads(data_str)
                        
                        # Handle different event types
                        if event_type == 'log':
                            self.log_message.emit(data['message'])
                        
                        elif event_type == 'status':
                            if 'current_epoch' in data and 'total_epochs' in data:
                                self.progress_update.emit(
                                    data['current_epoch'],
                                    data['total_epochs']
                                )
                        
                        elif event_type == 'metrics':
                            self.epoch_metrics.emit(data)
                        
                        elif event_type == 'complete':
                            success = data['status'] == 'completed'
                            message = data.get('message', 'Training finished')
                            self.training_finished.emit(success, message)
                            break
                        
                        elif event_type == 'error':
                            self.connection_error.emit(data.get('error', 'Unknown error'))
                            break
        
        except requests.exceptions.RequestException as e:
            self.connection_error.emit(f"Connection error: {str(e)}")
        except Exception as e:
            logger.error(f"Monitor error: {e}")
            self.connection_error.emit(f"Monitor error: {str(e)}")
    
    def stop(self):
        """Signal the monitor to stop."""
        self.should_stop = True


class QModelTrainerClient:
    """
    Client for QModelTrainer server API.
    
    This client uses QGeoAIClient for dynamic server URL discovery.
    It NEVER stores the server URL - it always gets it fresh from QGeoAIClient.
    """
    
    def __init__(self):
        """
        Initialize client.
        
        NO parameters - we get everything from QGeoAIClient dynamically.
        """
        # Create QGeoAI client for dynamic URL/token
        self.qgeoai_client = QGeoAIClient()
        self.session = requests.Session()
    
    @property
    def _server_url(self):
        """Get current server URL (dynamic)."""
        return self.qgeoai_client.base_url
    
    @property
    def _headers(self):
        """Get current headers with fresh token (dynamic)."""
        return self.qgeoai_client.headers
    
    def start_training(self, config: Dict[str, Any]) -> tuple[bool, Optional[str], Optional[str]]:
        """
        Start a training job.
        
        Args:
            config: Training configuration dictionary
        
        Returns:
            Tuple of (success, error_message, job_id)
        """
        # Get URL dynamically
        url = f"{self._server_url}/qmodeltrainer/training/start"
        
        # For diag
        print("=" * 80)
        print("DEBUG: QModelTrainerClient.start_training()")
        print("=" * 80)
        print(f"URL: {url}")
        print(f"Headers present: {bool(self._headers)}")
        print(f"Task: {config.get('task')}")
        print("=" * 80)
        
        try:
            response = self.session.post(
                url,
                json=config,
                headers=self._headers,  # Fresh headers
                timeout=30
            )
            
            print(f"Response Status: {response.status_code}")
            if response.status_code != 200:
                print(f"Response Text: {response.text}")
            print("=" * 80)
            
            response.raise_for_status()
            data = response.json()
            
            job_id = data.get('job_id')
            return True, None, job_id
        
        except requests.exceptions.HTTPError as e:
            print(f"❌ HTTPError: {e}")
            print(f"Status: {e.response.status_code}")
            print(f"URL: {e.response.url}")
            try:
                error_detail = e.response.json()
                print(f"Detail: {json.dumps(error_detail, indent=2)}")
                error_msg = error_detail.get('detail', str(e))
            except:
                print(f"Text: {e.response.text}")
                error_msg = e.response.text or str(e)
            
            return False, f"Server error ({e.response.status_code}): {error_msg}", None
            
        except Exception as e:
            print(f"❌ Exception: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            return False, f"Error: {str(e)}", None
    
    def stop_training(self, job_id: str) -> tuple[bool, Optional[str]]:
        """Stop a running training job."""
        url = f"{self._server_url}/qmodeltrainer/training/{job_id}/stop"
        
        try:
            response = self.session.post(url, headers=self._headers, timeout=10)
            response.raise_for_status()
            return True, None
        except requests.exceptions.RequestException as e:
            return False, f"Failed to stop: {str(e)}"
    
    def get_status(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get current status of a training job."""
        url = f"{self._server_url}/qmodeltrainer/training/{job_id}/status"
        
        try:
            response = self.session.get(url, headers=self._headers, timeout=10)
            response.raise_for_status()
            return response.json()
        except:
            return None
    
    def get_results(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get final results of a completed training job."""
        url = f"{self._server_url}/qmodeltrainer/training/{job_id}/results"
        
        try:
            response = self.session.get(url, headers=self._headers, timeout=10)
            response.raise_for_status()
            return response.json()
        except:
            return None
    
    def create_monitor(self, job_id: str) -> TrainingMonitor:
        """
        Create a training monitor for SSE streaming.
        
        Args:
            job_id: Job identifier to monitor
        
        Returns:
            TrainingMonitor instance (not yet started)
        """
        return TrainingMonitor(self.qgeoai_client, job_id)