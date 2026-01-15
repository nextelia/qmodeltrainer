"""
Training manager for QModel Trainer (Server-based version)

This module orchestrates training by communicating with the QGeoAI server.
It no longer performs training locally - all ML work is done server-side.

This is the bridge between the UI (trainer_dock.py) and the server.
"""

from pathlib import Path
from typing import Optional, Dict, Any

from qgis.PyQt.QtCore import QObject, QThread, pyqtSignal

from ..clients.qmodeltrainer_client import QModelTrainerClient, TrainingMonitor


class TrainingManager(QObject):
    """
    Manager for orchestrating training via the server.
    
    This class acts as a bridge between the UI (TrainerDockWidget)
    and the QGeoAI server.
    
    It handles:
    - Configuration validation (pre-flight checks)
    - Starting training on server
    - Monitoring progress via SSE
    - Stopping training
    - Cleanup
    
    Args:
        dock_widget: Reference to the TrainerDockWidget UI
        server_url: URL of the QGeoAI server
    """
    
    def __init__(self, dock_widget):
        """Initialize the training manager."""
        super().__init__()
        self.dock_widget = dock_widget
        
        # Create HTTP client (gets URL/token dynamically from QGeoAIClient)
        self.client = QModelTrainerClient()
        
        # Monitor thread
        self.monitor_thread = None
        self.monitor = None
        
        # Current job
        self.current_job_id = None
        
        # Connect UI signals
        self.dock_widget.training_requested.connect(self.start_training)
        self.dock_widget.training_cancelled.connect(self.stop_training)
    
    def start_training(self):
        """
        Start the training process on the server.
        
        This method:
        1. Validates configuration locally (pre-flight)
        2. Sends training request to server
        3. Starts monitoring progress via SSE
        """
        # =================================================================
        # STEP 1: GET AND VALIDATE CONFIGURATION
        # =================================================================
        config = self.dock_widget.get_training_config()
        
        # Get validated dataset info
        dataset_info = self.dock_widget.get_validated_dataset_info()
        
        if dataset_info is None:
            self.dock_widget.append_log("\n❌ No valid dataset loaded. Please validate dataset first.")
            self.dock_widget.training_finished()
            return
        
        # Validate task/dataset compatibility
        try:
            compatible, error_msg = self.dock_widget.validate_task_dataset_compatibility()
            
            if not compatible:
                self.dock_widget.append_log(f"\n❌ Task/Dataset incompatibility:\n{error_msg}\n")
                self.dock_widget.training_finished()
                return
        except Exception as e:
            self.dock_widget.append_log(f"\n⚠️ Could not validate task compatibility: {e}\n")
            # Continue anyway - server will validate
        
        # Server will perform complete validation
        self.dock_widget.append_log("\n📤 Sending configuration to server...\n")
        
        # Validate output directory exists
        output_dir = Path(config['output_dir'])
        if not output_dir.exists():
            try:
                output_dir.mkdir(parents=True, exist_ok=True)
                self.dock_widget.append_log(f"✅ Created output directory: {output_dir}\n")
            except Exception as e:
                self.dock_widget.append_log(f"❌ Cannot create output directory: {e}\n")
                self.dock_widget.training_finished()
                return
        
        # =================================================================
        # STEP 2: START TRAINING ON SERVER
        # =================================================================
        self.dock_widget.append_log("🚀 Starting training on server...\n")
        
        # Convert config to server format
        server_config = self._prepare_server_config(config)
        
        # Start training
        success, error, job_id = self.client.start_training(server_config)
        
        if not success:
            self.dock_widget.append_log(f"\n❌ Failed to start training:\n{error}\n")
            self.dock_widget.training_finished()
            return
        
        self.current_job_id = job_id
        self.dock_widget.append_log(f"✅ Training started (Job ID: {job_id})\n")
        
        # =================================================================
        # STEP 3: START MONITORING PROGRESS
        # =================================================================
        self._start_monitor(job_id)
    
    def stop_training(self):
        """Stop the training process on the server."""
        if not self.current_job_id:
            self.dock_widget.append_log("⚠️  No training in progress")
            return
        
        self.dock_widget.append_log("\n⏸️  Stopping training...")
        
        success, error = self.client.stop_training(self.current_job_id)
        
        if not success:
            self.dock_widget.append_log(f"❌ Failed to stop training: {error}")
        else:
            self.dock_widget.append_log("✅ Stop signal sent to server")
    
    def _prepare_server_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert plugin config to server API format.
        
        This handles the mapping between plugin keys and server API keys,
        including YOLO vs PyTorch specific configurations.
        
        Args:
            config: Plugin configuration from get_training_config()
        
        Returns:
            Server-compatible configuration (TrainingRequest or YOLOTrainingRequest)
        """
        task = config['task']
        is_yolo = task in ['Detection (YOLO)', 'Instance Segmentation (YOLO)']
        
        
        # =========================================================================
        # EXTRACT AUGMENTATION CONFIG
        # =========================================================================
        aug_config = None
        if config.get('augmentation_config', {}).get('enabled'):
            aug_config = {
                'enabled': True,
                # Geometric
                'hflip': config['augmentation_config'].get('hflip', 0),
                'vflip': config['augmentation_config'].get('vflip', 0),
                'rotate90': config['augmentation_config'].get('rotate90', 0),
                # Radiometric
                'brightness': config['augmentation_config'].get('brightness', 0),
                'contrast': config['augmentation_config'].get('contrast', 0),
                'hue': config['augmentation_config'].get('hue', 0),
                'saturation': config['augmentation_config'].get('saturation', 0),
                'blur': config['augmentation_config'].get('blur', 0),
                'noise': config['augmentation_config'].get('noise', 0)
            }
        
        # =========================================================================
        # BUILD BASE CONFIG (common to all tasks)
        # =========================================================================
        base_config = {
            # Dataset
            'dataset_path': config['dataset_path'],
            'task': config['task'],
            
            # Device (FIX: handle None for CPU)
            'device': config['device'].lower(),
            'cuda_device_id': config.get('cuda_device_id') if config.get('cuda_device_id') is not None else 0,
            
            # Augmentation (COMPLETE)
            'augmentation_config': aug_config,
            
            # Advanced options
            'resume_training': config.get('resume_training', False),
            'checkpoint_path': config.get('checkpoint_path'),
            'save_best': config.get('save_best', True),
            'generate_report': True,
            
            # Output
            'output_dir': config['output_dir'],
            'model_name': config.get('model_name')  
        }
        
        # =========================================================================
        # ADD TASK-SPECIFIC CONFIGURATION
        # =========================================================================
        if is_yolo:
            # YOLO CONFIGURATION
            yolo_config = {
                **base_config,
                'model_size': config['model_size'].lower() if config.get('model_size') else 's',
                'pretrained': config['pretrained'],
                'epochs': config['epochs'],
                'batch_size': config['batch_size'],
                'image_size': config['image_size'],
                'optimizer': config.get('optimizer', 'auto'),
                'learning_rate': config.get('learning_rate')
            }
            
            # Handle OBB vs regular detection
            if config.get('yolo_model') == 'YOLO11-obb':
                yolo_config['task'] = 'obb'
            
            return yolo_config
            
        else:
            # PYTORCH CONFIGURATION 
            scheduler_value = config.get('scheduler')
            if scheduler_value in [None, 'None', '']:
                scheduler_value = None
            
            pytorch_config = {
                **base_config,
                
                # Model architecture
                'architecture': config['architecture'].lower(),
                'backbone': config['backbone'],
                'pretrained': config['pretrained'],
                'encoder_weights': config.get('encoder_weights'),  
                
                # Training parameters
                'epochs': config['epochs'],
                'batch_size': config['batch_size'],
                'image_size': config['image_size'],
                'val_split': config['val_split'],
                
                # Optimizer & Scheduler
                'optimizer': config['optimizer'],
                'learning_rate': config['learning_rate'],
                'weight_decay': config.get('weight_decay', 0.0),
                'scheduler': scheduler_value,
                
                # LR Finder
                'use_lr_finder': config.get('auto_lr', False)
            }
            
            return pytorch_config
    
    def _start_monitor(self, job_id: str):
        """
        Start monitoring training progress via SSE.
        
        Args:
            job_id: Job identifier to monitor
        """
        # Create monitor
        self.monitor = self.client.create_monitor(job_id)
        
        # Connect signals
        self.monitor.log_message.connect(self.dock_widget.append_log)
        self.monitor.progress_update.connect(self._update_progress)
        self.monitor.epoch_metrics.connect(self._on_epoch_metrics)
        self.monitor.training_finished.connect(self._on_training_finished)
        self.monitor.connection_error.connect(self._on_connection_error)
        
        # Create thread
        self.monitor_thread = QThread()
        self.monitor.moveToThread(self.monitor_thread)
        
        # Connect thread signals
        self.monitor_thread.started.connect(self.monitor.run)
        self.monitor_thread.finished.connect(self._cleanup)
        
        # Start monitoring
        self.dock_widget.append_log("📡 Connected to training stream\n")
        self.monitor_thread.start()
    
    def _update_progress(self, current: int, total: int):
        """
        Update progress bar.
        
        Args:
            current: Current epoch number
            total: Total number of epochs
        """
        percentage = int((current / total) * 100)
        self.dock_widget.progress_bar.setValue(percentage)
    
    def _on_epoch_metrics(self, metrics: Dict[str, Any]):
        """
        Handle epoch metrics.
        
        Args:
            metrics: Dictionary of metrics from the epoch
        """
        # Metrics are already logged by the server
        # Will be used for live plotting in future version
        pass
    
    def _on_training_finished(self, success: bool, message: str):
        """
        Handle training completion.
        
        Args:
            success: Whether training completed successfully
            message: Completion message
        """
        if success:
            self.dock_widget.append_log(f"\n🎉 {message}")
            
            # Get final results
            if self.current_job_id:
                results = self.client.get_results(self.current_job_id)
                if results:
                    self.dock_widget.append_log("\n📊 Training Results:")
                    self.dock_widget.append_log(f"   Checkpoint: {results.get('best_checkpoint_path')}")
                    if results.get('report_path'):
                        self.dock_widget.append_log(f"   Report: {results.get('report_path')}")
                    if results.get('total_time'):
                        self.dock_widget.append_log(f"   Total time: {results.get('total_time')}")
        else:
            self.dock_widget.append_log(f"\n❌ Training failed: {message}")
        
        # Cleanup
        self._cleanup()
    
    def _on_connection_error(self, error_msg: str):
        """
        Handle connection error.
        
        Args:
            error_msg: Error message
        """
        self.dock_widget.append_log(f"\n❌ Connection error:\n{error_msg}")
        self.dock_widget.append_log("\n⚠️  Check that the QGeoAI server is running")
        
        # Cleanup
        self._cleanup()
    
    def _cleanup(self):
        """Cleanup after training finishes or fails."""
        # Stop monitor
        if self.monitor:
            self.monitor.stop()
        
        # Stop thread
        if self.monitor_thread is not None:
            if self.monitor_thread.isRunning():
                self.monitor_thread.quit()
                self.monitor_thread.wait(5000) 
            self.monitor_thread = None
        
        self.monitor = None
        self.current_job_id = None
        
        # Reset UI state
        self.dock_widget.training_finished()
        self.dock_widget.progress_bar.setValue(0)