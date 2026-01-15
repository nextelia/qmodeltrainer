"""
QModel Trainer - QGIS Plugin for GeoAI Model Training

This plugin allows users to train deep learning models directly from QGIS
using datasets exported from QAnnotate.

Main Plugin Class

Main features:
- Semantic segmentation (SMP)
- Instance segmentation (Mask R-CNN)
- YOLO 11 (Detection, Segmentation, OBB)
- Real-time progress monitoring
- Multiple architectures and backbones

Requires: QGeoAI Server running locally
Copyright (C) 2026 Nextelia®
Licensed under GPL v2+ - see LICENSE file
"""

from qgis.PyQt.QtGui import QIcon
from qgis.PyQt.QtWidgets import QAction, QMessageBox, QToolBar
from qgis.PyQt.QtCore import Qt
from pathlib import Path

# Import QGeoAI client for server management
import sys
qgeoai_path = str(Path.home() / '.qgeoai')
if qgeoai_path not in sys.path:
    sys.path.insert(0, qgeoai_path)

from qgeoai_client import QGeoAIClient

class QModelTrainer:
    """
    Main QGIS Plugin Implementation.
    
    This class handles plugin lifecycle:
    - Initialization
    - GUI setup (toolbar, menu, dock widget)
    - Training manager creation
    - Cleanup on unload
    """

    def __init__(self, iface):
        """
        Initialize the plugin.
        
        Args:
            iface: QGIS interface instance
        """
        self.iface = iface
        self.plugin_dir = Path(__file__).parent
        self.dock_widget = None
        self.training_manager = None
        self.toolbar_action = None
        
        # Plugin metadata
        self.plugin_name = "QModel Trainer"
        self.plugin_version = "0.9.0"
        
        # QGeoAI Server client
        self.client = QGeoAIClient()

    def initGui(self):
        """
        Create the menu entries and toolbar icons inside the QGIS GUI.
        
        This is called by QGIS when the plugin is loaded.
        """

        from qgis.core import QgsMessageLog, Qgis
        
        # Start QGeoAI Server if not running (non-blocking)
        if not self.client.is_server_running():
            self.client.start_server(wait=False)
        
        # Get icon path
        icon_path = str(self.plugin_dir / 'assets' / 'images' / 'icon.png')
        
        # Check if icon exists, use default if not
        if not Path(icon_path).exists():
            icon = QIcon()  # Empty icon
        else:
            icon = QIcon(icon_path)
        
        # Create action for toolbar
        self.toolbar_action = QAction(
            icon,
            "QModel Trainer",
            self.iface.mainWindow()
        )
        self.toolbar_action.setObjectName("QModelTrainerAction")
        self.toolbar_action.setWhatsThis("Train deep learning models for geospatial data")
        self.toolbar_action.setStatusTip("Open QModel Trainer")
        self.toolbar_action.triggered.connect(self.show_dock)
        self.toolbar_action.setCheckable(True)
        
        # Add to QGeoAI submenu
        self.iface.addPluginToMenu("&QGeoAI", self.toolbar_action)

        # Create or get shared QGeoAI toolbar
        self.toolbar = self.iface.mainWindow().findChild(QToolBar, 'QGeoAIToolbar')
        if not self.toolbar:
            self.toolbar = self.iface.addToolBar('QGeoAI')
            self.toolbar.setObjectName('QGeoAIToolbar')
        
        # Add to shared toolbar
        self.toolbar.addAction(self.toolbar_action)

    def unload(self):
        """
        Remove the plugin menu item and icon from QGIS GUI.
        
        This is called by QGIS when the plugin is unloaded.
        Also performs cleanup of resources.
        """
        # Cleanup server connection
        if hasattr(self, 'client') and self.client and self.client.is_server_running():
            try:
                import requests
                requests.post(
                    f"{self.client.base_url}/shutdown",
                    headers=self.client.headers,
                    timeout=5
                )
            except:
                pass  # Not critical if it fails
        
        # Stop any ongoing training FIRST
        if self.training_manager:
            if (self.training_manager.thread and 
                self.training_manager.thread.isRunning()):
                # Force stop
                self.training_manager.stop_training()
                # Wait for thread to finish (max 5 seconds)
                if self.training_manager.thread:
                    self.training_manager.thread.quit()
                    self.training_manager.thread.wait(5000)  # 5 seconds timeout
            
            # Clean up manager
            self.training_manager = None
        
        # Remove dock widget if it exists
        if self.dock_widget:
            # Cleanup before deletion
            if hasattr(self.dock_widget, 'cleanup'):
                self.dock_widget.cleanup()
            
            # Disconnect all signals to avoid crashes
            try:
                self.dock_widget.training_requested.disconnect()
                self.dock_widget.training_cancelled.disconnect()
            except:
                pass
            
            self.iface.removeDockWidget(self.dock_widget)
            self.dock_widget.deleteLater()
            self.dock_widget = None
        
        # Clear CUDA cache if available
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except:
            pass
        
        # Remove from QGeoAI menu
        self.iface.removePluginMenu("&QGeoAI", self.toolbar_action)
        
        # Remove from toolbar
        if self.toolbar:
            self.toolbar.removeAction(self.toolbar_action)
            
            # Remove toolbar if empty
            if len(self.toolbar.actions()) == 0:
                self.iface.mainWindow().removeToolBar(self.toolbar)
                self.toolbar.deleteLater()
                self.toolbar = None
        
        # Delete action
        if self.toolbar_action:
            self.toolbar_action.deleteLater()
            self.toolbar_action = None

    def show_dock(self):
        """
        Show or hide the dock widget.
        
        Creates the dock widget and training manager on first call.
        """
        # Create dock widget on first call
        if self.dock_widget is None:
            try:
                from .ui.trainer_dock import TrainerDockWidget
                from .managers.training_manager import TrainingManager
                
                # Create dock widget
                self.dock_widget = TrainerDockWidget(self.iface.mainWindow())
                self.iface.addDockWidget(Qt.RightDockWidgetArea, self.dock_widget)
                
                # Connect visibility changed signal
                self.dock_widget.visibilityChanged.connect(self.on_dock_visibility_changed)
                
                # Create training manager
                # Le client QModelTrainer obtient URL/token dynamiquement de QGeoAIClient
                self.training_manager = TrainingManager(self.dock_widget)
                
                # Log welcome message
                self.dock_widget.append_log(f"{self.plugin_name} v{self.plugin_version}")
                self.dock_widget.append_log("Configure your training and click 'Start Training'.\n")
                
            except Exception as e:
                QMessageBox.critical(
                    self.iface.mainWindow(),
                    "Error",
                    f"Failed to create QModel Trainer interface:\n{str(e)}"
                )
                return
        
        # Toggle visibility
        if self.dock_widget.isVisible():
            self.dock_widget.hide()
        else:
            self.dock_widget.show()
    
    def on_dock_visibility_changed(self, visible):
        """
        Update toolbar button when dock visibility changes.
        
        Args:
            visible: Whether the dock is visible
        """
        if self.toolbar_action:
            self.toolbar_action.setChecked(visible)
    
    def _show_welcome_message(self):
        """
        Show a welcome message when plugin is first loaded.
        
        Can be removed if not desired.
        """
        QMessageBox.information(
            self.iface.mainWindow(),
            "QModel Trainer",
            "QModel Trainer has been loaded successfully!\n\n"
            "Click the toolbar icon to open the training interface.\n\n"
            "Features:\n"
            "• Train semantic segmentation models\n"
            "• Support for QAnnotate datasets\n"
            "• Multiple architectures (UNet, DeepLabV3, etc.)\n"
            "• Real-time training progress\n"
            "• GPU acceleration support"
        )