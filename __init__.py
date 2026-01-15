"""
QModel Trainer - QGIS Plugin for GeoAI Model Training
"""

def classFactory(iface):
    """
    Load QModelTrainer class from file qmodel_trainer.
    
    This function is called by QGIS when the plugin is loaded.
    
    Args:
        iface: A QGIS interface instance
        
    Returns:
        QModelTrainer: Plugin instance
    """
    from .qmodel_trainer import QModelTrainer
    return QModelTrainer(iface)