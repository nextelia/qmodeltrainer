"""
Dock widget UI for QModel Trainer

This module provides the user interface for configuring and monitoring
deep learning model training within QGIS. It handles user inputs for
dataset selection, model architecture, training hyperparameters, and
displays training progress and logs.
"""

from qgis.PyQt.QtWidgets import (
    QDockWidget, QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QLabel, QLineEdit, QComboBox, QCheckBox, QSpinBox, QDoubleSpinBox,
    QFileDialog, QTextEdit, QProgressBar, QGroupBox, QFormLayout,
    QScrollArea, QGridLayout
)
from qgis.PyQt.QtCore import Qt, pyqtSignal
from qgis.PyQt.QtGui import QFont


class TrainerDockWidget(QDockWidget):
    """
    Main training dock widget for QModel Trainer.
    
    This widget provides a comprehensive interface for:
    - Dataset selection and validation
    - Model configuration (Task, Architecture, Backbone, Pretrained)
    - Training hyperparameters (epochs, batch size, learning rate, etc.)
    - Device selection (CPU/CUDA)
    - Training monitoring (progress bar and logs)
    
    Signals:
        training_requested: Emitted when user clicks Start button
        training_cancelled: Emitted when user clicks Stop button
    """
    
    # Signals for communication with TrainingManager
    training_requested = pyqtSignal()
    training_cancelled = pyqtSignal()

    # Model configurations
    # These are only used for UI population
    
    SMP_ARCHITECTURES = {
        'unet': 'UNet',
        'unet++': 'UNet++', 
        'manet': 'MA-Net',
        'linknet': 'LinkNet',
        'fpn': 'FPN',
        'pspnet': 'PSPNet',
        'pan': 'PAN',
        'deeplabv3': 'DeepLabV3',
        'deeplabv3+': 'DeepLabV3+'
    }
    
    SMP_ENCODER_FAMILIES = {
        'ResNet': {
            'encoders': ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152']
        },
        'ResNeXt': {
            'encoders': ['resnext50_32x4d', 'resnext101_32x8d']
        },
        'EfficientNet': {
            'encoders': ['efficientnet-b0', 'efficientnet-b1', 'efficientnet-b2', 'efficientnet-b3', 
                        'efficientnet-b4', 'efficientnet-b5', 'efficientnet-b6', 'efficientnet-b7']
        },
        'MobileNet': {
            'encoders': ['mobilenet_v2']
        },
        'DenseNet': {
            'encoders': ['densenet121', 'densenet169', 'densenet201']
        },
        'VGG': {
            'encoders': ['vgg11', 'vgg13', 'vgg16', 'vgg19']
        }
    }
    
    ARCHITECTURE_ENCODER_CONSTRAINTS = {}
    
    MASKRCNN_BACKBONES = [
        'resnet50_fpn',
        'resnet101_fpn',
        'mobilenet_v3_large_fpn',
        'mobilenet_v3_large_320_fpn'
    ]
    
    ENCODER_PRETRAINED_WEIGHTS = {
        'resnet50_fpn': ['COCO_V1'],
        'resnet101_fpn': ['COCO_V1'],
        'mobilenet_v3_large_fpn': ['COCO_V1'],
        'mobilenet_v3_large_320_fpn': ['COCO_V1']
    }
    
    # =========================================================================
    # MODEL CONFIGURATIONS (dynamically loaded from model_factory)
    # =========================================================================
    
    # PyTorch tasks (architectures loaded from SMP_ARCHITECTURES)
    @property
    def PYTORCH_TASKS(self):
        """Dynamically build PyTorch tasks from model_factory."""
        return {
            'Semantic Segmentation': list(self.SMP_ARCHITECTURES.keys()),
            'Instance Segmentation': ['Mask R-CNN']
        }
    
    # YOLO models (static configuration)
    YOLO_TASKS = {
        'Instance Segmentation (YOLO)': ['YOLO11-seg'],
        'Detection (YOLO)': ['YOLO11', 'YOLO11-obb']
    }
    
    # YOLO model sizes
    YOLO_SIZES = ['n', 's', 'm', 'l', 'x']
    
    # YOLO optimizer options
    YOLO_OPTIMIZERS = ['Auto', 'SGD', 'AdamW']
    
    # All tasks combined (dynamic property)
    @property
    def TASKS(self):
        """Combine PyTorch and YOLO tasks."""
        return {**self.PYTORCH_TASKS, **self.YOLO_TASKS}
    
    def __init__(self, parent=None):
        """
        Initialize the dock widget.
        
        Args:
            parent: Parent widget (typically QGIS main window)
        """
        super().__init__("QModel Trainer", parent)
        self.setObjectName("QModelTrainerDock") 
        self._dataset_validated = False
        self._dataset_info = None
        self.init_ui()
        
    def init_ui(self):
        """
        Initialize the user interface.
        
        Creates all UI components organized in collapsible groups:
        1. Dataset selection
        2. Model configuration
        3. Training hyperparameters
        4. Device selection
        5. Output directory
        6. Advanced options
        7. Progress monitoring
        """
        # Main container with scroll area
        main_widget = QWidget()
        main_layout = QVBoxLayout()  
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        
        # ===== Header =====
        header_widget = self._create_header()
        main_layout.addWidget(header_widget)
        
        # ===== Content Area =====
        # Scrollable content area
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setFrameShape(QScrollArea.NoFrame)
        scroll_area.setStyleSheet("QScrollArea { border: none; }")
        
        # Content widget (inside scroll area)
        content_widget = QWidget()
        layout = QVBoxLayout() 
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)
        
        # ===================================================================
        # DATASET CONFIGURATION
        # ===================================================================
        dataset_group = QGroupBox("Dataset")
        dataset_layout = QFormLayout()

        # Dataset path with browse button
        self.dataset_path_edit = QLineEdit()
        self.dataset_path_edit.setPlaceholderText("Select QAnnotate export folder...")
        self.dataset_path_edit.textChanged.connect(self.auto_validate_dataset)  
        dataset_btn = QPushButton("📁")
        dataset_btn.setMaximumWidth(40)
        dataset_btn.setToolTip("Browse for dataset folder")
        dataset_btn.clicked.connect(self.browse_dataset)

        # PAS DE BOUTON VALIDATE
        dataset_row = QHBoxLayout()
        dataset_row.addWidget(self.dataset_path_edit)
        dataset_row.addWidget(dataset_btn)
        dataset_layout.addRow("Path:", dataset_row)

        # Dataset info label
        self.dataset_info_label = QLabel("No dataset loaded")
        self.dataset_info_label.setStyleSheet("color: gray; font-size: 10px;")
        self.dataset_info_label.setWordWrap(True)
        dataset_layout.addRow(self.dataset_info_label)

        dataset_group.setLayout(dataset_layout)
        layout.addWidget(dataset_group)
        
        # ===================================================================
        # MODEL CONFIGURATION
        # ===================================================================
        model_group = QGroupBox("Model")
        model_layout = QFormLayout()

        # Task selection
        self.task_combo = QComboBox()
        self.task_combo.addItems(list(self.TASKS.keys()))
        self.task_combo.currentTextChanged.connect(self.update_model_config_visibility)
        self.task_combo.currentTextChanged.connect(self.update_architectures)
        self.task_combo.currentTextChanged.connect(self.update_auto_lr_availability)
        self.task_combo.currentTextChanged.connect(self.update_default_lr)
        self.task_combo.currentTextChanged.connect(self.update_augmentation_availability)
        self.task_combo.currentTextChanged.connect(self.update_optimizer_scheduler_defaults)
        model_layout.addRow("Task:", self.task_combo)

        # === PyTorch Model Configuration (Architecture + Backbone) ===
        # Architecture selection (e.g., UNet) - for PyTorch only
        self.arch_label = QLabel("Architecture:")
        self.arch_combo = QComboBox()
        self.arch_combo.currentTextChanged.connect(self.update_backbones)
        model_layout.addRow(self.arch_label, self.arch_combo)

        # Backbone selection (e.g., resnet50) - for PyTorch only
        self.backbone_label = QLabel("Backbone:")
        self.backbone_combo = QComboBox()
        self.backbone_combo.currentTextChanged.connect(self.update_weights_combo)
        model_layout.addRow(self.backbone_label, self.backbone_combo)

        # Pretrained weights selection (new field)
        self.weights_label = QLabel("Weights:")
        self.weights_combo = QComboBox()
        self.weights_combo.setEnabled(False)  
        self.weights_combo.setToolTip(
            "Pretrained weights for the encoder.\n"
            "• imagenet: Standard ImageNet weights\n"
            "• ssl: Semi-supervised learning (better for limited data)\n"
            "• swsl: Weakly-supervised learning\n"
            "Available weights depend on the selected encoder."
        )
        model_layout.addRow(self.weights_label, self.weights_combo)

        # === YOLO Model Configuration (Model + Size) ===
        # Model selection (YOLO11, YOLO11-seg, YOLO11-obb) - for YOLO only
        self.yolo_model_label = QLabel("Model:")
        self.yolo_model_combo = QComboBox()
        model_layout.addRow(self.yolo_model_label, self.yolo_model_combo)

        # Size selection (n, s, m, l, x) - for YOLO only
        self.yolo_size_label = QLabel("Size:")
        self.yolo_size_combo = QComboBox()
        self.yolo_size_combo.addItems(self.YOLO_SIZES)
        self.yolo_size_combo.setCurrentText('n') 
        self.yolo_size_combo.setToolTip(
            "YOLO model size:\n"
            "• n (nano): Fastest, smallest\n"
            "• s (small): Good balance\n"
            "• m (medium): Default\n"
            "• l (large): Better accuracy\n"
            "• x (xlarge): Best accuracy, slowest"
        )
        model_layout.addRow(self.yolo_size_label, self.yolo_size_combo)

        # Pretrained weights checkbox
        self.pretrained_cb = QCheckBox("Use Pretrained Weights")
        self.pretrained_cb.setChecked(True)  # Default: YES (recommended)
        self.pretrained_cb.setToolTip(
            "Use pretrained weights for the model.\n"
            "• Semantic Segmentation: Encoder weights (ImageNet, SSL, etc.)\n"
            "• Instance Segmentation: COCO weights (full model)\n"
            "• YOLO: COCO weights (full model)\n"
            "Highly recommended for geospatial applications.\n"
            "Unchecking will train from random initialization."
        )
        self.pretrained_cb.stateChanged.connect(self.toggle_weights_combo)
        model_layout.addRow("", self.pretrained_cb)

        model_group.setLayout(model_layout)
        layout.addWidget(model_group)
        
        # ===================================================================
        # TRAINING CONFIGURATION
        # ===================================================================
        training_group = QGroupBox("Training")
        training_layout = QFormLayout()
        
        # Number of epochs
        self.epochs_spin = QSpinBox()
        self.epochs_spin.setRange(1, 1000)
        self.epochs_spin.setValue(50)
        self.epochs_spin.setToolTip("Number of training epochs")
        training_layout.addRow("Epochs:", self.epochs_spin)
        
        # Batch size
        self.batch_spin = QSpinBox()
        self.batch_spin.setRange(1, 32)
        self.batch_spin.setValue(4)
        self.batch_spin.setToolTip(
            "Number of samples per batch.\n"
            "Reduce if out of memory errors occur."
        )
        training_layout.addRow("Batch Size:", self.batch_spin)
        
        # Image size (constrained to multiples of 32 for most architectures)
        self.image_size_spin = QSpinBox()
        self.image_size_spin.setRange(128, 2048)
        self.image_size_spin.setSingleStep(32) 
        self.image_size_spin.setValue(512)
        self.image_size_spin.setToolTip(
            "Input image size (height and width).\n"
            "Must be a multiple of 32 for most architectures."
        )
        training_layout.addRow("Image Size:", self.image_size_spin)
        
        # Validation split (PyTorch only, YOLO uses pre-split dataset)
        self.val_split_label = QLabel("Val Split:")
        self.val_split_spin = QDoubleSpinBox()
        self.val_split_spin.setRange(0.0, 0.5)
        self.val_split_spin.setSingleStep(0.05)
        self.val_split_spin.setValue(0.2)
        self.val_split_spin.setToolTip(
            "Fraction of data used for validation (0.0 to 0.5).\n"
            "0.2 = 20% validation, 80% training"
        )
        training_layout.addRow(self.val_split_label, self.val_split_spin)
        
        # --- Learning Rate (moved here after Val Split) ---
        # Manual LR input
        self.lr_spin = QDoubleSpinBox()
        self.lr_spin.setDecimals(6)
        self.lr_spin.setRange(0.000001, 1.0)
        self.lr_spin.setSingleStep(0.0001)
        self.lr_spin.setValue(0.001)
        self.lr_spin.setToolTip("Manual learning rate (disabled if Auto is checked)")
        training_layout.addRow("Learning Rate:", self.lr_spin)
        
        # Auto LR checkbox (placed directly under LR field)
        self.auto_lr_cb = QCheckBox("Auto (LR Finder)")
        self.auto_lr_cb.setChecked(False)  
        self.auto_lr_cb.setToolTip(
            "Estimate learning rate automatically (LR Finder).\n"
            "⚠️ Note: Uses Adam as a calibration probe to determine the LR scale for your selected optimizer.\n"
            "Adds a short calibration step (≈60 s) before training.\n"
        )
        self.auto_lr_cb.stateChanged.connect(self.toggle_lr_auto)
        training_layout.addRow("", self.auto_lr_cb)

        # =========================================================================
        # OPTIMIZER & SCHEDULER (PyTorch only)
        # =========================================================================

        # Optimizer dropdown (PyTorch only)
        self.optimizer_label_pytorch = QLabel("Optimizer:")
        self.optimizer_combo_pytorch = QComboBox()
        self.optimizer_combo_pytorch.addItems(['Adam', 'AdamW', 'SGD'])
        self.optimizer_combo_pytorch.setCurrentText('Adam')  # Default
        self.optimizer_combo_pytorch.setToolTip(
            "Optimizer for training:\n"
            "• Adam: Adaptive learning, fast convergence\n"
            "• AdamW: Adam with better weight decay\n"
            "• SGD: Classic, stable, good for large datasets"
        )
        training_layout.addRow(self.optimizer_label_pytorch, self.optimizer_combo_pytorch)

        # Scheduler dropdown (PyTorch only)
        self.scheduler_label_pytorch = QLabel("Scheduler:")
        self.scheduler_combo_pytorch = QComboBox()
        self.scheduler_combo_pytorch.addItems([
            'ReduceLROnPlateau', 
            'StepLR', 
            'OneCycleLR', 
            'CosineAnnealingLR', 
            'None'
        ])
        self.scheduler_combo_pytorch.setCurrentText('ReduceLROnPlateau')  # Default
        self.scheduler_combo_pytorch.setToolTip(
            "Learning rate scheduler:\n"
            "• ReduceLROnPlateau: Reduce LR when stuck (recommended)\n"
            "• StepLR: Fixed decay schedule\n"
            "• OneCycleLR: Fast training (expert mode)\n"
            "• CosineAnnealingLR: Smooth decay\n"
            "• None: Fixed learning rate"
        )
        training_layout.addRow(self.scheduler_label_pytorch, self.scheduler_combo_pytorch)

        # Optimizer for YOLO tasks
        self.optimizer_label = QLabel("Optimizer:")
        self.optimizer_combo = QComboBox()
        self.optimizer_combo.addItems(self.YOLO_OPTIMIZERS)
        self.optimizer_combo.setCurrentText('Auto')
        self.optimizer_combo.setToolTip(
            "Optimizer for training:\n"
            "• Auto: Let Ultralytics choose (recommended)\n"
            "• SGD: Stochastic Gradient Descent with momentum\n"
            "• AdamW: Adam with weight decay"
        )
        training_layout.addRow(self.optimizer_label, self.optimizer_combo)

        training_group.setLayout(training_layout)
        layout.addWidget(training_group)
        
        # ===================================================================
        # DATA AUGMENTATION
        # ===================================================================
        augmentation_group = QGroupBox("Data Augmentation")
        augmentation_layout = QVBoxLayout()
        
        # Master checkbox
        self.augmentation_enabled_cb = QCheckBox("Enable Data Augmentation")
        self.augmentation_enabled_cb.setChecked(False)  
        self.augmentation_enabled_cb.setToolTip(
            "Enable data augmentation during training.\n"
            "Augmentations are applied randomly to increase dataset variability.\n"
            "Recommended for better model generalization, especially with small datasets."
        )
        self.augmentation_enabled_cb.stateChanged.connect(self.toggle_augmentation_options)
        augmentation_layout.addWidget(self.augmentation_enabled_cb)
        
        # Separator
        separator = QWidget()
        separator.setFixedHeight(1)
        separator.setStyleSheet("background-color: #cccccc;")
        augmentation_layout.addWidget(separator)
        
        # ===================================================================
        # GEOMETRIC AUGMENTATIONS
        # ===================================================================
        geo_label = QLabel("Geometric Augmentations:")
        geo_label_font = QFont()
        geo_label_font.setBold(True)
        geo_label.setFont(geo_label_font)
        augmentation_layout.addWidget(geo_label)
        
        # Horizontal Flip
        self.aug_hflip_cb = QCheckBox("Horizontal Flip")
        self.aug_hflip_cb.setChecked(False)
        self.aug_hflip_cb.setToolTip(
            "Randomly flip images horizontally (left ↔ right).\n"
            "Applied with 50% probability.\n"
            "Compatible: PyTorch + YOLO"
        )
        augmentation_layout.addWidget(self.aug_hflip_cb)
        
        # Vertical Flip
        self.aug_vflip_cb = QCheckBox("Vertical Flip")
        self.aug_vflip_cb.setChecked(False)
        self.aug_vflip_cb.setToolTip(
            "Randomly flip images vertically (top ↔ bottom).\n"
            "Applied with 50% probability.\n"
            "Compatible: PyTorch + YOLO"
        )
        augmentation_layout.addWidget(self.aug_vflip_cb)
        
        # Rotate 90
        self.aug_rotate90_cb = QCheckBox("Rotate 90°")
        self.aug_rotate90_cb.setChecked(False)
        self.aug_rotate90_cb.setToolTip(
            "Randomly rotate images by 0°, 90°, 180°, or 270°.\n"
            "Useful for overhead imagery with no fixed orientation.\n"
            "Compatible: PyTorch + YOLO"
        )
        augmentation_layout.addWidget(self.aug_rotate90_cb)
        
        # ===================================================================
        # RADIOMETRIC AUGMENTATIONS
        # ===================================================================
        radio_label = QLabel("Radiometric Augmentations:")
        radio_label_font = QFont()
        radio_label_font.setBold(True)
        radio_label.setFont(radio_label_font)
        augmentation_layout.addWidget(radio_label)
        
        # Create grid for better alignment
        radio_grid = QGridLayout()
        radio_grid.setColumnStretch(0, 1)  # Checkbox column expands
        radio_grid.setColumnStretch(1, 0)  # +/- label column
        radio_grid.setColumnStretch(2, 0)  # SpinBox column fixed width
        radio_grid.setColumnStretch(3, 0)  # [PyTorch] label column fixed width

        row = 0

        # --- Brightness ---
        self.aug_brightness_cb = QCheckBox("Brightness")
        self.aug_brightness_cb.setChecked(False)
        self.aug_brightness_cb.setToolTip(
            "Randomly adjust image brightness.\n"
            "Simulates different lighting conditions.\n"
            "Compatible: PyTorch + YOLO"
        )
        self.aug_brightness_cb.stateChanged.connect(self.toggle_brightness_spinbox) 
        radio_grid.addWidget(self.aug_brightness_cb, row, 0)

        # Label +/-
        brightness_label = QLabel("±")
        brightness_label.setStyleSheet("color: gray; font-weight: bold;")
        brightness_label.setAlignment(Qt.AlignCenter)
        radio_grid.addWidget(brightness_label, row, 1)

        self.aug_brightness_spin = QSpinBox()
        self.aug_brightness_spin.setRange(0, 100)
        self.aug_brightness_spin.setValue(0)
        self.aug_brightness_spin.setSuffix("%")
        self.aug_brightness_spin.setFixedWidth(80)
        self.aug_brightness_spin.setEnabled(False)  
        self.aug_brightness_spin.setToolTip("Brightness adjustment intensity (0-100%)")
        radio_grid.addWidget(self.aug_brightness_spin, row, 2)
        row += 1

        # --- Contrast [PyTorch] ---
        self.aug_contrast_cb = QCheckBox("Contrast")
        self.aug_contrast_cb.setChecked(False)
        self.aug_contrast_cb.setToolTip(
            "Randomly adjust image contrast.\n"
            "Enhances or reduces differences between light and dark areas.\n"
            "⚠️ PyTorch only (not available for YOLO)"
        )
        self.aug_contrast_cb.stateChanged.connect(self.toggle_contrast_spinbox) 
        radio_grid.addWidget(self.aug_contrast_cb, row, 0)

        contrast_label = QLabel("±")
        contrast_label.setStyleSheet("color: gray; font-weight: bold;")
        contrast_label.setAlignment(Qt.AlignCenter)
        radio_grid.addWidget(contrast_label, row, 1)

        self.aug_contrast_spin = QSpinBox()
        self.aug_contrast_spin.setRange(0, 100)
        self.aug_contrast_spin.setValue(0)
        self.aug_contrast_spin.setSuffix("%")
        self.aug_contrast_spin.setFixedWidth(80)
        self.aug_contrast_spin.setEnabled(False) 
        self.aug_contrast_spin.setToolTip("Contrast adjustment intensity (0-100%)")
        radio_grid.addWidget(self.aug_contrast_spin, row, 2)

        self.aug_contrast_pytorch_label = QLabel("[PyTorch]")
        self.aug_contrast_pytorch_label.setStyleSheet("color: gray; font-size: 9px;")
        radio_grid.addWidget(self.aug_contrast_pytorch_label, row, 3)
        row += 1

        # --- Hue [PyTorch] ---
        self.aug_hue_cb = QCheckBox("Hue Shift")
        self.aug_hue_cb.setChecked(False)
        self.aug_hue_cb.setToolTip(
            "Randomly shift image hue (color tint).\n"
            "Useful for RGB imagery with color variations.\n"
            "⚠️ PyTorch only (not available for YOLO)"
        )
        self.aug_hue_cb.stateChanged.connect(self.toggle_hue_spinbox) 
        radio_grid.addWidget(self.aug_hue_cb, row, 0)

        hue_label = QLabel("±")
        hue_label.setStyleSheet("color: gray; font-weight: bold;")
        hue_label.setAlignment(Qt.AlignCenter)
        radio_grid.addWidget(hue_label, row, 1)

        self.aug_hue_spin = QSpinBox()
        self.aug_hue_spin.setRange(0, 100)
        self.aug_hue_spin.setValue(0)
        self.aug_hue_spin.setSuffix("%")
        self.aug_hue_spin.setFixedWidth(80)
        self.aug_hue_spin.setEnabled(False)  
        self.aug_hue_spin.setToolTip("Hue shift intensity (0-100%)")
        radio_grid.addWidget(self.aug_hue_spin, row, 2)

        self.aug_hue_pytorch_label = QLabel("[PyTorch]")
        self.aug_hue_pytorch_label.setStyleSheet("color: gray; font-size: 9px;")
        radio_grid.addWidget(self.aug_hue_pytorch_label, row, 3)
        row += 1

        # --- Saturation [PyTorch] ---
        self.aug_saturation_cb = QCheckBox("Saturation")
        self.aug_saturation_cb.setChecked(False)
        self.aug_saturation_cb.setToolTip(
            "Randomly adjust color saturation (intensity).\n"
            "Makes colors more or less vivid.\n"
            "⚠️ PyTorch only (not available for YOLO)"
        )
        self.aug_saturation_cb.stateChanged.connect(self.toggle_saturation_spinbox)
        radio_grid.addWidget(self.aug_saturation_cb, row, 0)

        saturation_label = QLabel("±")
        saturation_label.setStyleSheet("color: gray; font-weight: bold;")
        saturation_label.setAlignment(Qt.AlignCenter)
        radio_grid.addWidget(saturation_label, row, 1)

        self.aug_saturation_spin = QSpinBox()
        self.aug_saturation_spin.setRange(0, 100)
        self.aug_saturation_spin.setValue(0)
        self.aug_saturation_spin.setSuffix("%")
        self.aug_saturation_spin.setFixedWidth(80)
        self.aug_saturation_spin.setEnabled(False) 
        self.aug_saturation_spin.setToolTip("Saturation adjustment intensity (0-100%)")
        radio_grid.addWidget(self.aug_saturation_spin, row, 2)

        self.aug_saturation_pytorch_label = QLabel("[PyTorch]")
        self.aug_saturation_pytorch_label.setStyleSheet("color: gray; font-size: 9px;")
        radio_grid.addWidget(self.aug_saturation_pytorch_label, row, 3)
        row += 1

        # --- Blur [PyTorch] ---
        self.aug_blur_cb = QCheckBox("Gaussian Blur")
        self.aug_blur_cb.setChecked(False)
        self.aug_blur_cb.setToolTip(
            "Randomly apply gaussian blur.\n"
            "Simulates different sensor resolutions or atmospheric effects.\n"
            "⚠️ PyTorch only (not available for YOLO)"
        )
        self.aug_blur_cb.stateChanged.connect(self.toggle_blur_spinbox)
        radio_grid.addWidget(self.aug_blur_cb, row, 0)

        blur_label = QLabel("±")
        blur_label.setStyleSheet("color: gray; font-weight: bold;")
        blur_label.setAlignment(Qt.AlignCenter)
        radio_grid.addWidget(blur_label, row, 1)

        self.aug_blur_spin = QSpinBox()
        self.aug_blur_spin.setRange(0, 100)
        self.aug_blur_spin.setValue(0)
        self.aug_blur_spin.setSuffix("%")
        self.aug_blur_spin.setFixedWidth(80)
        self.aug_blur_spin.setEnabled(False) 
        self.aug_blur_spin.setToolTip("Blur intensity (0-100%)")
        radio_grid.addWidget(self.aug_blur_spin, row, 2)

        self.aug_blur_pytorch_label = QLabel("[PyTorch]")
        self.aug_blur_pytorch_label.setStyleSheet("color: gray; font-size: 9px;")
        radio_grid.addWidget(self.aug_blur_pytorch_label, row, 3)
        row += 1

        # --- Noise [PyTorch] ---
        self.aug_noise_cb = QCheckBox("Gaussian Noise")
        self.aug_noise_cb.setChecked(False)
        self.aug_noise_cb.setToolTip(
            "Randomly add gaussian noise to images.\n"
            "Simulates sensor noise in satellite/aerial imagery.\n"
            "⚠️ PyTorch only (not available for YOLO)"
        )
        self.aug_noise_cb.stateChanged.connect(self.toggle_noise_spinbox) 
        radio_grid.addWidget(self.aug_noise_cb, row, 0)

        noise_label = QLabel("±")
        noise_label.setStyleSheet("color: gray; font-weight: bold;")
        noise_label.setAlignment(Qt.AlignCenter)
        radio_grid.addWidget(noise_label, row, 1)

        self.aug_noise_spin = QSpinBox()
        self.aug_noise_spin.setRange(0, 100)
        self.aug_noise_spin.setValue(0)
        self.aug_noise_spin.setSuffix("%")
        self.aug_noise_spin.setFixedWidth(80)
        self.aug_noise_spin.setEnabled(False) 
        self.aug_noise_spin.setToolTip("Noise intensity (0-100%)")
        radio_grid.addWidget(self.aug_noise_spin, row, 2)

        self.aug_noise_pytorch_label = QLabel("[PyTorch]")
        self.aug_noise_pytorch_label.setStyleSheet("color: gray; font-size: 9px;")
        radio_grid.addWidget(self.aug_noise_pytorch_label, row, 3)

        augmentation_layout.addLayout(radio_grid)

        augmentation_group.setLayout(augmentation_layout)
        layout.addWidget(augmentation_group)
        
        # ===================================================================
        # DEVICE CONFIGURATION
        # ===================================================================
        device_group = QGroupBox("Device")
        device_layout = QFormLayout()
        
        # Device selection (CPU or CUDA)
        self.device_combo = QComboBox()
        self.device_combo.addItems(['CPU', 'CUDA'])
        self.device_combo.currentTextChanged.connect(self.on_device_changed)
        device_layout.addRow("Device:", self.device_combo)
        
        # CUDA device ID (only visible when CUDA is selected)
        self.cuda_container = QWidget()
        cuda_layout = QHBoxLayout()
        cuda_layout.setContentsMargins(0, 0, 0, 0)
        self.cuda_device_label = QLabel("GPU ID:")
        self.cuda_device_spin = QSpinBox()
        self.cuda_device_spin.setRange(0, 7)
        self.cuda_device_spin.setValue(0)
        self.cuda_device_spin.setFixedWidth(60)
        self.cuda_device_spin.setToolTip("CUDA device ID (0-7)")
        cuda_layout.addWidget(self.cuda_device_label)
        cuda_layout.addWidget(self.cuda_device_spin)
        cuda_layout.addStretch()
        self.cuda_container.setLayout(cuda_layout)
        device_layout.addRow("", self.cuda_container)
        
        device_group.setLayout(device_layout)
        layout.addWidget(device_group)
        
        # Initialize CUDA visibility
        self.on_device_changed(self.device_combo.currentText())
        
        # ===================================================================
        # OUTPUT CONFIGURATION
        # ===================================================================
        output_group = QGroupBox("Output")
        output_layout = QFormLayout()
        
        # Output directory
        self.output_dir_edit = QLineEdit()
        self.output_dir_edit.setPlaceholderText("Select output folder for models...")
        output_btn = QPushButton("📁")
        output_btn.setMaximumWidth(40)
        output_btn.setToolTip("Browse for output folder")
        output_btn.clicked.connect(self.browse_output)
        
        output_row = QHBoxLayout()
        output_row.addWidget(self.output_dir_edit)
        output_row.addWidget(output_btn)
        output_layout.addRow("Directory:", output_row)
        # Model name
        self.model_name_edit = QLineEdit()
        self.model_name_edit.setPlaceholderText("model_name")
        self.model_name_edit.setToolTip(
            "Name for the exported .qmtp file.\n"
            "Use only letters, numbers, and underscores."
        )
        output_layout.addRow("Model Name:", self.model_name_edit)
        
        output_group.setLayout(output_layout)
        layout.addWidget(output_group)
        
        # =======================================================================
        # ADVANCED OPTIONS
        # =======================================================================
        adv_group = QGroupBox("Advanced Options")
        adv_layout = QVBoxLayout()

        # Early stopping
        self.early_stop_cb = QCheckBox("Enable Early Stopping")
        self.early_stop_cb.setToolTip(
            "Stop training if validation loss doesn't improve.\n"
            "Helps prevent overfitting."
        )
        self.early_stop_cb.stateChanged.connect(self.toggle_patience)
        adv_layout.addWidget(self.early_stop_cb)

        # Patience (only enabled when early stopping is checked)
        patience_layout = QHBoxLayout()
        patience_layout.addWidget(QLabel("Patience:"))
        self.patience_spin = QSpinBox()
        self.patience_spin.setRange(1, 100)
        self.patience_spin.setValue(10)
        self.patience_spin.setEnabled(False)
        self.patience_spin.setToolTip("Number of epochs to wait before stopping")
        patience_layout.addWidget(self.patience_spin)
        patience_layout.addStretch()
        adv_layout.addLayout(patience_layout)

        # Save best model
        self.save_best_cb = QCheckBox("Save Best Model")
        self.save_best_cb.setChecked(True)  
        self.save_best_cb.setToolTip(
            "Save the model with the best validation loss.\n"
            "Recommended to keep checked."
        )
        adv_layout.addWidget(self.save_best_cb)

        # --- Resume Training Section ---
        # Separator line
        separator = QWidget()
        separator.setFixedHeight(1)
        separator.setStyleSheet("background-color: #cccccc;")
        adv_layout.addWidget(separator)

        # Resume from checkpoint
        self.resume_training_cb = QCheckBox("Resume from Checkpoint")
        self.resume_training_cb.setToolTip(
            "Continue training from a previously saved checkpoint.\n"
            "This will restore model, optimizer, and scheduler states.\n"
            "(Not implemented in MVP)"
        )
        self.resume_training_cb.stateChanged.connect(self.toggle_checkpoint_path)
        adv_layout.addWidget(self.resume_training_cb)

        # Checkpoint path (only enabled when resume is checked)
        checkpoint_layout = QHBoxLayout()
        checkpoint_layout.addWidget(QLabel("Checkpoint:"))
        self.checkpoint_path_edit = QLineEdit()
        self.checkpoint_path_edit.setPlaceholderText("Select checkpoint file (.pth)...")
        self.checkpoint_path_edit.setEnabled(False)

        self.checkpoint_browse_btn = QPushButton("📁")
        self.checkpoint_browse_btn.setMaximumWidth(40)
        self.checkpoint_browse_btn.setToolTip("Browse for checkpoint file")
        self.checkpoint_browse_btn.setEnabled(False)
        self.checkpoint_browse_btn.clicked.connect(self.browse_checkpoint)

        checkpoint_layout.addWidget(self.checkpoint_path_edit)
        checkpoint_layout.addWidget(self.checkpoint_browse_btn)
        adv_layout.addLayout(checkpoint_layout)

        adv_group.setLayout(adv_layout)
        layout.addWidget(adv_group)
        
        # Finalize scrollable content
        content_widget.setLayout(layout)
        scroll_area.setWidget(content_widget)

        # Add scroll area to main layout
        main_layout.addWidget(scroll_area, 1)  
        main_layout.addSpacing(5)  

        # ===================================================================
        # PROGRESS & LOG 
        # ===================================================================
        # Container for progress/log section with padding
        progress_log_container = QWidget()
        progress_log_layout = QVBoxLayout()
        progress_log_layout.setContentsMargins(10, 5, 10, 5)
        progress_log_layout.setSpacing(5)

        # Progress bar
        progress_label = QLabel("Progress:")
        progress_label_font = QFont()
        progress_label_font.setBold(True)
        progress_label.setFont(progress_label_font)
        progress_log_layout.addWidget(progress_label)

        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        progress_log_layout.addWidget(self.progress_bar)

        # Add spacing between progress and log
        progress_log_layout.addSpacing(10)

        # Training log
        log_header = QHBoxLayout()
        log_label = QLabel("Training Log:")
        log_label_font = QFont()
        log_label_font.setBold(True)
        log_label.setFont(log_label_font)
        log_header.addWidget(log_label)

        # Clear log button
        self.clear_log_btn = QPushButton("Clear")
        self.clear_log_btn.setMaximumWidth(60)
        self.clear_log_btn.clicked.connect(self.clear_log)
        log_header.addWidget(self.clear_log_btn)
        log_header.addStretch()

        progress_log_layout.addLayout(log_header)

        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMaximumHeight(180)
        self.log_text.setStyleSheet(
            "QTextEdit { "
            "font-family: 'Courier New', monospace; "
            "font-size: 9pt; "
            "}"
        )
        progress_log_layout.addWidget(self.log_text)

        progress_log_container.setLayout(progress_log_layout)
        main_layout.addWidget(progress_log_container)

        # ===================================================================
        # CONTROL BUTTONS
        # ===================================================================
        btn_container = QWidget()
        btn_layout = QHBoxLayout()
        btn_layout.setContentsMargins(10, 0, 10, 10)
        btn_layout.setSpacing(10)

        self.start_btn = QPushButton("Start Training")
        self.start_btn.setStyleSheet(
            "QPushButton {"
            "    background-color: #4CAF50; "
            "    color: white; "
            "    padding: 8px; "
            "    font-weight: bold; "
            "    border: none; "
            "    border-radius: 4px;"
            "}"
            "QPushButton:hover {"
            "    background-color: #45a049;"
            "}"
            "QPushButton:disabled {"
            "    background-color: #cccccc;"
            "}"
        )
        self.start_btn.clicked.connect(self.on_start_clicked)

        self.stop_btn = QPushButton("Stop Training")
        self.stop_btn.setStyleSheet(
            "QPushButton {"
            "    background-color: #f44336; "
            "    color: white; "
            "    padding: 8px; "
            "    border: none; "
            "    border-radius: 4px;"
            "}"
            "QPushButton:hover {"
            "    background-color: #da190b;"
            "}"
            "QPushButton:disabled {"
            "    background-color: #cccccc;"
            "}"
        )
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self.on_stop_clicked)

        btn_layout.addWidget(self.start_btn)
        btn_layout.addWidget(self.stop_btn)
        btn_container.setLayout(btn_layout)
        main_layout.addWidget(btn_container)

        # ===== Footer =====
        footer_widget = self._create_footer()
        main_layout.addWidget(footer_widget)

        # Set main layout
        main_widget.setLayout(main_layout)
        self.setWidget(main_widget)

        # Initialize model configuration (after all widgets are created)
        self.update_architectures(self.task_combo.currentText())
        self.update_default_lr(self.task_combo.currentText()) 
        self.update_model_config_visibility(self.task_combo.currentText())
        self.update_augmentation_availability(self.task_combo.currentText())
        self.update_optimizer_scheduler_defaults(self.task_combo.currentText())

        # Initialize augmentation state (disable all if master checkbox is unchecked)
        self.toggle_augmentation_options(self.augmentation_enabled_cb.checkState())
        
        # Dock widget configuration
        self.setAllowedAreas(Qt.RightDockWidgetArea | Qt.LeftDockWidgetArea)
        self.setFeatures(
            QDockWidget.DockWidgetMovable |
            QDockWidget.DockWidgetFloatable |
            QDockWidget.DockWidgetClosable
        )
    
    # =======================================================================
    # DYNAMIC UI UPDATES
    # =======================================================================
    def _create_header(self) -> QWidget:
        """Create header widget"""
        header = QWidget()
        header.setStyleSheet(
            "background-color: #156db4; "
            "color: white; "
            "padding: 15px;"
        )
        
        layout = QVBoxLayout()
        layout.setContentsMargins(15, 15, 15, 15)
        layout.setSpacing(5)
        
        # Title
        title = QLabel("QModel Trainer")
        title_font = QFont()
        title_font.setPointSize(14)
        title_font.setBold(True)
        title.setFont(title_font)
        title.setStyleSheet("color: white;")
        layout.addWidget(title)
        
        # Subtitle
        subtitle = QLabel("Model Training Tool")
        subtitle.setStyleSheet("color: #BDC3C7; font-size: 10px;")
        layout.addWidget(subtitle)
        
        header.setLayout(layout)
        return header

    def _create_footer(self) -> QWidget:
        """Create footer widget"""
        footer = QWidget()
        footer.setStyleSheet(
            "background-color: #ECF0F1; "
            "border-top: 1px solid #BDC3C7; "
            "padding: 10px;"
        )
        
        layout = QVBoxLayout()
        layout.setContentsMargins(15, 10, 15, 10)
        
        footer_label = QLabel(
            "<b>QGeoAI</b> | edited by "
            "<a href='https://qgeoai.nextelia.fr/' style='color: #3498DB;'>Nextelia</a>"
        )
        footer_label.setOpenExternalLinks(True)
        footer_label.setAlignment(Qt.AlignCenter)
        footer_label.setStyleSheet("color: #7F8C8D; font-size: 9px;")
        layout.addWidget(footer_label)
        
        footer.setLayout(layout)
        return footer


    def update_architectures(self, task):
        """
        Update available architectures when task changes.
        
        Args:
            task: Selected task name (e.g., 'Semantic Segmentation')
        """
        self.arch_combo.clear()
        if task in self.TASKS:
            self.arch_combo.addItems(self.TASKS[task])
            self.update_backbones(self.arch_combo.currentText())
    
    def update_backbones(self, arch):
        """
        Update available backbones when architecture changes.
        
        Loads encoders dynamically from model_factory:
        - Semantic segmentation: All compatible encoders from SMP_ENCODER_FAMILIES
        - Segformer/DPT: Constrained encoders only
        - Mask R-CNN: Hardcoded backbones from MASKRCNN_BACKBONES
        
        Args:
            arch: Selected architecture name (e.g., 'UNet', 'Mask R-CNN')
        """
        self.backbone_combo.clear()
        
        if arch == 'Mask R-CNN':
            # Mask R-CNN: Use torchvision backbones
            for backbone in self.MASKRCNN_BACKBONES:
                # Display name = backbone name (simple case)
                self.backbone_combo.addItem(backbone, backbone)
        
        elif arch in self.ARCHITECTURE_ENCODER_CONSTRAINTS:
            # Architecture-specific encoders (Segformer, DPT)
            allowed_encoders = self.ARCHITECTURE_ENCODER_CONSTRAINTS[arch]
            
            # Find the family and add only allowed encoders
            for family_name, family_data in self.SMP_ENCODER_FAMILIES.items():
                encoders = family_data['encoders']
                
                for enc in encoders:
                    if enc in allowed_encoders:
                        # Use encoder name as display name
                        self.backbone_combo.addItem(enc, enc)
        
        else:
            # Generic SMP architectures: All CNN encoders available
            for family_name, family_data in self.SMP_ENCODER_FAMILIES.items():
                encoders = family_data['encoders']
                
                for enc in encoders:
                    # Use encoder name as display name
                    self.backbone_combo.addItem(enc, enc)
        
        # Update weights combo for first backbone
        if self.backbone_combo.count() > 0:
            first_encoder = self.backbone_combo.currentData()
            if first_encoder:
                self.update_weights_combo(first_encoder)
            else:
                # Fallback for Mask R-CNN (no data stored)
                self.update_weights_combo(self.backbone_combo.currentText())

    def update_model_config_visibility(self, task):
        """
        Show/hide model configuration fields based on task type.
        
        PyTorch tasks (Semantic/Instance Segmentation):
            → Show: Architecture + Backbone + Weights
            → Hide: Model + Size + Optimizer
        
        YOLO tasks (Instance Segmentation YOLO / Detection YOLO):
            → Show: Model + Size + Optimizer
            → Hide: Architecture + Backbone
            → Weights shown but fixed to "COCO"
        
        Args:
            task: Selected task name
        """
        is_yolo = task in self.YOLO_TASKS
        
        # PyTorch fields (Architecture + Backbone + Weights)
        self.arch_label.setVisible(not is_yolo)
        self.arch_combo.setVisible(not is_yolo)
        self.backbone_label.setVisible(not is_yolo)
        self.backbone_combo.setVisible(not is_yolo)
        
        # Weights field visibility
        # Always show weights field, but content changes based on task
        self.weights_label.setVisible(True)
        self.weights_combo.setVisible(True)
        
        # YOLO fields (Model + Size + Optimizer)
        self.yolo_model_label.setVisible(is_yolo)
        self.yolo_model_combo.setVisible(is_yolo)
        self.yolo_size_label.setVisible(is_yolo)
        self.yolo_size_combo.setVisible(is_yolo)
        self.optimizer_label.setVisible(is_yolo)
        self.optimizer_combo.setVisible(is_yolo)

        # PyTorch Optimizer & Scheduler
        self.optimizer_label_pytorch.setVisible(not is_yolo)
        self.optimizer_combo_pytorch.setVisible(not is_yolo)
        self.scheduler_label_pytorch.setVisible(not is_yolo)
        self.scheduler_combo_pytorch.setVisible(not is_yolo)
        
        # Val Split (hide for YOLO)
        self.val_split_label.setVisible(not is_yolo)
        self.val_split_spin.setVisible(not is_yolo)

        # Save best model (always enabled for YOLO, cannot be disabled)
        if is_yolo:
            self.save_best_cb.setChecked(True)
            self.save_best_cb.setEnabled(False)
            self.save_best_cb.setToolTip(
                "YOLO automatically saves both best.pt and last.pt checkpoints.\n"
                "This option cannot be disabled for YOLO models."
            )
        else:
            self.save_best_cb.setEnabled(True)
            self.save_best_cb.setToolTip(
                "Save the model with the best validation loss.\n"
                "Recommended to keep checked."
            )
        
        # Resume training (disable for YOLO)
        if is_yolo:
            self.resume_training_cb.setEnabled(False)
            self.resume_training_cb.setChecked(False)
            self.resume_training_cb.setToolTip(
                "Resume training is not available yet for YOLO models.\n"
            )
        else:
            self.resume_training_cb.setEnabled(True)
            self.resume_training_cb.setToolTip(
                "Continue training from a previously saved checkpoint.\n"
                "This will restore model, optimizer, and scheduler states."
            )        
        
        # Update YOLO model dropdown based on task
        if is_yolo:
            self.yolo_model_combo.clear()
            self.yolo_model_combo.addItems(self.YOLO_TASKS[task])
            # Update weights for YOLO
            self.update_weights_combo('')
        else:
            # Update weights for PyTorch
            self.update_weights_combo(self.backbone_combo.currentText())

    def update_optimizer_scheduler_defaults(self, task):
        """
        Update default optimizer and scheduler based on selected task.
        
        Args:
            task: Selected task name
        """
        if task in self.YOLO_TASKS:
            # YOLO: Not applicable (uses YOLO optimizer dropdown)
            return
        
        if task == 'Semantic Segmentation':
            # Semantic: Adam + ReduceLROnPlateau
            self.optimizer_combo_pytorch.setCurrentText('Adam')
            self.scheduler_combo_pytorch.setCurrentText('ReduceLROnPlateau')
        
        elif task == 'Instance Segmentation':
            # Instance: SGD + StepLR
            self.optimizer_combo_pytorch.setCurrentText('SGD')
            self.scheduler_combo_pytorch.setCurrentText('StepLR')

    def toggle_weights_combo(self, state):
        """
        Enable/disable weights combo based on pretrained checkbox.
        
        Args:
            state: Checkbox state (Qt.Checked or Qt.Unchecked)
        """
        is_enabled = (state == Qt.Checked)
        self.weights_combo.setEnabled(is_enabled)
        
        # Visual feedback
        if not is_enabled:
            self.weights_combo.setStyleSheet("QComboBox { background-color: #f0f0f0; }")
        else:
            self.weights_combo.setStyleSheet("")
            # Update weights when re-enabled
            self.update_weights_combo(self.backbone_combo.currentText())
    
    def update_weights_combo(self, backbone_display_or_key: str):
        """
        Update available pretrained weights when backbone changes.
        
        Loads weights dynamically from ENCODER_PRETRAINED_WEIGHTS.
        
        Args:
            backbone_display_or_key: Backbone key or display name
        """
        self.weights_combo.clear()
        
        # Get actual encoder key from combo
        backbone = self.backbone_combo.currentData()
        if not backbone:
            # Fallback to text (for Mask R-CNN or manual calls)
            backbone = self.backbone_combo.currentText()
        
        # Check task type
        task = self.task_combo.currentText()
        is_yolo = task in self.YOLO_TASKS
        is_instance = task == 'Instance Segmentation'
        
        if is_yolo:
            # YOLO only has COCO weights
            self.weights_combo.addItem("COCO", "coco")
            self.weights_combo.setEnabled(self.pretrained_cb.isChecked())
        
        elif is_instance:
            # Mask R-CNN only has COCO weights (for full model)
            self.weights_combo.addItem("COCO", "coco")
            self.weights_combo.setEnabled(self.pretrained_cb.isChecked())
        
        else:
            # Semantic Segmentation: Get weights from model_factory
            if backbone in self.ENCODER_PRETRAINED_WEIGHTS:
                weights = self.ENCODER_PRETRAINED_WEIGHTS[backbone]
                
                # Add weights with user-friendly names
                for weight in weights:
                    if weight == 'imagenet':
                        self.weights_combo.addItem("ImageNet", weight)
                    elif weight == 'ssl':
                        self.weights_combo.addItem("SSL (Semi-Supervised)", weight)
                    elif weight == 'swsl':
                        self.weights_combo.addItem("SWSL (Weakly-Supervised)", weight)
                    else:
                        self.weights_combo.addItem(weight.upper(), weight)
                
                # Set default to ImageNet if available
                idx = self.weights_combo.findData('imagenet')
                if idx >= 0:
                    self.weights_combo.setCurrentIndex(idx)
            else:
                # Fallback: assume ImageNet
                self.weights_combo.addItem("ImageNet", "imagenet")
            
            self.weights_combo.setEnabled(self.pretrained_cb.isChecked())
    
    def on_device_changed(self, device_text):
        """
        Show/hide CUDA device selector based on device selection.
        
        Args:
            device_text: Selected device ('CPU' or 'CUDA')
        """
        self.cuda_container.setVisible(device_text == "CUDA")
    
    def toggle_patience(self, state):
        """
        Enable/disable patience spinbox based on early stopping checkbox.
        
        Args:
            state: Checkbox state (Qt.Checked or Qt.Unchecked)
        """
        self.patience_spin.setEnabled(state == Qt.Checked)

    def toggle_checkpoint_path(self, state):
        """
        Enable/disable checkpoint path controls based on resume checkbox.
        
        Args:
            state: Checkbox state (Qt.Checked or Qt.Unchecked)
        """
        is_enabled = (state == Qt.Checked)
        self.checkpoint_path_edit.setEnabled(is_enabled)
        self.checkpoint_browse_btn.setEnabled(is_enabled)

    def update_default_lr(self, task):
        """
        Update default learning rate based on selected task.
        
        Args:
            task: Selected task name
        """
        is_yolo = task in self.YOLO_TASKS
        
        if is_yolo:
            # YOLO default: 0.01
            self.lr_spin.setValue(0.01)
        else:
            # PyTorch default: 0.001 (Semantic) or 0.005 (Instance)
            if 'Instance' in task:
                self.lr_spin.setValue(0.005)
            else:
                self.lr_spin.setValue(0.001)
    
    def toggle_lr_auto(self, state):
        """
        Enable/disable manual LR input based on Auto LR checkbox.
        
        Args:
            state: Checkbox state (Qt.Checked or Qt.Unchecked)
        """
        # When Auto is checked, disable manual input
        is_manual = (state != Qt.Checked)
        self.lr_spin.setEnabled(is_manual)
        
        # Visual feedback
        if not is_manual:
            self.lr_spin.setStyleSheet("QDoubleSpinBox { background-color: #f0f0f0; }")
        else:
            self.lr_spin.setStyleSheet("")

    def update_auto_lr_availability(self, task):
        """
        Enable/disable Auto LR based on task compatibility.
        LR Finder is only supported for Semantic Segmentation (PyTorch).
        
        Args:
            task: Selected task name
        """
        is_semantic = (task == 'Semantic Segmentation')
        is_yolo = task in self.YOLO_TASKS
        
        # Enable Auto LR only for Semantic Segmentation
        self.auto_lr_cb.setEnabled(is_semantic)
        
        if is_yolo:
            # Disable for YOLO (Ultralytics handles LR internally)
            self.auto_lr_cb.setChecked(False)
            self.auto_lr_cb.setToolTip(
                "Auto LR Finder is not available for YOLO.\n"
                "Reason: Ultralytics manages learning rate scheduling internally.\n"
                "You can manually set the learning rate above."
            )
        elif not is_semantic:
            # Disable for Instance Segmentation (Mask R-CNN)
            self.auto_lr_cb.setChecked(False)
            self.auto_lr_cb.setToolTip(
                "Auto LR Finder is not supported for Instance Segmentation.\n"
                "Reason: Mask R-CNN uses multi-component loss + OneCycleLR.\n"
                "You can manually set the learning rate above (default: 0.005)."
            )
        else:
            # Enable for Semantic Segmentation
            self.auto_lr_cb.setToolTip(
                "Estimate learning rate automatically (LR Finder).\n"
                "⚠️ Note: Uses Adam as a calibration probe to determine the LR scale for your selected optimizer.\n"
                "Adds a short calibration step (≈60 s) before training.\n"
            )

    def update_default_lr(self, task):
        """
        Update default learning rate based on selected task and optimizer.
        
        Args:
            task: Selected task name
        """
        # Skip if YOLO task (has its own default)
        is_yolo = task in self.YOLO_TASKS
        
        if is_yolo:
            # YOLO default LR
            self.lr_spin.setValue(0.01)
        else:
            # PyTorch: use default LR based on optimizer and task
            optimizer_name = self.optimizer_combo_pytorch.currentText()
            
            # Default LRs (server will handle actual optimization)
            # Based on common best practices
            if 'Instance' in task:
                # Instance segmentation typically needs lower LR
                default_lrs = {
                    'Adam': 0.0005,
                    'AdamW': 0.0005,
                    'SGD': 0.005
                }
            else:
                # Semantic segmentation
                default_lrs = {
                    'Adam': 0.001,
                    'AdamW': 0.001,
                    'SGD': 0.01
                }
            
            default_lr = default_lrs.get(optimizer_name, 0.001)
            self.lr_spin.setValue(default_lr)

    def toggle_augmentation_options(self, state):
        """
        Enable/disable augmentation checkboxes and spinboxes based on master checkbox.
        Respects task-specific constraints (YOLO vs PyTorch).
        
        Args:
            state: Checkbox state (Qt.Checked or Qt.Unchecked)
        """
        is_enabled = (state == Qt.Checked)
        
        # Get current task to respect constraints
        task = self.task_combo.currentText()
        is_yolo = task in self.YOLO_TASKS
        
        # Geometric augmentations (always available for all tasks)
        self.aug_hflip_cb.setEnabled(is_enabled)
        self.aug_vflip_cb.setEnabled(is_enabled)
        self.aug_rotate90_cb.setEnabled(is_enabled)
        
        # Brightness (available for all tasks)
        self.aug_brightness_cb.setEnabled(is_enabled)
        self.aug_brightness_spin.setEnabled(is_enabled and self.aug_brightness_cb.isChecked())
        
        # PyTorch-only augmentations
        self.aug_contrast_cb.setEnabled(is_enabled and not is_yolo)
        self.aug_contrast_spin.setEnabled(is_enabled and not is_yolo and self.aug_contrast_cb.isChecked())
        
        self.aug_hue_cb.setEnabled(is_enabled and not is_yolo)
        self.aug_hue_spin.setEnabled(is_enabled and not is_yolo and self.aug_hue_cb.isChecked())
        
        self.aug_saturation_cb.setEnabled(is_enabled and not is_yolo)
        self.aug_saturation_spin.setEnabled(is_enabled and not is_yolo and self.aug_saturation_cb.isChecked())
        
        self.aug_blur_cb.setEnabled(is_enabled and not is_yolo)
        self.aug_blur_spin.setEnabled(is_enabled and not is_yolo and self.aug_blur_cb.isChecked())
        
        self.aug_noise_cb.setEnabled(is_enabled and not is_yolo)
        self.aug_noise_spin.setEnabled(is_enabled and not is_yolo and self.aug_noise_cb.isChecked())
    
    def update_augmentation_availability(self, task):
        """
        Update augmentation availability based on task type.
        
        YOLO tasks: Only Brightness available for radiometric augmentations
        PyTorch tasks: All augmentations available
        
        Args:
            task: Selected task name
        """
        is_yolo = task in self.YOLO_TASKS
        is_aug_enabled = self.augmentation_enabled_cb.isChecked()
        
        # Geometric augmentations: available for all tasks
        # (already handled by toggle_augmentation_options)
        
        # Brightness: available for all tasks (PyTorch + YOLO)
        self.aug_brightness_cb.setEnabled(is_aug_enabled)
        self.aug_brightness_spin.setEnabled(is_aug_enabled)
        
        # PyTorch-only augmentations (disable for YOLO)
        pytorch_only_widgets = [
            (self.aug_contrast_cb, self.aug_contrast_spin, self.aug_contrast_pytorch_label),
            (self.aug_hue_cb, self.aug_hue_spin, self.aug_hue_pytorch_label),
            (self.aug_saturation_cb, self.aug_saturation_spin, self.aug_saturation_pytorch_label),
            (self.aug_blur_cb, self.aug_blur_spin, self.aug_blur_pytorch_label),
            (self.aug_noise_cb, self.aug_noise_spin, self.aug_noise_pytorch_label)
        ]
        
        for checkbox, spinbox, label in pytorch_only_widgets:
            checkbox.setEnabled(not is_yolo and is_aug_enabled)
            spinbox.setEnabled(not is_yolo and is_aug_enabled)
            
            if is_yolo:
                # Uncheck and reset for YOLO
                checkbox.setChecked(False)
                if isinstance(spinbox, QDoubleSpinBox):
                    spinbox.setValue(0.0)
                else:
                    spinbox.setValue(0)
                
                # Update label style to indicate disabled
                label.setStyleSheet("color: #999999; font-size: 9px;")
            else:
                # Restore normal style for PyTorch tasks
                label.setStyleSheet("color: gray; font-size: 9px;")

    # =======================================================================
    # FILE BROWSER METHODS
    # =======================================================================
    
    def browse_dataset(self):
        """Open file dialog to select dataset directory."""
        directory = QFileDialog.getExistingDirectory(
            self,
            "Select Dataset Directory (QAnnotate Export)"
        )
        if directory:
            self.dataset_path_edit.setText(directory)
            # Auto-validate after selecting (optionnel)
            # self.validate_dataset()
    
    def browse_output(self):
        """Open file dialog to select output directory."""
        directory = QFileDialog.getExistingDirectory(
            self,
            "Select Output Directory for Trained Models"
        )
        if directory:
            self.output_dir_edit.setText(directory)

    def browse_checkpoint(self):
        """Open file dialog to select checkpoint file."""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Checkpoint File",
            "",
            "PyTorch Checkpoint (*.pth *.pt);;All Files (*)"
        )
        if file_path:
            self.checkpoint_path_edit.setText(file_path)

    # =======================================================================
    # DATASET VALIDATION
    # =======================================================================

    def auto_validate_dataset(self, text):
        """
        Auto-validate dataset when path changes.
        
        Args:
            text: New dataset path
        """
        from ..utils.ui_validators import validate_metadata_exists, get_basic_dataset_info, format_dataset_info
        
        # Reset state
        self._dataset_validated = False
        self._dataset_info = None
        
        dataset_path = text.strip()
        
        if not dataset_path:
            self.dataset_info_label.setText("No dataset loaded")
            self.dataset_info_label.setStyleSheet("color: gray; font-size: 10px;")
            return
        
        # Check if path exists before validating
        import os
        if not os.path.exists(dataset_path):
            self.dataset_info_label.setText("⚠️ Path does not exist")
            self.dataset_info_label.setStyleSheet("color: orange; font-size: 10px;")
            return
        
        # Show validating status
        self.dataset_info_label.setText("🔄 Validating...")
        self.dataset_info_label.setStyleSheet("color: blue; font-size: 10px;")
        
        # Quick check metadata exists
        valid, error, metadata = validate_metadata_exists(dataset_path)
        
        if not valid:
            self.dataset_info_label.setText(f"❌ {error}")
            self.dataset_info_label.setStyleSheet("color: red; font-size: 10px;")
            return
        
        # Get basic info for display
        dataset_info = get_basic_dataset_info(metadata)
        info_text = format_dataset_info(dataset_info)
        
        self.dataset_info_label.setText(f"✅ {info_text}")
        self.dataset_info_label.setStyleSheet("color: green; font-size: 10px;")
        
        self._dataset_validated = True
        self._dataset_info = dataset_info
        
        # Auto-select appropriate task based on dataset format
        export_format = dataset_info.get('export_format', 'mask')

        if export_format == 'coco':
            # COCO format → Instance Segmentation (PyTorch)
            task_index = self.task_combo.findText('Instance Segmentation')
            if task_index >= 0:
                self.task_combo.setCurrentIndex(task_index)
        elif export_format == 'mask':
            # Mask format → Semantic Segmentation
            task_index = self.task_combo.findText('Semantic Segmentation')
            if task_index >= 0:
                self.task_combo.setCurrentIndex(task_index)
        elif export_format == 'yolo11-detect':
            # YOLO detection → Detection (YOLO)
            task_index = self.task_combo.findText('Detection (YOLO)')
            if task_index >= 0:
                self.task_combo.setCurrentIndex(task_index)
                # Auto-select YOLO11 (not OBB)
                model_index = self.yolo_model_combo.findText('YOLO11')
                if model_index >= 0:
                    self.yolo_model_combo.setCurrentIndex(model_index)
        elif export_format == 'yolo11-seg':
            # YOLO segmentation → Instance Segmentation (YOLO)
            task_index = self.task_combo.findText('Instance Segmentation (YOLO)')
            if task_index >= 0:
                self.task_combo.setCurrentIndex(task_index)
        elif export_format == 'yolo11-obb':
            # YOLO OBB → Detection (YOLO)
            task_index = self.task_combo.findText('Detection (YOLO)')
            if task_index >= 0:
                self.task_combo.setCurrentIndex(task_index)
                # Auto-select YOLO11-obb
                model_index = self.yolo_model_combo.findText('YOLO11-obb')
                if model_index >= 0:
                    self.yolo_model_combo.setCurrentIndex(model_index)


    def get_validated_dataset_info(self):
        """
        Get validated dataset info.
        
        Returns:
            dict or None: Dataset info if validated, None otherwise
        """
        return self._dataset_info if self._dataset_validated else None
    
    
    def validate_task_dataset_compatibility(self) -> tuple[bool, str]:
        """
        Validate that the selected task is compatible with the dataset format.
        
        Returns:
            Tuple of (is_compatible, error_message)
        """
        if not self._dataset_validated or not self._dataset_info:
            return False, "No dataset validated"
        
        task = self.task_combo.currentText()
        export_format = self._dataset_info.get('export_format', 'unknown')
        
        # PyTorch tasks
        if task == 'Semantic Segmentation' and export_format != 'mask':
            return False, (
                f"Semantic Segmentation requires 'mask' format dataset.\n"
                f"Current dataset is '{export_format}' format.\n"
                f"Please select a mask format dataset."
            )
        
        if task == 'Instance Segmentation' and export_format != 'coco':
            return False, (
                f"Instance Segmentation (Mask R-CNN) requires 'coco' format dataset.\n"
                f"Current dataset is '{export_format}' format.\n"
                f"Please select a COCO format dataset."
            )
        
        # YOLO tasks
        if task == 'Instance Segmentation (YOLO)' and export_format != 'yolo11-seg':
            return False, (
                f"Instance Segmentation (YOLO) requires 'yolo11-seg' format dataset.\n"
                f"Current dataset is '{export_format}' format.\n"
                f"Please export in YOLO11-seg format from QAnnotate."
            )
        
        if task == 'Detection (YOLO)':
            model = self.yolo_model_combo.currentText()
            
            if model == 'YOLO11' and export_format != 'yolo11-detect':
                return False, (
                    f"YOLO11 detection requires 'yolo11-detect' format dataset.\n"
                    f"Current dataset is '{export_format}' format.\n"
                    f"Please export in YOLO11-detect format from QAnnotate."
                )
            
            if model == 'YOLO11-obb' and export_format != 'yolo11-obb':
                return False, (
                    f"YOLO11-obb requires 'yolo11-obb' format dataset.\n"
                    f"Current dataset is '{export_format}' format.\n"
                    f"Please export in YOLO11-obb format from QAnnotate."
                )
        
        return True, ""
    
    # =======================================================================
    # LOG MANAGEMENT
    # =======================================================================
    
    def append_log(self, message):
        """
        Append a message to the training log.
        
        Args:
            message: Text message to append
        """
        self.log_text.append(message)
        # Auto-scroll to bottom
        self.log_text.verticalScrollBar().setValue(
            self.log_text.verticalScrollBar().maximum()
        )
    
    def clear_log(self):
        """Clear the training log."""
        self.log_text.clear()
    
    # =======================================================================
    # TRAINING CONTROL
    # =======================================================================
    
    def on_start_clicked(self):
        """
        Handle Start button click.
        Validates dataset before starting if not already validated.
        """
        # Check if dataset is validated
        if not hasattr(self, '_dataset_validated') or not self._dataset_validated:
            self.append_log("\n❌ Dataset validation failed. Please select a valid QAnnotate export folder.\n")
            return
        # Check task/dataset compatibility
        compatible, error_msg = self.validate_task_dataset_compatibility()
        if not compatible:
            self.append_log(f"\n❌ Task/Dataset incompatibility:\n{error_msg}\n")
            return
        
        # Log detailed dataset info when starting training
        dataset_info = self._dataset_info
        self.append_log("\n" + "="*50)
        self.append_log("Dataset Information")
        self.append_log("="*50)
        self.append_log(f"Format: {dataset_info['export_format']}")
        self.append_log(f"Images: {dataset_info['num_images']}")
        self.append_log(f"   - Image format: {dataset_info['image_format']}")
        self.append_log(f"   - Mask format: {dataset_info['mask_format']}")
        self.append_log(f"Classes: {dataset_info['num_classes']} (including background)")
        class_names = dataset_info['class_names']
        if len(class_names) <= 5:
            self.append_log(f"   - {', '.join(class_names)}")
        else:
            self.append_log(f"   - {', '.join(class_names[:3])}, ... (+{len(class_names)-3} more)")
        self.append_log("="*50 + "\n")
        
        # Disable start button, enable stop button
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        
        # Emit signal (will be connected to TrainingManager)
        self.training_requested.emit()
    
    def on_stop_clicked(self):
        """
        Handle Stop button click.
        Emits training_cancelled signal for TrainingManager to handle.
        """
        self.append_log("\n⚠️ Stop requested by user...")
        self.training_cancelled.emit()
    
    def training_finished(self):
        """
        Called when training completes or is cancelled.
        Resets button states.
        """
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
    
    # =======================================================================
    # CONFIGURATION GETTERS
    # =======================================================================
    
    def get_training_config(self):
        """
        Collect all training configuration from UI.
        
        Returns:
            dict: Complete training configuration dictionary
        """
        task = self.task_combo.currentText()
        is_yolo = task in self.YOLO_TASKS
        
        # Base config
        config = {
            # Dataset
            'dataset_path': self.dataset_path_edit.text(),
            'dataset_info': self._dataset_info,
            
            # Model
            'task': task,
            'pretrained': self.pretrained_cb.isChecked(),
            'encoder_weights': self.weights_combo.currentData() if self.pretrained_cb.isChecked() else None,
            
            # Training hyperparameters
            'epochs': self.epochs_spin.value(),
            'batch_size': self.batch_spin.value(),
            'learning_rate': self.lr_spin.value(),
            'auto_lr': self.auto_lr_cb.isChecked() and not is_yolo,
            'image_size': self.image_size_spin.value(),
            'val_split': self.val_split_spin.value(),
            
            # Data augmentation configuration
            'augmentation_config': {
                'enabled': self.augmentation_enabled_cb.isChecked(),
                # Geometric (probabilities as percentages 0-100)
                'hflip': 50 if self.aug_hflip_cb.isChecked() else 0,
                'vflip': 50 if self.aug_vflip_cb.isChecked() else 0,
                'rotate90': 50 if self.aug_rotate90_cb.isChecked() else 0,
                # Radiometric (intensities 0-100)
                'brightness': self.aug_brightness_spin.value() if self.aug_brightness_cb.isChecked() else 0,
                'contrast': self.aug_contrast_spin.value() if self.aug_contrast_cb.isChecked() else 0,
                'hue': self.aug_hue_spin.value() if self.aug_hue_cb.isChecked() else 0,
                'saturation': self.aug_saturation_spin.value() if self.aug_saturation_cb.isChecked() else 0,
                'blur': self.aug_blur_spin.value() if self.aug_blur_cb.isChecked() else 0,
                'noise': self.aug_noise_spin.value() if self.aug_noise_cb.isChecked() else 0,
            },
            
            # Device
            'device': self.device_combo.currentText(),
            'cuda_device_id': self.cuda_device_spin.value() if self.device_combo.currentText() == 'CUDA' else None,
            
            # Output
            'output_dir': self.output_dir_edit.text(),
            'model_name': self.model_name_edit.text() or 'trained_model',
            
            # Advanced options
            'early_stopping': self.early_stop_cb.isChecked(),
            'patience': self.patience_spin.value() if self.early_stop_cb.isChecked() else None,
            'save_best': self.save_best_cb.isChecked(),
            
            # Resume training
            'resume_training': self.resume_training_cb.isChecked(),
            'checkpoint_path': self.checkpoint_path_edit.text() if self.resume_training_cb.isChecked() else None
        }
        
        # Add task-specific config
        if is_yolo:
            # YOLO-specific config
            config['yolo_model'] = self.yolo_model_combo.currentText()
            config['model_size'] = self.yolo_size_combo.currentText()
            config['optimizer'] = self.optimizer_combo.currentText()
            # Set dummy values for PyTorch fields (not used but expected by some validators)
            config['architecture'] = None
            config['backbone'] = None
            config['pytorch_optimizer'] = None  
            config['pytorch_scheduler'] = None  
        else:
            # PyTorch-specific config
            config['architecture'] = self.arch_combo.currentText()
            
            # Backbone (serveur attend "backbone")
            backbone_data = self.backbone_combo.currentData()
            config['backbone'] = backbone_data if backbone_data else self.backbone_combo.currentText()
            
            # Use PyTorch canonical names directly (no mapping)
            config['optimizer'] = self.optimizer_combo_pytorch.currentText()  # 'Adam', 'AdamW', 'SGD'
            
            scheduler_text = self.scheduler_combo_pytorch.currentText()
            config['scheduler'] = None if scheduler_text == 'None' else scheduler_text  # 'ReduceLROnPlateau', etc. or None
            
            # Set dummy values for YOLO fields (not used)
            config['yolo_model'] = None
            config['model_size'] = None
        
        return config

    # =======================================================================
    # AUGMENTATION SPINBOX TOGGLES
    # =======================================================================

    def toggle_brightness_spinbox(self, state):
        """Enable/disable brightness spinbox based on checkbox."""
        is_enabled = (state == Qt.Checked) and self.augmentation_enabled_cb.isChecked()
        self.aug_brightness_spin.setEnabled(is_enabled)

    def toggle_contrast_spinbox(self, state):
        """Enable/disable contrast spinbox based on checkbox."""
        task = self.task_combo.currentText()
        is_yolo = task in self.YOLO_TASKS
        is_enabled = (state == Qt.Checked) and self.augmentation_enabled_cb.isChecked() and not is_yolo
        self.aug_contrast_spin.setEnabled(is_enabled)

    def toggle_hue_spinbox(self, state):
        """Enable/disable hue spinbox based on checkbox."""
        task = self.task_combo.currentText()
        is_yolo = task in self.YOLO_TASKS
        is_enabled = (state == Qt.Checked) and self.augmentation_enabled_cb.isChecked() and not is_yolo
        self.aug_hue_spin.setEnabled(is_enabled)

    def toggle_saturation_spinbox(self, state):
        """Enable/disable saturation spinbox based on checkbox."""
        task = self.task_combo.currentText()
        is_yolo = task in self.YOLO_TASKS
        is_enabled = (state == Qt.Checked) and self.augmentation_enabled_cb.isChecked() and not is_yolo
        self.aug_saturation_spin.setEnabled(is_enabled)

    def toggle_blur_spinbox(self, state):
        """Enable/disable blur spinbox based on checkbox."""
        task = self.task_combo.currentText()
        is_yolo = task in self.YOLO_TASKS
        is_enabled = (state == Qt.Checked) and self.augmentation_enabled_cb.isChecked() and not is_yolo
        self.aug_blur_spin.setEnabled(is_enabled)

    def toggle_noise_spinbox(self, state):
        """Enable/disable noise spinbox based on checkbox."""
        task = self.task_combo.currentText()
        is_yolo = task in self.YOLO_TASKS
        is_enabled = (state == Qt.Checked) and self.augmentation_enabled_cb.isChecked() and not is_yolo
        self.aug_noise_spin.setEnabled(is_enabled)   
    
    def closeEvent(self, event):
        """
        Called when dock is closed (X button).
        
        Args:
            event: Close event
        """
        event.ignore()
        self.hide()
    
    def cleanup(self):
        """
        Cleanup resources before unload.
        Called by plugin's unload() method.
        """
        # Clear log to free memory
        self.log_text.clear()
        
        # Reset state
        self._dataset_validated = False
        self._dataset_info = None