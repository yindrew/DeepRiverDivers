import matplotlib.pyplot as plt
from utils.training_history import TrainingHistory

def plot_training_metrics(training_history: TrainingHistory, save_path: str | None = None):
    """
    Plot training and validation metrics (loss and accuracy) over epochs.
    
    Args:
        training_history: TrainingHistory object containing the metrics
        save_path: Optional path to save the plot. If None, the plot is displayed.
    """
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))
    
    # Get epochs for x-axis - use the length of training metrics
    epochs = range(1, len(training_history.training_loss_in_epochs))
    
    # Plot losses - ensure we only plot up to the length of training metrics
    ax1.plot(epochs, training_history.training_loss_in_epochs[1:], 'b-', label='Training Loss')
    if len(training_history.validation_loss_in_epochs) > 1:
        # Ensure validation loss array matches the length of epochs
        val_loss = training_history.validation_loss_in_epochs[1:len(epochs)+1]
        ax1.plot(epochs[:len(val_loss)], val_loss, 'r-', label='Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Plot accuracies - ensure we only plot up to the length of training metrics
    if training_history.training_accuracy_in_epochs:
        ax2.plot(epochs[:len(training_history.training_accuracy_in_epochs)], 
                training_history.training_accuracy_in_epochs, 'b-', label='Training Accuracy')
    if training_history.validation_accuracy_in_epochs:
        ax2.plot(epochs[:len(training_history.validation_accuracy_in_epochs)], 
                training_history.validation_accuracy_in_epochs, 'r-', label='Validation Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Training and Validation Accuracy')
    ax2.legend()
    ax2.grid(True)
    
    # Adjust layout to prevent overlap
    plt.tight_layout()
    
    # Save or show the plot
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def plot_learning_curves(training_history: TrainingHistory, save_path: str | None = None):
    """
    Plot learning curves showing the relationship between training and validation metrics.
    
    Args:
        training_history: TrainingHistory object containing the metrics
        save_path: Optional path to save the plot. If None, the plot is displayed.
    """
    plt.figure(figsize=(10, 6))
    
    # Only plot if we have both training and validation losses
    if len(training_history.training_loss_in_epochs) > 1 and len(training_history.validation_loss_in_epochs) > 1:
        # Get the minimum length of both arrays
        min_len = min(len(training_history.training_loss_in_epochs[1:]), 
                     len(training_history.validation_loss_in_epochs[1:]))
        
        # Get the arrays to plot
        train_loss = training_history.training_loss_in_epochs[1:min_len+1]
        val_loss = training_history.validation_loss_in_epochs[1:min_len+1]
        
        # Plot training vs validation loss
        plt.scatter(train_loss, val_loss, alpha=0.5)
        
        # Add a diagonal line
        min_val = min(min(train_loss), min(val_loss))
        max_val = max(max(train_loss), max(val_loss))
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Correlation')
        
        plt.xlabel('Training Loss')
        plt.ylabel('Validation Loss')
        plt.title('Learning Curves: Training vs Validation Loss')
        plt.legend()
        plt.grid(True)
        
        # Save or show the plot
        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()

def plot_accuracy_curves(training_history: TrainingHistory, save_path: str | None = None):
    """
    Plot accuracy curves showing the relationship between training and validation accuracy.
    
    Args:
        training_history: TrainingHistory object containing the metrics
        save_path: Optional path to save the plot. If None, the plot is displayed.
    """
    plt.figure(figsize=(10, 6))
    
    # Only plot if we have both training and validation accuracies
    if training_history.training_accuracy_in_epochs and training_history.validation_accuracy_in_epochs:
        # Get the minimum length of both arrays
        min_len = min(len(training_history.training_accuracy_in_epochs), 
                     len(training_history.validation_accuracy_in_epochs))
        
        # Get the arrays to plot
        train_acc = training_history.training_accuracy_in_epochs[:min_len]
        val_acc = training_history.validation_accuracy_in_epochs[:min_len]
        
        # Plot training vs validation accuracy
        plt.scatter(train_acc, val_acc, alpha=0.5)
        
        # Add a diagonal line
        min_val = min(min(train_acc), min(val_acc))
        max_val = max(max(train_acc), max(val_acc))
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Correlation')
        
        plt.xlabel('Training Accuracy')
        plt.ylabel('Validation Accuracy')
        plt.title('Accuracy Curves: Training vs Validation Accuracy')
        plt.legend()
        plt.grid(True)
        
        # Save or show the plot
        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()

def plot_accuracy_over_time(training_history: TrainingHistory, save_path: str | None = None):
    """
    Plot training and validation accuracy over epochs.
    
    Args:
        training_history: TrainingHistory object containing the metrics
        save_path: Optional path to save the plot. If None, the plot is displayed.
    """
    plt.figure(figsize=(10, 6))
    
    # Get epochs for x-axis
    epochs = range(1, len(training_history.training_loss_in_epochs))
    
    # Plot accuracies
    if training_history.training_accuracy_in_epochs:
        plt.plot(epochs[:len(training_history.training_accuracy_in_epochs)], 
                training_history.training_accuracy_in_epochs, 'b-', label='Training Accuracy')
    if training_history.validation_accuracy_in_epochs:
        plt.plot(epochs[:len(training_history.validation_accuracy_in_epochs)], 
                training_history.validation_accuracy_in_epochs, 'r-', label='Validation Accuracy')
    
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy Over Time')
    plt.legend()
    plt.grid(True)
    
    # Save or show the plot
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def plot_all_metrics(training_history: TrainingHistory, save_dir: str | None = None):
    """
    Plot all training metrics and save them to files if a directory is provided.
    
    Args:
        training_history: TrainingHistory object containing the metrics
        save_dir: Optional directory to save the plots. If None, the plots are displayed.
    """
    # Create save paths if directory is provided
    metrics_path = f"{save_dir}/training_metrics.png" if save_dir else None
    learning_curves_path = f"{save_dir}/learning_curves.png" if save_dir else None
    accuracy_curves_path = f"{save_dir}/accuracy_curves.png" if save_dir else None
    accuracy_over_time_path = f"{save_dir}/accuracy_over_time.png" if save_dir else None
    
    # Plot all metrics
    plot_training_metrics(training_history, metrics_path)
    plot_learning_curves(training_history, learning_curves_path)
    plot_accuracy_curves(training_history, accuracy_curves_path)
    plot_accuracy_over_time(training_history, accuracy_over_time_path) 