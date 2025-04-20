import matplotlib.pyplot as plt
import numpy as np
import json
import os
from typing import Dict, List, Optional

def plot_training_metrics(
    history: Dict[str, List[float]],
    output_dir: str,
    figsize: tuple = (12, 9)
) -> None:
    """
    Plot training metrics and save the plots to the output directory.
    
    Args:
        history: Dictionary containing training metrics (train_loss, val_loss, etc.)
        output_dir: Directory to save the plots
        figsize: Size of the figures
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Set up plot style
    plt.style.use('ggplot')
    
    # Create a figure with multiple subplots for losses
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    # Plot training and validation loss
    ax = axes[0, 0]
    epochs = range(1, len(history['train_loss']) + 1)
    ax.plot(epochs, history['train_loss'], 'b-', label='Training Loss')
    if 'val_loss' in history and len(history['val_loss']) > 0:
        ax.plot(epochs, history['val_loss'], 'r-', label='Validation Loss')
    ax.set_title('Loss over Epochs')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Loss')
    ax.legend()
    
    # Plot GZSL and vision loss components
    ax = axes[0, 1]
    ax.plot(epochs, history['train_gzsl_loss'], 'g-', label='GZSL Loss')
    ax.plot(epochs, history['train_vision_loss'], 'm-', label='Vision Loss')
    ax.set_title('Loss Components over Epochs')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Loss')
    ax.legend()
    
    # Plot validation metrics if available
    ax = axes[1, 0]
    if 'val_accuracy' in history and len(history['val_accuracy']) > 0:
        ax.plot(epochs, history['val_accuracy'], 'c-', label='Validation Accuracy')
    ax.set_title('Validation Accuracy')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Accuracy')
    if ax.get_legend_handles_labels()[0]:  # Check if we have any labels
        ax.legend()
    
    # Plot learning rate if available
    ax = axes[1, 1]
    if 'learning_rate' in history and len(history['learning_rate']) > 0:
        ax.plot(epochs, history['learning_rate'], 'r-')
        ax.set_title('Learning Rate over Epochs')
        ax.set_xlabel('Epochs')
        ax.set_ylabel('Learning Rate')
    else:
        # If learning rate isn't available, plot something else or leave blank
        ax.set_title('Reserved for Future Metrics')
    
    # Adjust layout and save the figure
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_metrics.png'), dpi=300)
    plt.savefig(os.path.join(output_dir, 'training_metrics.pdf'))
    plt.close()
    
    # Create separate figure for geographical accuracy metrics if they exist
    geo_metrics = []
    if 'country_accuracy' in history and len(history['country_accuracy']) > 0:
        geo_metrics.append(('country_accuracy', 'Country Accuracy', 'C0'))
    if 'city_accuracy' in history and len(history['city_accuracy']) > 0:
        geo_metrics.append(('city_accuracy', 'City Accuracy', 'C1'))
    if 'continent_accuracy' in history and len(history['continent_accuracy']) > 0:
        geo_metrics.append(('continent_accuracy', 'Continent Accuracy', 'C2'))
    
    if geo_metrics:
        plt.figure(figsize=(10, 6))
        for metric_key, metric_label, color in geo_metrics:
            plt.plot(epochs, history[metric_key], color=color, label=metric_label)
        
        plt.title('Geographical Accuracy Metrics')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'geographical_accuracy.png'), dpi=300)
        plt.savefig(os.path.join(output_dir, 'geographical_accuracy.pdf'))
        plt.close()
    
    print(f"Training metrics plots saved to {output_dir}")

def plot_evaluation_metrics(
    metrics: Dict[str, float],
    output_dir: str,
    figsize: tuple = (10, 6)
) -> None:
    """
    Plot evaluation metrics and save the plots to the output directory.
    
    Args:
        metrics: Dictionary containing evaluation metrics
        output_dir: Directory to save the plots
        figsize: Size of the figure
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Set up plot style
    plt.style.use('ggplot')
    
    # Create a bar plot for evaluation metrics
    fig, ax = plt.subplots(figsize=figsize)
    
    # Filter out metrics that are not accuracy-related
    accuracy_metrics = {k: v for k, v in metrics.items() if 'accuracy' in k.lower() or 'recall' in k.lower() or 'precision' in k.lower()}
    
    if not accuracy_metrics:
        print("No accuracy metrics found for plotting.")
        return
    
    # Sort metrics by value
    sorted_metrics = sorted(accuracy_metrics.items(), key=lambda x: x[1])
    metric_names, metric_values = zip(*sorted_metrics)
    
    # Create bar plot
    bars = ax.barh(metric_names, metric_values, color='skyblue')
    
    # Add value labels on the bars
    for bar in bars:
        width = bar.get_width()
        label_x_pos = width * 1.01
        ax.text(label_x_pos, bar.get_y() + bar.get_height()/2, f'{width:.4f}',
                va='center')
    
    ax.set_title('Evaluation Metrics')
    ax.set_xlabel('Value')
    
    # Adjust layout and save the figure
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'evaluation_metrics.png'), dpi=300)
    plt.savefig(os.path.join(output_dir, 'evaluation_metrics.pdf'))
    plt.close()
    
    # Create separate figures for geographical accuracy
    geo_categories = {
        'continent': [k for k in metrics.keys() if 'continent' in k.lower()],
        'country': [k for k in metrics.keys() if 'country' in k.lower()], 
        'city': [k for k in metrics.keys() if 'city' in k.lower()]
    }
    
    for category, metrics_keys in geo_categories.items():
        if not metrics_keys:
            continue
            
        category_metrics = {k: metrics[k] for k in metrics_keys if k in metrics}
        if not category_metrics:
            continue
            
        plt.figure(figsize=(8, 5))
        sorted_items = sorted(category_metrics.items(), key=lambda x: x[1])
        labels, values = zip(*sorted_items)
        
        # Clean up labels for display
        display_labels = [label.replace('_', ' ').replace('accuracy', '').strip() for label in labels]
        if all(label == '' for label in display_labels):
            display_labels = [label.replace('_', ' ') for label in labels]
            
        # Create bar chart
        bars = plt.barh(display_labels, values, color=f'C{list(geo_categories.keys()).index(category)}')
        
        # Add value labels
        for bar in bars:
            width = bar.get_width()
            plt.text(width * 1.01, bar.get_y() + bar.get_height()/2, f'{width:.4f}', va='center')
            
        plt.title(f'{category.capitalize()} Level Accuracy')
        plt.xlabel('Accuracy')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{category}_accuracy.png'), dpi=300)
        plt.savefig(os.path.join(output_dir, f'{category}_accuracy.pdf'))
        plt.close()
    
    print(f"Evaluation metrics plots saved to {output_dir}") 