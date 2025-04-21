import os
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple


def plot_training_metrics(
    history: Dict[str, List[float]],
    output_dir: str,
    figsize: Tuple[int, int] = (12, 9)
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    plt.style.use('ggplot') 

    epochs = range(1, len(history['train_loss']) + 1)
    fig, axes = plt.subplots(2, 2, figsize=figsize)

    ax = axes[0, 0]
    ax.plot(epochs, history['train_loss'], label='Train Loss')
    if history.get('val_loss'):
        ax.plot(epochs, history['val_loss'], label='Val Loss')
    ax.set(xlabel='Epochs', ylabel='Loss', title='Loss over Epochs')
    ax.legend()

    ax = axes[0, 1]
    ax.plot(epochs, history['train_gzsl_loss'], label='GZSL Loss')
    ax.plot(epochs, history['train_vision_loss'], label='Vision Loss')
    ax.set(xlabel='Epochs', ylabel='Loss', title='Loss Components')
    ax.legend()

    ax = axes[1, 0]
    if history.get('val_accuracy'):
        ax.plot(epochs, history['val_accuracy'], label='Val Accuracy')
        ax.set(xlabel='Epochs', ylabel='Accuracy', title='Val Accuracy')
        ax.legend()
    else:
        ax.set(title='Validation Accuracy (n/a)')

    ax = axes[1, 1]
    if history.get('learning_rate'):
        ax.plot(epochs, history['learning_rate'], label='LR')
        ax.set(xlabel='Epochs', ylabel='Learning Rate', title='Learning Rate')
    else:
        ax.set(title='Reserved')

    fig.tight_layout()
    for ext in ('png', 'pdf'):
        fig.savefig(os.path.join(output_dir, f'training_metrics.{ext}'), dpi=300)
    plt.close(fig)

    # plot geographic metrics if present
    geo_keys = [
        ('country_accuracy', 'Country'),
        ('city_accuracy', 'City'),
        ('continent_accuracy', 'Continent')
    ]
    geo = [(k, lbl) for k, lbl in geo_keys if history.get(k)]
    if geo:
        fig, ax = plt.subplots(figsize=(10, 6))
        for key, label in geo:
            ax.plot(epochs, history[key], label=label)
        ax.set(xlabel='Epochs', ylabel='Accuracy', title='Geographical Accuracy')
        ax.legend()
        fig.tight_layout()
        for ext in ('png', 'pdf'):
            fig.savefig(os.path.join(output_dir, f'geographical_accuracy.{ext}'), dpi=300)
        plt.close(fig)

    print(f"Training metrics plots saved to {output_dir}")


def plot_evaluation_metrics(
    metrics: Dict[str, float],
    output_dir: str,
    figsize: Tuple[int, int] = (10, 6)
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    plt.style.use('ggplot') 

    # filter metrics to only include accuracy-like entries 
    acc_metrics = {
        k: v for k, v in metrics.items()
        if any(term in k.lower() for term in ('accuracy', 'precision', 'recall'))
    }
    if not acc_metrics:
        print("No accuracy metrics to plot.")
        return

    names, values = zip(*sorted(acc_metrics.items(), key=lambda x: x[1]))
    fig, ax = plt.subplots(figsize=figsize)
    bars = ax.barh(names, values)
    for bar in bars:
        ax.text(bar.get_width() * 1.01, bar.get_y() + bar.get_height() / 2,
                f'{bar.get_width():.4f}', va='center')
    ax.set(xlabel='Value', title='Evaluation Metrics')
    fig.tight_layout()
    for ext in ('png', 'pdf'):
        fig.savefig(os.path.join(output_dir, f'evaluation_metrics.{ext}'), dpi=300)
    plt.close(fig)

    # group and plot geographic accuracy by category based on key substring
    for category in ('continent', 'country', 'city'):
        keys = [k for k in acc_metrics if category in k.lower()]
        if not keys:
            continue
        items = {k: metrics[k] for k in keys}
        labels, vals = zip(*sorted(items.items(), key=lambda x: x[1]))

        display = [lbl.replace('_', ' ').title() for lbl in labels]
        fig, ax = plt.subplots(figsize=(8, 5))
        bars = ax.barh(display, vals)
        for bar in bars:
            ax.text(bar.get_width() * 1.01, bar.get_y() + bar.get_height() / 2,
                    f'{bar.get_width():.4f}', va='center')
        ax.set(xlabel='Accuracy', title=f'{category.title()} Accuracy')
        fig.tight_layout()
        for ext in ('png', 'pdf'):
            fig.savefig(os.path.join(output_dir, f'{category}_accuracy.{ext}'), dpi=300)
        plt.close(fig)

    print(f"Evaluation metrics plots saved to {output_dir}")
