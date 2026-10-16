import matplotlib.pyplot as plt


def plot_wafer(wafer, title: str):
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(wafer, cmap="gray", vmin=0, vmax=2)
    ax.set_title(title)
    ax.axis("off")
    fig.tight_layout()
    return fig


def plot_gradcam(normalized_wafer, cam, title: str):
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(normalized_wafer, cmap="gray", vmin=0, vmax=1)
    ax.imshow(cam, cmap="jet", alpha=0.45, vmin=0, vmax=1)
    ax.set_title(title)
    ax.axis("off")
    fig.tight_layout()
    return fig
