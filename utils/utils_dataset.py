from transforms import get_train_transform, get_val_transform, IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
import numpy as np
import matplotlib.pyplot as plt

def plot_image_and_masks(transformed):
    image = transformed["image"].squeeze(0).permute(1, 2, 0).cpu().numpy()  # [H, W, C]
    masks = {k: v.squeeze(0).cpu().numpy() for k, v in transformed.items() if k.startswith("mask") or "mask" in k}
    color_names = ["light_brown", "dark_brown", "black", "blue_gray", "red", "white"]

    # Normalize
    mean = np.array(IMAGENET_DEFAULT_MEAN)
    std = np.array(IMAGENET_DEFAULT_STD)
    image = (image * std + mean).clip(0, 1)

    n_masks = 8
    fig, axes = plt.subplots(1, n_masks + 1, figsize=(4*(n_masks+1), 4))

    # Plot image
    axes[0].imshow(image)
    axes[0].set_title("Image")
    axes[0].axis("off")

    # Plot masks
    for i, (name, mask) in enumerate(masks.items(), start=1):
        if mask.shape[0] == 6:
            for j in range(mask.shape[0]):
                axes[j+1].imshow(mask[j], cmap="gray")
                axes[j+1].set_title(color_names[j])
                axes[j+1].axis("off")
        else:
            axes[i+j].imshow(mask, cmap="gray")
            axes[i+j].set_title(name)
            axes[i+j].axis("off")

    plt.savefig("example.png")