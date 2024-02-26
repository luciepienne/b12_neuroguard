import matplotlib.pyplot as plt
import cv2


def plot_images(X, y, n):
    import math

    yes_images = X[y == 1][:n]
    no_images = X[y == 0][:n]

    n_rows = int(math.sqrt(n))
    n_cols = math.ceil(n / n_rows)

    fig_yes, ax_yes = plt.subplots(n_rows, n_cols, figsize=(12, 10))
    fig_yes.suptitle('Class: Yes')

    for i in range(min(n, len(yes_images))):
        ax_yes[i // n_cols, i % n_cols].imshow(yes_images[i])

    fig_no, ax_no = plt.subplots(n_rows, n_cols, figsize=(12, 10))
    fig_no.suptitle('Class: No')

    for i in range(min(n, len(no_images))):
        ax_no[i // n_cols, i % n_cols].imshow(no_images[i])

    plt.show()