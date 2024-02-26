import numpy as np
import cv2
import os


def collect_data(directory, tumor_type, database):
    data = []
    filenames = os.listdir(directory)
    for filename in filenames:
        file_path = os.path.join(directory, filename)
        # Assign label based on tumor_type
        label = 1 if tumor_type == 'yes' else 0
        data.append({"img_path": file_path, "tumor": label, "dataset": database})
    return data


def load_data(dir_path):
    """
    Load as np.arrays to workspace
    """
    X = []
    y = []
    labels = {}  # Dictionary to map label indices to class names
    i = 0
    for label in sorted(os.listdir(dir_path)):
        if not label.startswith('.'):
            labels[i] = label  # Map label index to class name
            label_path = os.path.join(dir_path, label)
            for image_file in os.listdir(label_path):
                if not image_file.startswith('.'):
                    image_path = os.path.join(label_path, image_file)
                    img = cv2.imread(image_path)
                    X.append(img)
                    y.append(i)
            i += 1
    X = np.array(X,dtype='object')
    y = np.array(y)
    print(f'{len(X)} images loaded from {dir_path} directory.')
    return X, y, labels