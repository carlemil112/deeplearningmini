import cv2 
import numpy as np
import os 
import matplotlib.pyplot as plt


def analyze_dataset(image_dir):
    class_counts = {}
    image_sizes = []

    for class_name in os.listdir(image_dir):
        class_path = os.path.join(image_dir, class_name)
        if os.path.isdir(class_path):
            images = os.listdir(class_path)
            class_counts[class_name] = len(images)

            for img_name in images:
                img_path = os.path.join(class_path, img_name)
                img = cv2.imread(img_path)
                if img is not None:
                    image_sizes.append(img.shape[:2]) # (height, width)
                else:
                    print(f"Warning: Could not read image '{img_path}'")

    return class_counts, image_sizes
    cv2.imwrite(f'augmented_image_{i}.png', img)


if __name__ == "__main__":
    dataset_path = 'FER-2013/train'
    class_counts, image_sizes = analyze_dataset(dataset_path)

    print("Class Distribution:")
    for class_name, count in class_counts.items():
        print(f"{class_name}: {count} images")

    heights, widths = zip(*image_sizes)
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.hist(heights, bins=30, color='blue', alpha=0.7)
    plt.title('Image Height Distribution')
    plt.xlabel('Height (pixels)')
    plt.ylabel('Frequency')

    plt.subplot(1, 2, 2)
    plt.hist(widths, bins=30, color='green', alpha=0.7)
    plt.title('Image Width Distribution')
    plt.xlabel('Width (pixels)')
    plt.ylabel('Frequency')

    plt.tight_layout()
    plt.show()
    