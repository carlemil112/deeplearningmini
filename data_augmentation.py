import numpy as np
import cv2

def augment_image(image):
    h, w = image.shape[:2]
    img = image

    # Random rotation (små vinkler er bedre til ansigter)
    angle = np.random.uniform(-15, 15)
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    img = cv2.warpAffine(
        img, M, (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101
    )

    # Kun horisontal flip (p=0.5)
    if np.random.rand() < 0.5:
        img = cv2.flip(img, 1)

    # Let zoom (bevar ansigtsgeometri)
    zoom = np.random.uniform(0.9, 1.1)
    if zoom < 1.0:
        # zoom ind: center-crop -> resize
        new_w, new_h = int(w * zoom), int(h * zoom)
        cx, cy = w // 2, h // 2
        x1, x2 = cx - new_w // 2, cx + new_w // 2
        y1, y2 = cy - new_h // 2, cy + new_h // 2
        img = img[max(0,y1):min(h,y2), max(0,x1):min(w,x2)]
        img = cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)
    else:
        # zoom ud: skalér ned og pad tilbage
        sw, sh = int(w / zoom), int(h / zoom)
        scaled = cv2.resize(img, (sw, sh), interpolation=cv2.INTER_LINEAR)
        top = (h - sh) // 2
        left = (w - sw) // 2
        img = cv2.copyMakeBorder(
            scaled,
            top, h - top - sh, left, w - left - sw,
            borderType=cv2.BORDER_REFLECT_101
        )

    return img

def augment_dataset(image_paths):
    out = []
    for p in image_paths:
        img = cv2.imread(p, cv2.IMREAD_UNCHANGED)
        if img is None:
            print(f"Advarsel: kunne ikke læse '{p}', springer over.")
            continue
        out.append(augment_image(img))
    return out

if __name__ == "__main__":
    image_paths = [
        'FER-2013/train/surprise'
    ]
    augmented_images = augment_dataset(image_paths)
    for i, img in enumerate(augmented_images):
        ok = cv2.imwrite(f'augmented_image_{i}.jpg', img)
        if not ok:
            print(f"Kunne ikke gemme augmented_image_{i}.jpg")
    print("Augmented images saved successfully.")