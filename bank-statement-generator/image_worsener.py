import os
import cv2
import numpy as np
import random
import glob

# --- CONFIGURATION ---
INPUT_DIR = "bank_statements"
OUTPUT_DIR = "hard_to_read_images"

# Adjust the intensity of the effects here
ROTATION_RANGE = (-5, 5)  # Degrees
PERSPECTIVE_RANGE = 0.05   # Percentage of image width/height for corner shifts
BRIGHTNESS_RANGE = (-30, 30) # Pixel value change
CONTRAST_RANGE = (0.8, 1.2)   # Multiplier
BLUR_KERNEL_RANGE = (3, 5)    # Must be an odd number
TEMP_CHANGE_RANGE = (-20, 20) # Color temperature shift

def augment_image(image):
    """
    Applies a series of random augmentations to an image to make it harder to read.
    """
    h, w = image.shape[:2]
    augmented_image = image.copy()

    # Define a white background for borders created by transformations
    border_color = (255, 255, 255)

    # 1. Perspective Transform (change viewing angle)
    src_points = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    max_shift_x = int(w * PERSPECTIVE_RANGE)
    max_shift_y = int(h * PERSPECTIVE_RANGE)

    dst_points = np.float32([
        [random.randint(0, max_shift_x), random.randint(0, max_shift_y)],
        [w - random.randint(1, max_shift_x), random.randint(0, max_shift_y)],
        [w - random.randint(1, max_shift_x), h - random.randint(1, max_shift_y)],
        [random.randint(0, max_shift_x), h - random.randint(1, max_shift_y)]
    ])

    perspective_matrix = cv2.getPerspectiveTransform(src_points, dst_points)
    augmented_image = cv2.warpPerspective(augmented_image, perspective_matrix, (w, h), borderValue=border_color)

    # 2. Rotation
    angle = random.uniform(ROTATION_RANGE[0], ROTATION_RANGE[1])
    rot_matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1)
    augmented_image = cv2.warpAffine(augmented_image, rot_matrix, (w, h), borderValue=border_color)

    # 3. Brightness & Contrast
    contrast = random.uniform(CONTRAST_RANGE[0], CONTRAST_RANGE[1])
    brightness = random.randint(BRIGHTNESS_RANGE[0], BRIGHTNESS_RANGE[1])
    augmented_image = cv2.convertScaleAbs(augmented_image, alpha=contrast, beta=brightness)

    # 4. Blur (reduce sharpness)
    if random.random() > 0.5: # Apply blur 50% of the time
        kernel_size = random.randrange(BLUR_KERNEL_RANGE[0], BLUR_KERNEL_RANGE[1] + 1, 2)
        augmented_image = cv2.GaussianBlur(augmented_image, (kernel_size, kernel_size), 0)

    # 5. Color Temperature (Wärme)
    temp_change = random.randint(TEMP_CHANGE_RANGE[0], TEMP_CHANGE_RANGE[1])
    if temp_change != 0:
        b, g, r = cv2.split(augmented_image)
        r = np.clip(r.astype(np.int16) + temp_change, 0, 255).astype(np.uint8)
        b = np.clip(b.astype(np.int16) - temp_change, 0, 255).astype(np.uint8)
        augmented_image = cv2.merge((b, g, r))

    return augmented_image

def main():
    """
    Main function to find images, apply augmentations, and save them.
    """
    print("🚀 Starting image augmentation...")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    image_paths = glob.glob(os.path.join(INPUT_DIR, "*.png")) + \
                  glob.glob(os.path.join(INPUT_DIR, "*.jpg")) + \
                  glob.glob(os.path.join(INPUT_DIR, "*.jpeg"))

    if not image_paths:
        print(f"❌ No images found in the '{INPUT_DIR}' folder.")
        return

    print(f"Found {len(image_paths)} images to process.")

    for i, img_path in enumerate(image_paths):
        try:
            image = cv2.imread(img_path)
            if image is None:
                print(f"⚠️ Could not read image: {img_path}")
                continue

            augmented_image = augment_image(image)

            basename = os.path.basename(img_path)
            filename, ext = os.path.splitext(basename)
            output_filename = f"{filename}_aug_{i+1}.png"
            output_path = os.path.join(OUTPUT_DIR, output_filename)

            cv2.imwrite(output_path, augmented_image)
            print(f"✅ Successfully created: {output_path}")

        except Exception as e:
            print(f"❌ Failed to process {img_path}: {e}")

    print(f"\n🎉 Augmentation complete! Check the '{OUTPUT_DIR}' folder.")

if __name__ == "__main__":
    main()
