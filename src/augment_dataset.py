import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img, array_to_img
from tqdm import tqdm

# Paths to your small datasets
base_dir = "data/raw/fall_dataset/images/train"
violence_dir = "data/raw/violence"
normal_dir = "data/raw/normal"

# Output directories for augmented images
out_violence = "data/augmented/violence"
out_normal = "data/augmented/normal"


os.makedirs(out_violence, exist_ok=True)
os.makedirs(out_normal, exist_ok=True)

# Configure augmentation
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    brightness_range=[0.8, 1.2],
    fill_mode='nearest'
)

def augment_images(input_dir, output_dir, target_count=100):
    images = [f for f in os.listdir(input_dir) if f.lower().endswith(('.jpg','.png','.jpeg'))]
    if not images:
        print(f"No images found in {input_dir}")
        return

    print(f"\nAugmenting {input_dir} -> {output_dir}")
    total = 0

    for img_name in tqdm(images):
        img_path = os.path.join(input_dir, img_name)
        img = load_img(img_path)
        x = img_to_array(img)
        x = x.reshape((1,) + x.shape)

        i = 0
        for batch in datagen.flow(x, batch_size=1, save_to_dir=output_dir,
                                  save_prefix="aug", save_format="jpg"):
            i += 1
            total += 1
            if i >= target_count // len(images):  # stop after enough samples
                break
    print(f"✅ Created {total} images in {output_dir}")

if __name__ == "__main__":
    augment_images(violence_dir, out_violence)
    augment_images(normal_dir, out_normal)
