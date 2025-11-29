import os
import cv2
from tqdm import tqdm

def images_to_video(img_folder, out_video, fps=10, size=(224,224)):
    """
    Convert a folder of images into a .mp4 video.
    """
    images = sorted([f for f in os.listdir(img_folder) if f.lower().endswith(('.jpg','.png'))])
    if not images:
        raise RuntimeError(f"No images found in {img_folder}")

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(out_video, fourcc, fps, size)

    for img_name in tqdm(images, desc=f"Writing {out_video}"):
        img_path = os.path.join(img_folder, img_name)
        img = cv2.imread(img_path)
        if img is None:
            continue
        img = cv2.resize(img, size)
        out.write(img)

    out.release()
    print(f"✅ Saved video: {out_video} ({len(images)} frames)")

def main():
    # Updated folder paths
    categories = {
        "fall_dataset": "data/raw/fall_dataset/images/train",
        "violence": "data/raw/violence",
        "normal": "data/raw/normal"
    }

    out_dir = "data/test_clips"
    os.makedirs(out_dir, exist_ok=True)

    for cat, img_folder in categories.items():
        out_video = os.path.join(out_dir, f"{cat}_01.mp4")
        if not os.path.exists(img_folder):
            print(f"⚠️ Skipping {cat}, folder not found: {img_folder}")
            continue
        images_to_video(img_folder=img_folder, out_video=out_video, fps=10, size=(224,224))

if __name__ == "__main__":
    main()
