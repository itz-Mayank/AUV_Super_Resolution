import os
import cv2

hr_dir = r"C:\Users\Mayank Meghwal\Desktop\Reserch Papers AUV\AUV\Code\Data\UFO-120\train_val\hr"
lr2x_dir = r"C:\Users\Mayank Meghwal\Desktop\Reserch Papers AUV\AUV\Code\Data\UFO-120\train_val\lr_2x"

os.makedirs(lr2x_dir, exist_ok=True)

files = sorted(os.listdir(hr_dir))

print("Generating lr_2x images...")

for f in files:
    hr_path = os.path.join(hr_dir, f)
    img = cv2.imread(hr_path)

    if img is None:
        print("Skipping (cannot read):", f)
        continue

    h, w = img.shape[:2]

    # SRDRM expects 320×240 LR for 640×480 HR
    lr_img = cv2.resize(img, (w // 2, h // 2), interpolation=cv2.INTER_CUBIC)

    save_path = os.path.join(lr2x_dir, f)
    cv2.imwrite(save_path, lr_img)

    print("Saved:", save_path)

print("\nAll lr_2x images generated successfully!")
