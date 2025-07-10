import pandas as pd
import os
from collections import defaultdict

# مسیر فایل‌ها
CSV_PATH = "xHR_expData_followUp_gender.csv"             # مسیر فایل CSV ورودی
IMAGE_FOLDER = "ImageStimuli"     # مسیر فولدر عکس‌ها
OUTPUT_CSV_PATH = "final_output.csv"    # مسیر فایل خروجی



# خواندن فایل CSV
df = pd.read_csv(CSV_PATH)


df.insert(0, "id", range(1, len(df) + 1))

# لیست فایل‌های تصویر
all_images = [f for f in os.listdir(IMAGE_FOLDER) if f.endswith(".png")]

# ساخت دیکشنری: key = gender_ethnicity_glasses → value = [تصاویر مناسب]
image_pool = defaultdict(list)

for img in all_images:
    try:
        parts = img.split("_")
        if len(parts) != 4:
            continue  # رد فایل‌های خراب

        gender = parts[0].upper()
        ethnicity = parts[1].upper()
        glasses = "1" if "GL" in parts[3].upper() else "0"
        key = f"{gender}_{ethnicity}_{glasses}"
        image_pool[key].append(img)
    except Exception as e:
        print(f"خطا در پردازش {img}: {e}")

# شمارنده استفاده از عکس
image_use_counter = defaultdict(int)

# تابع برای ساخت نام فایل مناسب
def get_image_filename(gender, ethnicity, glasses):
    gender = gender.upper()
    ethnicity = ethnicity.upper()
    glasses = str(glasses)
    key = f"{gender}_{ethnicity}_{glasses}"

    if key not in image_pool or len(image_pool[key]) == 0:
        raise ValueError(f"No images found for key: {key}")

    # سعی کن عکس تکراری استفاده نشه
    index = image_use_counter[key] % len(image_pool[key])
    image_use_counter[key] += 1
    return image_pool[key][index]

# ساخت ستون image
df["image"] = df.apply(lambda row: get_image_filename(row["gender"], row["ethnicity"], row["glasses"]), axis=1)

# ذخیره فایل نهایی
df.to_csv(OUTPUT_CSV_PATH, index=False)
print(f"✅ تمام شد! فایل خروجی ذخیره شد در: {OUTPUT_CSV_PATH}")