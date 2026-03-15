# AI_machine_AI_machine_separates_fish
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import numpy as np
import os

# ตั้งค่า Path ของ Dataset
base_dir = '/content/drive/MyDrive/Colab Notebooks/content/fishfolder'

# ตรวจสอบว่า Path มีอยู่จริงหรือไม่
if not os.path.exists(base_dir):
    raise FileNotFoundError(f"ไม่พบไดเร็กทอรี: {base_dir}. กรุณาตรวจสอบว่าคุณได้ mount Google Drive และโฟลเดอร์ 'fish_dataset' อยู่ในตำแหน่งที่ถูกต้องใน Google Drive ของคุณ")

# เตรียม Data Generator พร้อมทำ Data Augmentation (ช่วยให้โมเดลแม่นยำขึ้น)
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,      # หมุนรูป
    width_shift_range=0.2,  # เลื่อนรูปซ้าย-ขวา
    height_shift_range=0.2, # เลื่อนรูปบน-ล่าง
    horizontal_flip=True,   # พลิกรูปซ้ายขวา
    validation_split=0.2    # แบ่ง validation 20%
)

train_gen = train_datagen.flow_from_directory(
    base_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical',
    subset='training'
)

val_gen = train_datagen.flow_from_directory(
    base_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical',
    subset='validation'
)

from google.colab import drive
drive.mount('/content/drive')

model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    layers.MaxPooling2D(2, 2),

    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),

    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),

    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5), # สุ่มปิด Neuron 50% ป้องกัน Overfitting
    layers.Dense(train_gen.num_classes, activation='softmax')
])

model.compile(optimizer='adam', 
              loss='categorical_crossentropy', 
              metrics=['accuracy'])

# สรุปโครงสร้างโมเดล
model.summary()

# ฝึกฝนโมเดลด้วยชุดข้อมูลสำหรับฝึก (train_gen) และประเมินผลด้วยชุดข้อมูลสำหรับตรวจสอบ (val_gen) เป็นจำนวน 20 รอบ (epochs)
history = model.fit(train_gen, validation_data=val_gen, epochs=20)

# กราฟแสดงผล
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.title('Accuracy Improvement')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Loss Reduction')
plt.legend()
plt.show()


# ติดตั้งฟอนต์ภาษาไทย (เช่น Sarabun)
!pip install fonttools
!wget -q https://github.com/google/fonts/raw/main/ofl/sarabun/Sarabun-Regular.ttf -O /usr/local/lib/python3.12/dist-packages/matplotlib/mpl-data/fonts/ttf/Sarabun-Regular.ttf

# กำหนดค่า Matplotlib ให้ใช้ฟอนต์ Sarabun
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# ลบแคชฟอนต์เก่าของ Matplotlib
fm._load_fontmanager(try_read_cache=False)

# เพิ่มฟอนต์ Sarabun เข้าไปใน Matplotlib
font_dirs = ['/usr/local/lib/python3.12/dist-packages/matplotlib/mpl-data/fonts/ttf/']
font_files = fm.findSystemFonts(fontpaths=font_dirs)
for font_file in font_files:
    fm.fontManager.addfont(font_file)

# กำหนดฟอนต์เริ่มต้นเป็น Sarabun
plt.rcParams['font.family'] = 'Sarabun'

print("ตั้งค่าฟอนต์ Sarabun สำหรับ Matplotlib เรียบร้อยแล้ว")

# สร้าง Dictionary สำหรับเก็บข้อมูลปลาแต่ละสายพันธุ์
fish_info = {
    'ปลาฉลาม': {
        'ชื่อสามัญ': 'ปลาฉลาม',
        'ชื่อวิทยาศาสตร์': 'Selachimorpha',
        'คำอธิบาย': 'ปลาฉลามเป็นปลากระดูกอ่อนในชั้น Chondrichthyes ลักษณะเด่นคือมีกระดูกอ่อนและไม่มีกระดูกแข็ง มีความหลากหลายในขนาดและถิ่นที่อยู่ พบได้ทั้งในน้ำจืดและน้ำเค็ม',
        'ถิ่นที่อยู่': 'มหาสมุทรทั่วโลก',
        'อาหาร': 'สัตว์น้ำชนิดอื่น เช่น ปลา, แมวน้ำ, และสัตว์ไม่มีกระดูกสันหลัง'
    },
    'ปลาดุก': {
        'ชื่อสามัญ': 'ปลาดุก',
        'ชื่อวิทยาศาสตร์': 'Clarias batrachus',
        'คำอธิบาย': 'ปลาดุกเป็นปลาน้ำจืดจำพวกปลาไม่มีเกล็ด มีหนวดยาว 4 คู่ ลำตัวลื่น รูปร่างกลมยาว หัวแบน',
        'ถิ่นที่อยู่': 'แหล่งน้ำจืดในเอเชียและแอฟริกา',
        'อาหาร': 'ซากสัตว์, แมลง, พืชน้ำ, สัตว์น้ำขนาดเล็ก'
    },
    'ปลาทับทิม': {
        'ชื่อสามัญ': 'ปลาทับทิม',
        'ชื่อวิทยาศาสตร์': 'Oreochromis niloticus',
        'คำอธิบาย': 'ปลาทับทิมเป็นปลาน้ำจืดที่พัฒนาสายพันธุ์มาจากปลานิล มีเนื้อแน่น สีสวย รสชาติดี และเป็นที่นิยมบริโภค',
        'ถิ่นที่อยู่': 'แหล่งน้ำจืดทั่วโลก (ถูกเพาะเลี้ยง)',
        'อาหาร': 'พืชน้ำ, แพลงก์ตอน, อาหารเม็ด'
    },
    'ปลาปักเป้า': {
        'ชื่อสามัญ': 'ปลาปักเป้า',
        'ชื่อวิทยาศาสตร์': 'Tetraodontidae',
        'คำอธิบาย': 'ปลาปักเป้าเป็นปลาที่มีรูปร่างกลมป้อม มีความสามารถในการพองตัวเมื่อถูกคุกคาม ผิวหนังมีพิษ โดยเฉพาะในเครื่องในบางชนิด',
        'ถิ่นที่อยู่': 'น้ำจืด, น้ำกร่อย, น้ำเค็มในเขตร้อนและอบอุ่น',
        'อาหาร': 'สัตว์ไม่มีกระดูกสันหลังขนาดเล็ก, สาหร่าย, เศษซาก'
    },
    'ปลากัด': {
        'ชื่อสามัญ': 'ปลากัด',
        'ชื่อวิทยาศาสตร์': 'Betta splendens',
        'คำอธิบาย': 'ปลากัดเป็นปลาน้ำจืดขนาดเล็กที่สวยงาม มีถิ่นกำเนิดในประเทศไทย มีสีสันสดใส และมีพฤติกรรมดุดันเมื่อเจอปลากัดตัวอื่น',
        'ถิ่นที่อยู่': 'แหล่งน้ำจืดในเอเชียตะวันออกเฉียงใต้ (โดยเฉพาะประเทศไทย)',
        'อาหาร': 'ลูกน้ำ, แมลงขนาดเล็ก, อาหารสำเร็จรูป'
    },
    'ปลาช่อน': {
        'ชื่อสามัญ': 'ปลาช่อน',
        'ชื่อวิทยาศาสตร์': 'Channa striata',
        'คำอธิบาย': 'ปลาช่อนเป็นปลาน้ำจืดที่พบได้ทั่วไปในภูมิภาคเอเชียตะวันออกเฉียงใต้ มีลำตัวยาวกลมเรียว หัวโต ปากกว้าง มีฟันแหลมคม',
        'ถิ่นที่อยู่': 'แหล่งน้ำจืดในเอเชียตะวันออกเฉียงใต้',
        'อาหาร': 'ปลาเล็ก, กบ, ลูกอ๊อด, สัตว์น้ำขนาดเล็กอื่นๆ'
    }
}

def predict_fish(img_path):
    # โหลดรูป
    img = tf.keras.preprocessing.image.load_img(img_path, target_size=(150, 150))
    img_array = tf.keras.preprocessing.image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # ทำนา
    prediction = model.predict(img_array)
    class_idx = np.argmax(prediction)
    confidence = np.max(prediction) * 100
    class_labels = list(train_gen.class_indices.keys())
    result_label = class_labels[class_idx]

    # แสดงผล
    plt.imshow(img)
    plt.title(f"Predict: {result_label} ({confidence:.2f}%)")
    plt.axis('off')
    plt.show()

    return result_label, confidence

# ฟังก์ชันสำหรับแสดงข้อมูลของปลาที่ทำนายได้
def display_fish_details(predicted_fish_name):
    if predicted_fish_name in fish_info:
        print(f"\n--- ข้อมูลของ {predicted_fish_name} ---")
        for key, value in fish_info[predicted_fish_name].items():
            print(f"{key}: {value}")
    else:
        print(f"ไม่พบข้อมูลสำหรับสายพันธุ์: {predicted_fish_name}")

# ทดสอบและแสดงข้อมูล
result, conf = predict_fish('/content/drive/MyDrive/Image/ปลา7.jpg')
display_fish_details(result)



