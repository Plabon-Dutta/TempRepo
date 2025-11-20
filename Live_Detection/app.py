import os
import cv2
import numpy as np
from flask import Flask, request, render_template, send_file
from ultralytics import YOLO
import tensorflow as tf

# ============= Configs =============
UPLOAD_FOLDER = 'uploads'
RESULT_FOLDER = 'results'

OBJ_MODEL_WEIGHTS = 'Yolo weights/yolo_multihead_model.pt'
ANOM_MODEL_WEIGHTS = 'Yolo weights/best (1).pt'
OBJ_CLASSES_TXT = 'Yolo weights/objects28_classes.txt'
IMG_SIZE = 224

CNN_MODEL_PATH = 'multihead_combined_model.h5'
CNN_CLASS_NAMES_TXT = 'class_names.txt'

ALLOWED_IMAGE_EXTENSIONS = {'png', 'jpg', 'jpeg'}
ALLOWED_VIDEO_EXTENSIONS = {'mp4', 'avi', 'mov'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULT_FOLDER'] = RESULT_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

# ============= Load YOLO Models and Classes =============
obj_model = YOLO(OBJ_MODEL_WEIGHTS)
anom_model = YOLO(ANOM_MODEL_WEIGHTS)
with open(OBJ_CLASSES_TXT, 'r') as f:
    object_classes = [line.strip() for line in f if line.strip()]
hazmat_classes = set([c for c in object_classes if 'hazmat' in c.lower() or 'Hazmat' in c])

# ============= Load CNN Model and Classes =============
cnn_model = tf.keras.models.load_model(CNN_MODEL_PATH)
with open(CNN_CLASS_NAMES_TXT, 'r') as f:
    cnn_class_names = [line.strip() for line in f.readlines()]

def allowed_file(filename, allowed_set):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_set

def preprocess_image_for_yolo(frame):
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    return img

def make_yolo_prediction(image_path, model_choice):
    # 1. Anomaly Detection
    anom_result = anom_model.predict(image_path, imgsz=IMG_SIZE, save=False, verbose=False)[0]
    anom_probs = anom_result.probs.data.cpu().numpy()
    anom_idx = int(np.argmax(anom_probs))
    anom_label = list(anom_result.names.values())[anom_idx]
    is_anomaly = (anom_label.lower() == 'anomaly')

    # 2. Object Classification
    obj_result = obj_model.predict(image_path, imgsz=IMG_SIZE, save=False, verbose=False)[0]
    obj_probs = obj_result.probs.data.cpu().numpy()
    obj_idx = int(np.argmax(obj_probs))
    obj_label = object_classes[obj_idx] if obj_idx < len(object_classes) else "unknown"
    obj_conf = float(obj_probs[obj_idx])

    # Hazmat override logic
    if obj_label in hazmat_classes:
        is_anomaly = False

    # Model choice logic
    if model_choice == "object":
        label = f"Detected: {obj_label} (Object)"
        color = (0, 255, 0)
    elif model_choice == "anomaly":
        label = f"Detected: {'Anomaly' if is_anomaly else obj_label} ({'Defective' if is_anomaly else 'Good'})"
        color = (0, 0, 255) if is_anomaly else (0, 255, 0)
    else:  # both
        if obj_label in hazmat_classes:
            label = f"Detected: {obj_label} (Hazmat)"
            color = (255, 127, 0)
        else:
            if is_anomaly:
                label = f"Detected: {obj_label} (Defective)"
                color = (0, 0, 255)
            else:
                label = f"Detected: {obj_label} (Good)"
                color = (0, 255, 0)

    return label, color

def make_cnn_prediction(image_path, model_choice):
    frame = cv2.imread(image_path)
    disp_w = 640
    disp_h = int(frame.shape[0] * disp_w / frame.shape[1])
    frame_disp = cv2.resize(frame, (disp_w, disp_h))
    # Preprocess for model
    img = cv2.cvtColor(frame_disp, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img.astype(np.float32) / 255.0
    input_img = np.expand_dims(img, axis=0)
    class_probs, anomaly_probs = cnn_model.predict(input_img)
    pred_idx = np.argmax(class_probs[0])
    pred_class = cnn_class_names[pred_idx] if pred_idx < len(cnn_class_names) else "UNKNOWN"
    anomaly_score = anomaly_probs[0][0]

    def parse_class_label(pred_class):
        if pred_class.startswith('anomaly_'):
            splits = pred_class.split('_')
            obj = splits[1].capitalize()
            return obj
        elif pred_class.startswith('hazmat_'):
            obj = pred_class.replace('hazmat_', '').replace('-', ' ').replace('_', ' ').title()
            return obj
        else:
            return pred_class

    obj_name = parse_class_label(pred_class)

    # Model choice logic
    if model_choice == "object":
        label = f"Detected: {obj_name} (Object)"
        color = (0, 255, 0)
    elif model_choice == "anomaly":
        label = f"Detected: {obj_name} ({'Defective' if anomaly_score > 0.00005 else 'Good'})"
        color = (0, 0, 255) if anomaly_score > 0.00005 else (0, 255, 0)
    else:  # both
        if pred_class.startswith("hazmat_"):
            label = f"Detected: {obj_name} (Hazmat)"
            color = (255, 127, 0)
        else:
            if anomaly_score > 0.00005:
                label = f"Detected: {obj_name} (Defective)"
                color = (0, 0, 255)
            else:
                label = f"Detected: {obj_name} (Good)"
                color = (0, 255, 0)
    return label, color

def draw_label_with_background(img, text, pos, font=cv2.FONT_HERSHEY_SIMPLEX, font_scale=1.1, font_thickness=3, text_color=(255, 127, 0), bg_color=(0, 0, 0)):
    x, y = pos
    (w, h), baseline = cv2.getTextSize(text, font, font_scale, font_thickness)
    x = max(10, min(img.shape[1] - w - 10, x))
    y = max(h + 10, min(img.shape[0] - 10, y))
    cv2.rectangle(img, (x-2, y-h-2), (x+w+2, y+baseline+2), bg_color, -1)
    cv2.putText(img, text, (x, y), font, font_scale, text_color, font_thickness, lineType=cv2.LINE_AA)

def detect_and_annotate(image_path, result_path, model_type, model_choice):
    frame = cv2.imread(image_path)
    disp_w = 840
    disp_h = int(frame.shape[0] * disp_w / frame.shape[1])
    frame_disp = cv2.resize(frame, (disp_w, disp_h))

    if model_type == "yolo":
        label, color = make_yolo_prediction(image_path, model_choice)
    else:
        label, color = make_cnn_prediction(image_path, model_choice)

    h, w = frame_disp.shape[:2]
    cv2.rectangle(frame_disp, (0, 0), (w-1, h-1), color, 4)
    draw_label_with_background(frame_disp, label, (10, 50), font_scale=1.0, font_thickness=2, text_color=color, bg_color=(0,0,0))

    cv2.imwrite(result_path, frame_disp)
    return label

def detect_and_annotate(image_path, result_path, model_type, model_choice):
    frame = cv2.imread(image_path)
    disp_w = 840
    disp_h = int(frame.shape[0] * disp_w / frame.shape[1])
    frame_disp = cv2.resize(frame, (disp_w, disp_h))

    if model_type == "yolo":
        label, color = make_yolo_prediction(image_path, model_choice)
    else:
        label, color = make_cnn_prediction(image_path, model_choice)

    h, w = frame_disp.shape[:2]
    cv2.rectangle(frame_disp, (0, 0), (w-1, h-1), color, 4)
    draw_label_with_background(frame_disp, label, (10, 50), font_scale=1.0, font_thickness=2, text_color=color, bg_color=(0,0,0))

    cv2.imwrite(result_path, frame_disp)
    return label


def process_video(video_path, result_video_path, model_type, model_choice):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    disp_w = 840
    disp_h = int(frame_h * disp_w / frame_w)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(result_video_path, fourcc, fps, (disp_w, disp_h))

    frame_count = 0
    labels = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        temp_img_path = os.path.join(RESULT_FOLDER, f'temp_frame_{frame_count}.jpg')
        cv2.imwrite(temp_img_path, frame)
        label = detect_and_annotate(temp_img_path, temp_img_path, model_type, model_choice)
        labels.append(label)
        processed_frame = cv2.imread(temp_img_path)
        out.write(processed_frame)
        os.remove(temp_img_path)
        frame_count += 1

    cap.release()
    out.release()
    return labels

@app.route('/', methods=['GET', 'POST'])
def index():
    result_text = None
    result_media = None
    is_video = False
    if request.method == 'POST':
        file = request.files['image']
        model_type = request.form.get('model_type', 'yolo')
        model_choice = request.form.get('model_choice', 'both')
        if file:
            filename = file.filename
            ext = filename.rsplit('.', 1)[1].lower()
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            if allowed_file(filename, ALLOWED_IMAGE_EXTENSIONS):
                result_image_path = os.path.join(app.config['RESULT_FOLDER'], f'result_{filename}')
                result_text = detect_and_annotate(filepath, result_image_path, model_type, model_choice)
                result_media = f'result_{filename}'
                is_video = False
            elif allowed_file(filename, ALLOWED_VIDEO_EXTENSIONS):
                result_video_name = f'result_{filename.rsplit(".", 1)[0]}.mp4'
                result_video_path = os.path.join(app.config['RESULT_FOLDER'], result_video_name)
                labels = process_video(filepath, result_video_path, model_type, model_choice)
                result_text = f"Processed {len(labels)} frames. Sample result: {labels[:3]}"
                result_media = result_video_name
                is_video = True
            else:
                result_text = "Unsupported file type."
                result_media = None

            return render_template('result.html', result_media=result_media, result_text=result_text, is_video=is_video)
    return render_template('index.html')

@app.route('/result_media/<filename>')
def result_media(filename):
    ext = filename.rsplit('.', 1)[1].lower()
    mimetype = 'video/mp4' if ext in ALLOWED_VIDEO_EXTENSIONS else 'image/jpeg'
    return send_file(os.path.join(app.config['RESULT_FOLDER'], filename), mimetype=mimetype)

if __name__ == '__main__':
    app.run(debug=True)