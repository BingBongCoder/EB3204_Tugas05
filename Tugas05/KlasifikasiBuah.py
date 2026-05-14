import cv2
import numpy as np
import pickle
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

# Loading Model Random Forest
try:
    with open('model_buah_RF.pkl', 'rb') as f:
        package = pickle.load(f)
    
    model = package['model']
    sc = package['scaler']
    indices_h = package['selected_indices']
    class_names = package['class_names']
except FileNotFoundError:
    print("Error: File 'model_buah_RF.pkl' tidak ditemukan di folder ini!")
    exit()

def extract_features(roi):
    
    # Konversi Warna
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    # Ekstraksi 12 fitur dasar
    mean_h, std_h = np.mean(h), np.std(h)
    mean_s, std_s = np.mean(s), np.std(s)
    mean_v, std_v = np.mean(v), np.std(v)
    ratio_s_h = mean_s / (mean_h + 1e-5)
    ratio_v_s = mean_v / (mean_s + 1e-5)
    
    hist_h = cv2.calcHist([hsv], [0], None, [180], [0,180])
    hist_h = hist_h / (hist_h.sum() + 1e-10)
    entropy_h = -np.sum(hist_h * np.log2(hist_h + 1e-10))
    
    total_pixels = h.size
    prop_kuning = np.sum((h >= 20) & (h <= 40)) / total_pixels
    prop_hijau  = np.sum((h >= 50) & (h <= 70)) / total_pixels
    std_gray = np.std(gray)

    return np.array([
        mean_h, mean_s, mean_v, std_h, std_s, std_v,
        ratio_s_h, ratio_v_s, entropy_h, prop_kuning, prop_hijau, std_gray
    ]).reshape(1, -1)

apple_art = """
          ###
         ####
         ##
     ##########
    ############
   ##############
   ##############
   ##############
    ############
     ##########
"""
print(apple_art)

print("Klasifikasi Buah Matang, Mentah, atau Busuk")
print("EB3204 Pembelajaran Mesin dalam Teknik Biomedis")
print("Tugas 5")
print("Dibuat oleh Michael Liebing / 18323016")

# Menjalankan Webcam Real Time
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret: break

    # Mirroring Kamera
    frame = cv2.flip(frame, 1)

    h_f, w_f, _ = frame.shape

    # Teks Instruksi di Atas Tengah
    instr_text = "Letakkan Buah di dalam Kotak Deteksi!"
    font = cv2.FONT_HERSHEY_SIMPLEX
    t_size = cv2.getTextSize(instr_text, font, 0.6, 2)[0]
    t_x = (w_f - t_size[0]) // 2
    cv2.rectangle(frame, (t_x - 10, 20), (t_x + t_size[0] + 10, 60), (0, 0, 0), -1)
    cv2.putText(frame, instr_text, (t_x, 45), font, 0.6, (255, 255, 255), 2)

    # Kotak Deteksi
    size = int(min(h_f, w_f) * 0.4)
    x1, y1 = (w_f - size) // 2, (h_f - size) // 2
    x2, y2 = x1 + size, y1 + size
    roi = frame[y1:y2, x1:x2]

    # Proses Penentuan Kematangan Buah
    feat_raw = extract_features(roi)
    feat_std = sc.transform(feat_raw)     # Standarisasi 12 fitur dasar
    feat_input = feat_std[:, indices_h]   # Mengambil 9 fitur terpenting berdasarkan Hybrid SBS-RF

    pred = model.predict(feat_input)[0]
    prob = model.predict_proba(feat_input)[0]
    conf = np.max(prob) * 100

    # Inverse Mapping
    label_name = class_names.get(pred, "UNKNOWN")

    # Logika Warna dari Kotak Deteksi
    if label_name == 'MATANG':
        color = (0, 255, 0)      # Hijau
    elif label_name == 'MENTAH':
        color = (0, 165, 255)    # Orange
    elif label_name == 'BUSUK':
        color = (0, 0, 255)      # Merah
    else:
        color = (255, 255, 255)  # Putih

    # Visualisasi

    # Gambar Kotak Deteksi
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
    
    # Header Label
    cv2.rectangle(frame, (x1, y1-40), (x2, y1), color, -1) 
    cv2.putText(frame, f"{label_name} ({conf:.1f}%)", (x1 + 5, y1 - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.imshow("Klasifikasi Buah Matang, Mentah, dan Busuk", frame)

    if cv2.waitKey(1) & 0xFF == 27: # Tekan ESC untuk keluar
        break

cap.release()
cv2.destroyAllWindows()
