from flask import Flask, render_template, Response, jsonify, request
import cv2
import numpy as np
import mediapipe as mp
from collections import deque
import pickle
import os
import threading
import base64
import requests as http_requests
from scipy.stats import skew, kurtosis
from scipy.spatial.distance import euclidean
from sklearn.decomposition import PCA # Added PCA import

app = Flask(__name__)

# ==================== CONFIGURATION ====================
ALPHABET_MODEL_PATH = 'random_forest.pkl'
DIGIT_MODEL_PATH = 'random_forest_model.pkl'

# Added Paths for PCA and HMM
PCA_ALPHABET_PATH = 'pca_alphabet.pkl'
PCA_DIGIT_PATH = 'pca_digit.pkl'
HMM_ALPHABET_PATH = 'hmm_alphabet.pkl'
HMM_DIGIT_PATH = 'hmm_digit.pkl'

# ==================== GOOGLE DRIVE MODEL DOWNLOAD ====================
# Map each local filename to its Google Drive FILE ID.
# Get the file ID from the shareable link:
#   https://drive.google.com/file/d/<FILE_ID>/view
# Leave the value as empty string "" if you don't have that model.
GDRIVE_MODEL_IDS = {
    ALPHABET_MODEL_PATH : os.environ.get('GDRIVE_ALPHABET_RF',  'https://drive.google.com/file/d/1jfUt7nxckePnygjYjsKv7Mr-Vcs7S9bS/view?usp=drive_link'),
    DIGIT_MODEL_PATH    : os.environ.get('GDRIVE_DIGIT_RF',     'https://drive.google.com/file/d/1tCn3dqKivMLek2mSpepg6kVrpkowy8b6/view?usp=drive_link'),
    PCA_ALPHABET_PATH   : os.environ.get('GDRIVE_PCA_ALPHABET', 'https://drive.google.com/file/d/1ncAkwgX3tRwKsf51PLw9PlLK8lMAPRl4/view?usp=drive_link'),
    PCA_DIGIT_PATH      : os.environ.get('GDRIVE_PCA_DIGIT',    'https://drive.google.com/file/d/1oSvYmi8OlLAUkB_eV_JlJrof7I4Atmhu/view?usp=drive_link'),
    HMM_ALPHABET_PATH   : os.environ.get('GDRIVE_HMM_ALPHABET', 'https://drive.google.com/file/d/1uXNJV5FaDD3SCx1lqHixHguaAExnYxwS/view?usp=drive_link'),
    HMM_DIGIT_PATH      : os.environ.get('GDRIVE_HMM_DIGIT',    'https://drive.google.com/file/d/1kMJ3VAhpt1UNimcyFKl6BqskVsWyrVPl/view?usp=drive_link'),
}

def download_from_gdrive(file_id: str, dest_path: str) -> bool:
    """Download a file from Google Drive by file ID. Returns True on success."""
    if not file_id:
        return False
    if os.path.exists(dest_path):
        print(f"  ✓ {dest_path} already exists, skipping download.")
        return True

    print(f"  ⬇ Downloading {dest_path} from Google Drive ...")
    # Google Drive direct-download URL (works for files shared as 'Anyone with link')
    url = f"https://drive.google.com/uc?export=download&id={file_id}"
    session = http_requests.Session()

    try:
        response = session.get(url, stream=True, timeout=300)

        # For large files Drive returns a warning page — confirm it
        token = None
        for key, value in response.cookies.items():
            if key.startswith('download_warning'):
                token = value
                break

        if token:
            response = session.get(url, params={'confirm': token},
                                   stream=True, timeout=300)

        if response.status_code != 200:
            print(f"  ✗ Failed ({response.status_code}) for {dest_path}")
            return False

        with open(dest_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=32768):
                if chunk:
                    f.write(chunk)

        size_mb = os.path.getsize(dest_path) / (1024 * 1024)
        print(f"  ✓ {dest_path} downloaded ({size_mb:.1f} MB)")
        return True

    except Exception as e:
        print(f"  ✗ Download error for {dest_path}: {e}")
        if os.path.exists(dest_path):
            os.remove(dest_path)   # remove partial file
        return False

def download_all_models():
    """Download all models from Google Drive if not already present."""
    any_id_set = any(GDRIVE_MODEL_IDS.values())
    if not any_id_set:
        print("No Google Drive IDs configured — skipping download. "
              "Set GDRIVE_* environment variables to enable auto-download.")
        return

    print("\n📥 Checking / downloading models from Google Drive ...")
    for local_path, file_id in GDRIVE_MODEL_IDS.items():
        if file_id:
            download_from_gdrive(file_id, local_path)
        else:
            print(f"  – {local_path}: no Drive ID set, skipping.")
    print("📥 Model download step complete.\n")

# ==================== GLOBAL VARIABLES ====================
# NOTE: Camera is now handled by the browser (WebRTC).
# The server receives frames via POST /process_frame instead of cv2.VideoCapture.
camera_lock = threading.Lock()
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Lip landmark indices (MediaPipe Face Mesh)
OUTER_LIP = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95]
INNER_LIP = [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95]
lip_indices = list(set(OUTER_LIP + INNER_LIP))

# Lip open/close detection indices (MediaPipe Face Mesh)
LIP_TOP_INNER = 13
LIP_BOTTOM_INNER = 14
LIP_LEFT_CORNER = 78
LIP_RIGHT_CORNER = 308
LIP_OPEN_THRESHOLD = 0.02

# Buffer for temporal analysis
landmark_buffer = deque(maxlen=15)

# Current state
current_mode = "alphabet"  # "alphabet" or "digit"
current_prediction = "-"
current_confidence = 0.0
current_hmm_score = 0.0
is_detecting = False
is_lip_open = False
frame_count = 0

# ==================== LOAD MODELS ====================
alphabet_model = None
digit_model = None

pca_alphabet = None
pca_digit = None
hmm_alphabet = None
hmm_digit = None

models_status = {
    "alphabet_rf": False, "digit_rf": False,
    "alphabet_pca": False, "digit_pca": False,
    "alphabet_hmm": False, "digit_hmm": False
}

def load_pickle(path):
    try:
        if os.path.exists(path):
            with open(path, 'rb') as f:
                return pickle.load(f)
    except Exception as e:
        print(f"Error loading {path}: {e}")
    return None

# Download models from Google Drive if needed (runs at startup)
download_all_models()

print("Loading Random Forest models...")
alphabet_model = load_pickle(ALPHABET_MODEL_PATH)
models_status["alphabet_rf"] = alphabet_model is not None

digit_model = load_pickle(DIGIT_MODEL_PATH)
models_status["digit_rf"] = digit_model is not None

print("Loading PCA models...")
pca_alphabet = load_pickle(PCA_ALPHABET_PATH)
models_status["alphabet_pca"] = pca_alphabet is not None
if pca_alphabet: print("✓ PCA Alphabet loaded")

pca_digit = load_pickle(PCA_DIGIT_PATH)
models_status["digit_pca"] = pca_digit is not None
if pca_digit: print("✓ PCA Digit loaded")

print("Loading HMM models...")
hmm_alphabet = load_pickle(HMM_ALPHABET_PATH)
models_status["alphabet_hmm"] = hmm_alphabet is not None
if hmm_alphabet: print("✓ HMM Alphabet loaded")

hmm_digit = load_pickle(HMM_DIGIT_PATH)
models_status["digit_hmm"] = hmm_digit is not None
if hmm_digit: print("✓ HMM Digit loaded")

# ==================== LIP OPEN/CLOSE DETECTION ====================

def check_lip_open(landmarks, img_w, img_h):
    top    = landmarks.landmark[LIP_TOP_INNER]
    bottom = landmarks.landmark[LIP_BOTTOM_INNER]
    left   = landmarks.landmark[LIP_LEFT_CORNER]
    right  = landmarks.landmark[LIP_RIGHT_CORNER]

    top_pt    = np.array([top.x    * img_w, top.y    * img_h])
    bottom_pt = np.array([bottom.x * img_w, bottom.y * img_h])
    left_pt   = np.array([left.x   * img_w, left.y   * img_h])
    right_pt  = np.array([right.x  * img_w, right.y  * img_h])

    opening   = euclidean(top_pt, bottom_pt)
    width     = euclidean(left_pt, right_pt)

    if width < 1e-6:
        return False, 0.0

    mar = opening / width
    return mar >= LIP_OPEN_THRESHOLD, mar

# ==================== FEATURE EXTRACTION ====================

def extract_simple_lip_features(landmarks):
    features = []
    points = landmarks.reshape(-1, 2)

    features.extend(landmarks.flatten())

    centroid = np.mean(points, axis=0)
    features.extend(centroid)

    distances = [euclidean(p, centroid) for p in points]
    features.extend([np.mean(distances), np.std(distances),
                     np.max(distances), np.min(distances)])

    x_coords, y_coords = points[:, 0], points[:, 1]
    width = np.max(x_coords) - np.min(x_coords)
    height = np.max(y_coords) - np.min(y_coords)
    aspect_ratio = width / (height + 1e-6)
    features.extend([width, height, aspect_ratio])

    features.extend([np.mean(x_coords), np.std(x_coords), skew(x_coords), kurtosis(x_coords)])
    features.extend([np.mean(y_coords), np.std(y_coords), skew(y_coords), kurtosis(y_coords)])

    key_indices = [0, len(points)//4, len(points)//2, 3*len(points)//4, -1]
    for i in range(len(key_indices)-1):
        dist = euclidean(points[key_indices[i]], points[key_indices[i+1]])
        features.append(dist)

    if len(points) > 10:
        mouth_opening = euclidean(points[3], points[9]) if len(points) > 9 else 0
        features.append(mouth_opening)

    return np.array(features)


def extract_temporal_features(landmark_sequence):
    if len(landmark_sequence) == 0:
        return None
    
    frame_features = [extract_simple_lip_features(lm) for lm in landmark_sequence]
    frame_features_array = np.array(frame_features)
    
    features = []
    
    features.extend(np.mean(frame_features_array, axis=0))
    features.extend(np.std(frame_features_array, axis=0))
    features.extend(np.max(frame_features_array, axis=0))
    features.extend(np.min(frame_features_array, axis=0))
    
    if len(landmark_sequence) > 1:
        velocities = np.diff(frame_features_array, axis=0)
        features.extend(np.mean(velocities, axis=0))
        features.extend(np.std(velocities, axis=0))
    else:
        features.extend(np.zeros(len(frame_features_array[0])))
        features.extend(np.zeros(len(frame_features_array[0])))
    
    return np.array(features)

def calculate_hmm_score(sequence_features, mode):
    hmm_models = hmm_alphabet if mode == "alphabet" else hmm_digit
    
    if not hmm_models:
        return 0.0, None

    best_score = float('-inf')
    best_label = None

    pca_model = pca_alphabet if mode == "alphabet" else pca_digit
    
    processed_seq = sequence_features
    if pca_model:
        try:
            processed_seq = pca_model.transform(sequence_features)
        except:
            pass

    try:
        if isinstance(hmm_models, dict):
            for label, model in hmm_models.items():
                try:
                    score = model.score(processed_seq)
                    if score > best_score:
                        best_score = score
                        best_label = label
                except:
                    continue
    except Exception as e:
        print(f"HMM scoring error: {e}")
        return 0.0, None
        
    return best_score, best_label

def predict_from_landmarks(landmark_sequence, mode):
    global current_prediction, current_confidence, current_hmm_score
    
    if len(landmark_sequence) < 5:
        return None, 0.0
    
    rf_model = alphabet_model if mode == "alphabet" else digit_model
    pca_model = pca_alphabet if mode == "alphabet" else pca_digit
    
    if rf_model is None:
        return None, 0.0
    
    try:
        raw_features = extract_temporal_features(landmark_sequence)
        
        if raw_features is None:
            return None, 0.0
        
        features_for_pred = raw_features.reshape(1, -1)
        
        if pca_model is not None:
            try:
                features_for_pred = pca_model.transform(features_for_pred)
                print(f"PCA transform: {raw_features.shape} -> {features_for_pred.shape}")
            except Exception as e:
                print(f"PCA Transformation failed: {e}")
        
        frame_features_seq = np.array([extract_simple_lip_features(lm) for lm in landmark_sequence])
        hmm_score, hmm_label = calculate_hmm_score(frame_features_seq, mode)
        current_hmm_score = hmm_score
        
        if hasattr(rf_model, 'n_features_in_'):
            expected = rf_model.n_features_in_
            current = features_for_pred.shape[1]
            if current < expected:
                features_for_pred = np.hstack([features_for_pred, np.zeros((1, expected - current))])
            elif current > expected:
                features_for_pred = features_for_pred[:, :expected]
        
        prediction = rf_model.predict(features_for_pred)[0]
        
        if hasattr(rf_model, 'predict_proba'):
            probabilities = rf_model.predict_proba(features_for_pred)[0]
            class_idx = np.where(rf_model.classes_ == prediction)[0]
            confidence = probabilities[class_idx[0]] if len(class_idx) > 0 else 0.0
        else:
            confidence = 1.0
        
        if mode == "alphabet":
            if 0 <= prediction <= 25:
                predicted_char = chr(int(prediction) + ord('A'))
            else:
                predicted_char = chr(int(prediction % 26) + ord('A'))
        else:
            predicted_char = str(int(prediction) % 10)
        
        current_prediction = predicted_char
        current_confidence = float(confidence)
        
        print(f"Pred: {predicted_char} | Conf: {confidence:.2f} | HMM Score: {hmm_score:.2f}")
        
        return predicted_char, float(confidence)
        
    except Exception as e:
        print(f"Prediction error: {e}")
        import traceback
        traceback.print_exc()
        return None, 0.0

# ==================== FRAME PROCESSING (replaces server-side camera) ====================

def process_frame_data(frame):
    """
    Process a single frame received from the browser.
    Runs MediaPipe, draws landmarks, runs prediction.
    Returns annotated JPEG bytes + metadata.
    """
    global landmark_buffer, current_prediction, current_confidence
    global current_mode, is_detecting, current_hmm_score, is_lip_open, frame_count

    h, w, _ = frame.shape

    # Flip for mirror effect
    frame = cv2.flip(frame, 1)

    # Convert to RGB for MediaPipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    # Draw mode indicator
    mode_text = f"Mode: {current_mode.upper()}"
    mode_color = (0, 255, 0) if is_detecting else (128, 128, 128)
    cv2.putText(frame, mode_text, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, mode_color, 2)

    lip_open = False
    mar = 0.0

    if results.multi_face_landmarks:
        landmarks = results.multi_face_landmarks[0]

        lip_open, mar = check_lip_open(landmarks, w, h)
        is_lip_open = lip_open

        lip_state_text = f"Lip: {'OPEN' if lip_open else 'CLOSED'}  MAR:{mar:.3f}"
        lip_state_color = (0, 255, 0) if lip_open else (0, 0, 255)
        cv2.putText(frame, lip_state_text, (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, lip_state_color, 2)

        if not lip_open:
            landmark_buffer.clear()
            current_prediction = "-"
            current_confidence = 0.0
            current_hmm_score = 0.0

        lip_points = []
        for i in lip_indices:
            x = int(landmarks.landmark[i].x * w)
            y = int(landmarks.landmark[i].y * h)
            lip_points.append([x, y])
            cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)

        lip_points_array = np.array(lip_points, dtype=np.int32)
        cv2.polylines(frame, [lip_points_array], True, (0, 255, 0), 1)

        if len(lip_points_array) > 0:
            x_min, y_min = np.min(lip_points_array, axis=0)
            x_max, y_max = np.max(lip_points_array, axis=0)

            x_range = x_max - x_min if x_max > x_min else 1
            y_range = y_max - y_min if y_max > y_min else 1

            normalized_points = (lip_points_array - np.array([x_min, y_min])) / \
                               np.array([x_range, y_range])

            if lip_open:
                landmark_buffer.append(normalized_points)

                if is_detecting and frame_count % 15 == 0 and len(landmark_buffer) >= 10:
                    predict_from_landmarks(list(landmark_buffer), current_mode)
    else:
        is_lip_open = False
        cv2.putText(frame, "No face detected", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    if is_detecting and current_prediction != "-":
        text = f"{current_prediction}"
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 2.5, 4)[0]

        cv2.rectangle(frame, (w - text_size[0] - 30, 10),
                      (w - 10, text_size[1] + 30), (0, 255, 0), -1)

        cv2.putText(frame, text, (w - text_size[0] - 20, text_size[1] + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 2.5, (0, 0, 0), 4)

        conf_text = f"RF: {current_confidence:.1%}"
        cv2.putText(frame, conf_text, (w - 150, text_size[1] + 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        hmm_text = f"HMM: {current_hmm_score:.1f}"
        cv2.putText(frame, hmm_text, (w - 150, text_size[1] + 85),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

    status_text = "DETECTING..." if is_detecting else "PAUSED"
    status_color = (0, 255, 0) if is_detecting else (128, 128, 128)
    cv2.putText(frame, status_text, (10, h - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)

    buffer_text = f"Buffer: {len(landmark_buffer)}/15"
    cv2.putText(frame, buffer_text, (10, h - 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    frame_count += 1

    ret, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return buf.tobytes()

# ==================== FLASK ROUTES ====================

@app.route('/')
def index():
    """Render main page"""
    return render_template('index.html')

@app.route('/process_frame', methods=['POST'])
def process_frame():
    """
    Receives a base64-encoded JPEG frame from the browser,
    runs MediaPipe + ML pipeline, returns annotated frame + state.
    Replaces the server-side /video_feed SSE stream.
    """
    try:
        data = request.get_json()
        if not data or 'frame' not in data:
            return jsonify({'error': 'No frame data'}), 400

        # Decode base64 image
        img_data = base64.b64decode(data['frame'].split(',')[1] if ',' in data['frame'] else data['frame'])
        nparr = np.frombuffer(img_data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if frame is None:
            return jsonify({'error': 'Invalid frame'}), 400

        # Process frame (all ML logic runs here)
        annotated_bytes = process_frame_data(frame)

        # Return annotated frame as base64 + current state
        annotated_b64 = base64.b64encode(annotated_bytes).decode('utf-8')

        return jsonify({
            'frame': 'data:image/jpeg;base64,' + annotated_b64,
            'prediction': current_prediction,
            'confidence': float(current_confidence),
            'hmm_score': float(current_hmm_score),
            'is_lip_open': is_lip_open,
            'is_detecting': is_detecting,
            'current_mode': current_mode,
        })

    except Exception as e:
        print(f"Frame processing error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/status')
def status():
    """Get system status"""
    return jsonify({
        'camera_active': True,  # Camera is always "active" — it lives in the browser
        'alphabet_model_loaded': models_status["alphabet_rf"],
        'digit_model_loaded': models_status["digit_rf"],
        'pca_loaded': models_status["alphabet_pca"],
        'hmm_loaded': models_status["alphabet_hmm"],
        'current_mode': current_mode,
        'is_detecting': is_detecting,
        'current_prediction': current_prediction,
        'current_confidence': float(current_confidence),
        'current_hmm_score': float(current_hmm_score),
        'is_lip_open': is_lip_open
    })

@app.route('/set_mode', methods=['POST'])
def set_mode():
    """Set detection mode (alphabet or digit)"""
    global current_mode, current_prediction, landmark_buffer, current_hmm_score
    
    data = request.json
    mode = data.get('mode', 'alphabet')
    
    if mode in ['alphabet', 'digit']:
        current_mode = mode
        current_prediction = "-"
        current_hmm_score = 0.0
        landmark_buffer.clear()
        print(f"Mode switched to: {mode}")
        return jsonify({'success': True, 'mode': current_mode})
    
    return jsonify({'success': False, 'error': 'Invalid mode'})

@app.route('/toggle_detection', methods=['POST'])
def toggle_detection():
    """Toggle detection on/off"""
    global is_detecting, current_prediction, landmark_buffer, current_hmm_score
    
    is_detecting = not is_detecting
    
    if not is_detecting:
        current_prediction = "-"
        current_hmm_score = 0.0
        landmark_buffer.clear()
    
    print(f"Detection {'started' if is_detecting else 'stopped'}")
    
    return jsonify({'success': True, 'is_detecting': is_detecting})

# ==================== MAIN ====================

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))

    print("\n" + "=" * 70)
    print("🎥 DUAL MODE LIP READING APPLICATION (PCA + HMM)".center(70))
    print("=" * 70)
    
    print("\n📋 MODEL STATUS:")
    print(f"   [{'✓' if models_status['alphabet_rf'] else '✗'}] Alphabet RF Model")
    print(f"   [{'✓' if models_status['digit_rf'] else '✗'}] Digit RF Model")
    print(f"   [{'✓' if models_status['alphabet_pca'] else '✗'}] Alphabet PCA Model")
    print(f"   [{'✓' if models_status['digit_pca'] else '✗'}] Digit PCA Model")
    print(f"   [{'✓' if models_status['alphabet_hmm'] else '✗'}] Alphabet HMM Model")
    
    if not any(models_status.values()):
        print("\n⚠️  WARNING: No models loaded!")
    
    print(f"\n📡 SERVER INFO:")
    print(f"   URL: http://localhost:{port}")
    
    print("\n⏹️  TO STOP: Press Ctrl+C")
    print("=" * 70 + "\n")
    
    app.run(debug=False, host='0.0.0.0', port=port, threaded=True)
