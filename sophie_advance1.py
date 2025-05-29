import cv2
import numpy as np
from ultralytics import YOLO
from sklearn.cluster import KMeans
from gtts import gTTS
import os
import uuid
from playsound import playsound
from transformers import MarianMTModel, MarianTokenizer
import torch
import speech_recognition as sr

# ---------------- Language Setup ----------------

# Ask user to choose the language
language_input = input("Choose language (en = English, zh = Chinese, es = Spanish, fr = French): ").strip().lower()
language_map = {"en": "en", "zh": "zh", "es": "es", "fr": "fr"}
target_language = language_map.get(language_input, "en")  # Default to English if invalid input
source_language = "en"

# Check if CUDA (GPU) is available
device = "cuda" if torch.cuda.is_available() else "cpu"

def load_translation_model():
    """
    Load MarianMT translation model and tokenizer for the selected language.
    Falls back to English if the requested model fails to load.
    """
    try:
        model_name = f"Helsinki-NLP/opus-mt-{source_language}-{target_language}"
        tokenizer = MarianTokenizer.from_pretrained(model_name)
        model = MarianMTModel.from_pretrained(model_name).to(device)
        return tokenizer, model
    except Exception:
        print("[WARNING] Translator fallback to English.")
        return None, None

# Initialize the translator
tokenizer, translator = load_translation_model()

def translate(text):
    """
    Translate the input text to the target language.
    Return the original text if English or if translator is not loaded.
    """
    if target_language == "en" or not tokenizer:
        return text
    batch = tokenizer([text], return_tensors="pt", padding=True).to(device)
    translated = translator.generate(**batch)
    return tokenizer.decode(translated[0], skip_special_tokens=True)

def update_language_by_voice():
    """
    Switch the target language using voice command.
    Supports English, Chinese, Spanish, French.
    """
    global target_language, tokenizer, translator
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("[VOICE] Say language: English, Chinese, Spanish, or French.")
        audio = recognizer.listen(source, phrase_time_limit=4)
        try:
            spoken_text = recognizer.recognize_google(audio).lower()
            print(f"[VOICE] You said: {spoken_text}")
            voice_map = {"english": "en", "chinese": "zh", "spanish": "es", "french": "fr"}
            if spoken_text in voice_map:
                target_language = voice_map[spoken_text]
                tokenizer, translator = load_translation_model()
                print(f"[INFO] Language set to {spoken_text.capitalize()}")
        except Exception:
            print("[ERROR] Voice switch failed.")

# ---------------- Color + Audio Utilities ----------------

def closest_color_name(rgb_tuple):
    """
    Return the name of the color closest to the given RGB tuple using Euclidean distance.
    """
    color_reference = {
        (255, 0, 0): "Red", (0, 255, 0): "Green", (0, 0, 255): "Blue",
        (255, 255, 0): "Yellow", (0, 255, 255): "Cyan", (255, 0, 255): "Magenta",
        (0, 0, 0): "Black", (255, 255, 255): "White", (128, 0, 0): "Dark Red",
        (0, 128, 0): "Dark Green", (0, 0, 128): "Dark Blue", (128, 128, 128): "Gray",
        (192, 192, 192): "Silver", (165, 42, 42): "Brown", (255, 165, 0): "Orange"
    }
    min_distance = float("inf")
    closest_name = "Unknown"
    for color_rgb, name in color_reference.items():
        distance = np.linalg.norm(np.array(color_rgb) - np.array(rgb_tuple))
        if distance < min_distance:
            min_distance = distance
            closest_name = name
    return closest_name

def get_dominant_color(region, clusters=3):
    """
    Identify the dominant color in the given region using KMeans clustering.
    """
    region_rgb = cv2.cvtColor(region, cv2.COLOR_BGR2RGB)
    pixels = region_rgb.reshape((-1, 3)).astype(np.float32)
    kmeans = KMeans(n_clusters=clusters, n_init='auto')
    kmeans.fit(pixels)
    dominant_color = kmeans.cluster_centers_[np.argmax(np.bincount(kmeans.labels_))].astype(int)
    return tuple(dominant_color)

def speak_and_save(text, lang_code):
    """
    Generate speech audio and play it.
    Temporary audio files are created and deleted after playback.
    """
    if lang_code == 'zh':
        lang_code = 'zh-CN'  # Fix for Chinese gTTS
    tts = gTTS(text=text, lang=lang_code)
    filename = f"sophie_audio_{uuid.uuid4()}.mp3"
    tts.save(filename)
    playsound(filename)
    os.remove(filename)

# ---------------- Main Function ----------------

def run_sophie():
    """
    Main loop for running Sophie: performs object detection, color identification,
    translation, and audio description.
    """
    model = YOLO("yolov8n.pt")
    cap = cv2.VideoCapture(0)
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))

    # Setup video logger
    os.makedirs("logs", exist_ok=True)
    log_file = os.path.join("logs", "sophie_log.avi")
    video_writer = cv2.VideoWriter(log_file, cv2.VideoWriter_fourcc(*'XVID'), 20.0, (frame_width, frame_height))

    spoken_descriptions = set()
    color_blind_mode = False
    edge_mode = False
    blur_mode = False

    print("[INFO] Press 'q' to quit | 'c' for color-blind | 'e' for edge | 'b' for blur | 'v' for voice switch")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_copy = frame.copy()
        display_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if color_blind_mode else frame.copy()
        if color_blind_mode:
            display_frame = cv2.cvtColor(display_frame, cv2.COLOR_GRAY2BGR)

        if blur_mode:
            blurred_frame = cv2.GaussianBlur(display_frame, (35, 35), 0)
            mask = np.zeros_like(display_frame)

        if edge_mode:
            gray_frame = cv2.cvtColor(frame_copy, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray_frame, 100, 200)
            contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(display_frame, contours, -1, (255, 255, 255), 1)

        results = model(frame_copy, verbose=False)
        boxes = results[0].boxes.xyxy.cpu().numpy()
        class_ids = results[0].boxes.cls.cpu().numpy()

        color_names = []
        detected_labels = []

        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = map(int, box)
            class_id = int(class_ids[i])
            label = model.names[class_id]

            # Crop region with padding
            cropped_roi = frame_copy[max(y1 + 10, 0):max(y2 - 10, 0), max(x1 + 10, 0):max(x2 - 10, 0)]
            if cropped_roi.size == 0:
                continue

            # Center crop for color analysis
            center_region = cropped_roi[cropped_roi.shape[0]//4:3*cropped_roi.shape[0]//4,
                                        cropped_roi.shape[1]//4:3*cropped_roi.shape[1]//4]
            dominant_rgb = get_dominant_color(center_region)
            color_name = closest_color_name(dominant_rgb)

            color_names.append(color_name)
            detected_labels.append(label)

            description = f"A {color_name.lower()} {label} is in the frame."
            translated_description = translate(description)

            # Speak and print translation only once per unique phrase
            if translated_description not in spoken_descriptions:
                print(f"[INFO] {description} → {translated_description}")
                speak_and_save(translated_description, target_language)
                spoken_descriptions.add(translated_description)

            # Draw on frame (English only to avoid rendering issues)
            if blur_mode:
                mask[y1:y2, x1:x2] = frame_copy[y1:y2, x1:x2]
            else:
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(display_frame, description, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                if color_blind_mode:
                    cv2.putText(display_frame, color_name, ((x1 + x2)//2, (y1 + y2)//2),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180, 180, 255), 2)

        # Apply blur if needed
        if blur_mode:
            display_frame = blurred_frame.copy()
            for i, box in enumerate(boxes):
                x1, y1, x2, y2 = map(int, box)
                display_frame[y1:y2, x1:x2] = frame_copy[y1:y2, x1:x2]
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(display_frame, description, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                if color_blind_mode:
                    cv2.putText(display_frame, color_names[i], ((x1 + x2)//2, (y1 + y2)//2),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180, 180, 255), 2)

        # Show dashboard info
        dashboard = [
            f"Mode: {'ColorBlind' if color_blind_mode else 'Normal'}",
            f"Edges: {'ON' if edge_mode else 'OFF'}",
            f"Blur: {'ON' if blur_mode else 'OFF'}",
            f"Last: {color_names[-1] if color_names else 'N/A'}",
            f"Total: {len(detected_labels)}",
            f"Objects: {', '.join(detected_labels) if detected_labels else 'None'}"
        ]
        for idx, line in enumerate(dashboard):
            cv2.putText(display_frame, line, (10, 30 + idx * 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

        video_writer.write(display_frame)
        cv2.imshow("Sophie - Object & Color", display_frame)

        # Handle keyboard input
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c'):
            color_blind_mode = not color_blind_mode
        elif key == ord('e'):
            edge_mode = not edge_mode
        elif key == ord('b'):
            blur_mode = not blur_mode
        elif key == ord('v'):
            update_language_by_voice()

    # Cleanup
    cap.release()
    video_writer.release()
    cv2.destroyAllWindows()

# ---------------- Run the program ----------------
run_sophie()
