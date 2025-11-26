from flask import Flask, request, jsonify, render_template
import cv2
import joblib
import numpy as np
import pandas as pd
from ultralytics import YOLO
import matplotlib.pyplot as plt
import base64
from io import BytesIO

app = Flask(__name__)

# Load YOLOv8 model and regression/scaler models
yolo_model = YOLO('Models/csrp-trained-model.pt')
red_diameter_regression = joblib.load('Models/(RED) SVR_RBF_Model - Width(Diameter).pkl')
yellow_diameter_regression = joblib.load('Models/(YELLOW) SVR_RBF_Model - Width(Diameter).pkl')
red_scaler = joblib.load('Models/(RED)scaler - Diameter.pkl')
yellow_scaler = joblib.load('Models/(Yellow)scaler - Diameter.pkl')
red_mass_regression = joblib.load('Models/(RED)(H) SVR_RBF_Model - Projected_Length&Area w True_Diameter.pkl')
yellow_mass_regression = joblib.load('Models/(YELLOW)(H) SVR_RBF_Model - Projected_Length&Area w True_Diameter.pkl')
red_projected_scaler = joblib.load('Models/(RED)scaler - Mass.pkl')
yellow_projected_scaler = joblib.load('Models/(Yellow)scaler - Mass.pkl')

red_onion_size_thresholds = {'small': 3.0, 'medium': 5.0, 'Large': 7.0}
yellow_onion_size_thresholds = {'small': 4.5, 'medium': 6.5, 'Large': 8.5}
red_onion_min_diameter = 1.5
yellow_onion_min_diameter = 1.5

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    uploaded_image = request.files['image']
    image_array = np.fromstring(uploaded_image.read(), np.uint8)
    uploaded_image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    # Resize the image to 640x640
    uploaded_image = cv2.resize(uploaded_image, (640, 640))


    try:
        results, df, annotated_image = process_image(uploaded_image)

        annotated_image_path = 'static/annotated_image.png'
        cv2.imwrite(annotated_image_path, cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))

        return jsonify({"message": "Image processed", "data": df.to_dict(orient='records'), "annotated_image": annotated_image_path})
    
    except ValueError as ve:
        # Display errors related to missing reference object or empty results
        return jsonify({"error": str(ve)}), 400

    except AttributeError as ae:
        return jsonify({"error": "No Objects detected in the image. Please upload an image with Onion and a Reference Object."}), 400

def process_image(uploaded_image):
    # Perform YOLOv8 detection
    results = yolo_model(uploaded_image)

    masks = results[0].masks

    if masks is None:
        raise AttributeError("No Objects detected in the image. Please upload an image with Onion and a Reference Object.")

    onion_data = []
    reference_data = None

    class_names = yolo_model.names

    reference_count = 0
    onion_count = 0
    for i, mask in enumerate(masks.data):
        binary_mask = mask.cpu().numpy()
        y_coords, x_coords = np.where(binary_mask == 1)

        width = np.max(x_coords) - np.min(x_coords)
        length = np.max(y_coords) - np.min(y_coords)
        area = np.sum(binary_mask == 1)

        class_id = int(results[0].boxes.cls[i])
        class_label = class_names[class_id]

        leftmost_x = np.min(x_coords)
        bottommost_y = np.min(y_coords)

        if class_label in ['Red-Onion', 'Yellow-Onion']:
            onion_data.append({'Class': class_label, 'width': width, 'length': length, 'area': area, 'leftmost_x': leftmost_x, 'bottommost_y': bottommost_y})
            onion_count += 1
        elif class_label == 'Reference-Object' and reference_count < 2:
            reference_data = {'width': width, 'length': length, 'area': area}
            reference_count += 1

    onion_data.sort(key=lambda x: x['leftmost_x'])

    if onion_count == 0:
        raise ValueError("No Onion detected. Please upload image with Onion")

    if reference_count >= 2:
        raise ValueError("Multiple Reference Objects detected in the image. Please upload an image with only one Reference Object.")

    if reference_data is None:
        raise ValueError("No Reference Object found in the image. Please include a Reference Object.")


    combined_data = []
    if reference_data:
        for idx, onion in enumerate(onion_data):
            projected_length = (onion['length'] * 3) / reference_data['length']
            projected_area = (onion['area'] * (np.pi * (1.5**2))) / reference_data['area']

            combined_data.append({
                'Object': idx + 1,
                'Class': onion['Class'],
                'Width_x': onion['width'],
                'Width_y': reference_data['width'],
                'length_x': onion['length'],
                'length_y': reference_data['length'],
                'area_x': onion['area'],
                'area_y': reference_data['area'],
                'projected_length': projected_length,
                'projected_area': projected_area,
                'leftmost_x': onion['leftmost_x'],
                'bottommost_y': onion['bottommost_y'],
                'length': onion['length'],
                'width': onion['width']
            })

    df = pd.DataFrame(combined_data, columns=['Object', 'Class', 'Width_x', 'Width_y', 'length_x', 'length_y', 'area_x', 'area_y', 'projected_length', 'projected_area'])

    # Size Classification
    red_onion_size_thresholds = {'small': 3.0, 'medium': 5.0, 'Large': 7.0}
    yellow_onion_size_thresholds = {'small': 4.5, 'medium': 6.5, 'Large': 8.5}

    red_onion_min_diameter = 1.5
    yellow_onion_min_diameter = 1.5

    def classify_size(diameter, thresholds, min_size):
        """Classify onion size based on diameter, thresholds, and minimum size."""
        if diameter < min_size:
            return 'No Class'
        elif diameter <= thresholds['small']:
            return 'S'
        elif diameter <= thresholds['medium']:
            return 'M'
        elif diameter <= thresholds['Large']:
            return 'L'
        else:
            return 'XL'

    # Scale and Predict Diameter
    for index, row in df.iterrows():
        if row['Class'] == 'Red-Onion':
            scaled_features = red_scaler.transform(df[['Width_x', 'Width_y']].iloc[[index]])

            df.at[index, 'width_x_scaled'] = scaled_features[0, 0]
            df.at[index, 'width_y_scaled'] = scaled_features[0, 1]

            diameter_prediction = red_diameter_regression.predict([[df.at[index, 'width_x_scaled'], df.at[index, 'width_y_scaled']]])[0]
            df.at[index, 'Diameter'] = diameter_prediction

            red_scaled_projected_features = red_projected_scaler.transform(df[['Diameter', 'projected_length', 'projected_area']].iloc[[index]])

            df.at[index, 'Diameter_scaled'] = red_scaled_projected_features[0, 0]
            df.at[index, 'projected_length_scaled'] = red_scaled_projected_features[0, 1]
            df.at[index, 'projected_area_scaled'] = red_scaled_projected_features[0, 2]

            df.at[index, 'Size'] = classify_size(df.at[index, 'Diameter'], red_onion_size_thresholds, red_onion_min_diameter)

        elif row['Class'] == 'Yellow-Onion':
            yellow_scaled_features = yellow_scaler.transform(df[['Width_x', 'Width_y']].iloc[[index]])

            df.at[index, 'width_x_scaled'] = yellow_scaled_features[0, 0]
            df.at[index, 'width_y_scaled'] = yellow_scaled_features[0, 1]

            diameter_prediction = yellow_diameter_regression.predict([[df.at[index, 'width_x_scaled'], df.at[index, 'width_y_scaled']]])[0]
            df.at[index, 'Diameter'] = diameter_prediction

            yellow_scaled_projected_features = yellow_projected_scaler.transform(df[['Diameter', 'projected_length', 'projected_area']].iloc[[index]])

            df.at[index, 'Diameter_scaled'] = yellow_scaled_projected_features[0, 0]
            df.at[index, 'projected_length_scaled'] = yellow_scaled_projected_features[0, 1]
            df.at[index, 'projected_area_scaled'] = yellow_scaled_projected_features[0, 2]

            df.at[index, 'Size'] = classify_size(df.at[index, 'Diameter'], yellow_onion_size_thresholds, yellow_onion_min_diameter)

    # Predict Mass
    for index, row in df.iterrows():
        if row['Class'] == 'Red-Onion':
            mass_input = [[row['Diameter_scaled'], row['projected_length_scaled'], row['projected_area_scaled']]]
            predicted_mass = red_mass_regression.predict(mass_input)[0]
            df.at[index, 'Mass'] = predicted_mass

        elif row['Class'] == 'Yellow-Onion':
            mass_input = [[row['Diameter_scaled'], row['projected_length_scaled'], row['projected_area_scaled']]]
            predicted_mass = yellow_mass_regression.predict(mass_input)[0]
            df.at[index, 'Mass'] = predicted_mass

    result_columns = df[['Object', 'Class', 'Diameter', 'Mass', 'Size']]
    print(result_columns)

    # Diplay image with annotation
    def draw_transparent_box(image, box_coords, color, alpha=0.6):
        overlay = image.copy()
        cv2.rectangle(overlay, box_coords[0], box_coords[1], color, cv2.FILLED)
        return cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)

    # Wrap text
    def wrap_text(text, font, font_scale, thickness, max_width):
        lines = text.split('\n')
        wrapped_lines = []

        for line in lines:
            words = line.split(' ')
            current_line = words[0]

            for word in words[1:]:
                (line_width, _), _ = cv2.getTextSize(current_line + ' ' + word, font, font_scale, thickness)
                if line_width <= max_width:
                    current_line += ' ' + word
                else:
                    wrapped_lines.append(current_line)
                    current_line = word
            wrapped_lines.append(current_line)

        return wrapped_lines

    image_rgb = cv2.cvtColor(uploaded_image, cv2.COLOR_BGR2RGB)
    color = (37, 82, 186)
    font_scale = 0.4
    thickness = 1
    textbox_color = (37, 82, 186)
    padding_top = 10

    # Bounding Box
    for idx, item in enumerate(combined_data):
        if item['Class'] in ['Red-Onion', 'Yellow-Onion']:
            x_min = item['leftmost_x']
            y_min = item['bottommost_y']
            y_max = item['bottommost_y'] + item['length']
            x_max = x_min + item['width']

            cv2.rectangle(image_rgb, (x_min, y_min), (x_max, y_max), color, 2)

            df_row = df.iloc[idx]
            diameter = df_row['Diameter']
            mass = df_row['Mass']
            size = df_row['Size']

            label = f"ID: {item['Object']} {item['Class']} \nDiameter: {diameter:.2f}cm \nSize: {size}\nMass: {mass:.2f}g"

            max_text_width = x_max - x_min
            wrapped_lines = wrap_text(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness, max_text_width)

            line_height = cv2.getTextSize(wrapped_lines[0], cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0][1]
            total_text_height = (line_height + 2) * len(wrapped_lines) + padding_top

            box_coords = (
                (x_min, y_min - total_text_height - 10),
                (x_max, y_min)
            )

            image_rgb = draw_transparent_box(image_rgb, box_coords, textbox_color, alpha=0.6)

            for i, line in enumerate(wrapped_lines):
                text_position = (x_min, y_min - total_text_height + (line_height + 5) * i + padding_top // 2)
                cv2.putText(image_rgb, line, text_position, cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)


    annotated_image = image_rgb

    return results, df, annotated_image

if __name__ == '__main__':
    app.run(host='172.21.144.1', port=5000, debug=True, threaded=False)
