from flask import Flask, request, render_template, flash, redirect, url_for
from werkzeug.utils import secure_filename
import os
import torch
from PIL import Image
import sys

# Add project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.model_utils import load_model, predict_image_with_confidence

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['SECRET_KEY'] = 'your-secret-key'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Model and classes setup
model = None
CLASS_NAMES = {
    0: "No DR",
    1: "Mild DR",
    2: "Moderate DR", 
    3: "Severe DR",
    4: "Proliferative DR"
}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'png', 'jpg', 'jpeg'}

def load_app_model():
    global model
    try:
        model_path = os.path.join('..', 'models', 'checkpoints', 'best_model.pth')
        model = load_model(model_path, num_classes=len(CLASS_NAMES))
        print("Model loaded successfully")
    except Exception as e:
        print(f"Failed to load model: {str(e)}")
        raise

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file selected')
            return redirect(request.url)
        
        file = request.files['file']
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
            
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            try:
                pred_label, confidence, _ = predict_image_with_confidence(
                    model, filepath, list(CLASS_NAMES.values()))
                
                severity_colors = {
                    "No DR": "success",
                    "Mild DR": "info",
                    "Moderate DR": "warning",
                    "Severe DR": "danger",
                    "Proliferative DR": "danger"
                }
                
                return render_template('index.html', 
                                    prediction=pred_label,
                                    confidence=f"{confidence*100:.1f}%",
                                    severity=severity_colors.get(pred_label, "secondary"),
                                    image_url=url_for('static', filename=f'uploads/{filename}'))
            
            except Exception as e:
                flash(f"Prediction error: {str(e)}")
                return redirect(request.url)
    
    return render_template('index.html')

if __name__ == '__main__':
    load_app_model()
    app.run(debug=True)