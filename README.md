# Diabetic Retinopathy and Glaucoma Detection Using Machine Learning

This project presents a machine learning-based system for the early and accurate detection of **Diabetic Retinopathy (DR)** and **Glaucoma** using retinal fundus images. It aims to support ophthalmologists in screening and diagnosing these two major causes of blindness.

## 📌 Project Objective
To automate the detection and classification of DR and Glaucoma through a scalable, efficient, and accessible AI-based solution using retinal fundus images.

## 🗂️ Datasets
- The datasets for this project have been sourced from **[Kaggle](https://www.kaggle.com/)**:
  - **EyePACS** – for Diabetic Retinopathy detection
  - **APTOS 2019 Blindness Detection**
  - **RIM-ONE / REFUGE / DRISHTI-GS** – for Glaucoma detection

## 🛠️ Technologies Used
- **Languages:** Python
- **Libraries/Frameworks:** TensorFlow, Keras, PyTorch, OpenCV, scikit-learn
- **Web Stack:** Flask, HTML, Bootstrap
- **IDE:** VS Code, Jupyter Notebook
- **Deployment Options:** Heroku, Docker (optional)

## 🧠 Key Features
- Image preprocessing (resizing, noise reduction, contrast enhancement)
- Feature extraction using both CNN and traditional techniques
- DR classification (5 severity levels: No DR to Proliferative DR)
- Glaucoma detection (binary classification)
- Grad-CAM visualizations for explainability
- Web UI for image upload and prediction results

## ⚙️ Model Architecture
- ResNet50-based deep learning architecture
- Fine-tuned with transfer learning
- Accuracy: **90%+ for DR**, **88% for Glaucoma**

## 🌐 How to Run
1. Clone the repository
2. Install dependencies:  
   ```bash
   pip install -r requirements.txt
   ```
3. Start the web server:  
   ```bash
   python app.py
   ```
4. Open your browser at `http://localhost:5000` and upload a retina image.

## 📸 Sample Screenshot
![DR Detection Sample UI](screenshots/sample_ui.png) *(Add this image to your repo)*

## 📝 Future Scope
- Real-time deployment with mobile app integration
- Expansion to more eye diseases
- Clinical validation with real-world data

## 📄 License
This project is licensed under the [MIT License](LICENSE).

## 🙏 Acknowledgements
- Datasets sourced from [Kaggle](https://www.kaggle.com/)
- Vignan’s Institute of Information Technology for project guidance
- Open-source contributors from TensorFlow, PyTorch, and Flask communities
