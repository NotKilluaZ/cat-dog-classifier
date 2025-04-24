# 🐾 Cat vs Dog Classifier (PyTorch + OpenCV)

A simple yet powerful image classifier that distinguishes between cats and dogs using Convolutional Neural Networks (CNNs) built with PyTorch.

✅ Includes real-time webcam prediction  
✅ Uses data augmentation for better generalization  
✅ Live confidence bar overlay with OpenCV  
✅ Designed as a first-time AI/ML project

---

## 📂 Project Structure

```
Cat & Dog Classifier/
├── data/               # Training & validation images (excluded from Git)
├── model/              # Saved model (.pth)
├── main.py             # Training script
├── predict.py          # Predict from single image
├── webcam_predict.py   # Real-time webcam predictions
├── split_dataset.py    # Splits flat dataset into train/val
├── organize_images.py  # Organizes raw cat/dog images into folders
├── .gitignore          # Prevents pushing venv/data
└── README.md
```

---

## 🚀 How to Run It

### 1️⃣ Install Dependencies

```bash
pip install torch torchvision opencv-python matplotlib
```

---

### 2️⃣ Prepare Your Dataset

- Download [Dogs vs Cats dataset from Kaggle](https://www.kaggle.com/c/dogs-vs-cats)
- Extract `train.zip` into `data/train/`
- Run:
  ```bash
  python organize_images.py     # Sort cat.0.jpg → data/train/cats/
  python split_dataset.py       # Create data/val/ structure
  ```

---

### 3️⃣ Train the Model

```bash
python main.py
```

This trains the CNN for 5 epochs and saves it to `model/cat_dog_model.pth`.

---

### 4️⃣ Predict on a Single Image

```bash
python predict.py
```

Modify the image path in the script to test any image:
```python
img_path = "data/val/cats/cat.1000.jpg"
```

---

### 5️⃣ Real-Time Webcam Prediction

```bash
python webcam_predict.py
```

Press **`q`** to quit the live window.

✔️ Displays live prediction  
✔️ Includes a visual confidence bar (like a power meter)  
✔️ Works even on CPU!

---

## 🔍 Features

- Built from scratch with PyTorch
- Binary classifier for cats vs dogs
- Data augmentation (flip, crop, rotate)
- Sigmoid-based confidence scoring
- Webcam UI using OpenCV with overlays

---

## 🧠 What You'll Learn

- Custom CNN architecture
- Data loading with `ImageFolder`
- Data augmentation strategies
- Model evaluation and inference
- Real-time prediction using camera

---

## 🛠️ Future Ideas

- [ ] Transfer Learning with ResNet
- [ ] Deploy with Streamlit or Gradio
- [ ] Add breed classification (multi-class)
- [ ] Host on Hugging Face Spaces
- [ ] Save webcam snapshots and predictions

---

## 📦 Requirements

- Python 3.8+
- PyTorch
- torchvision
- OpenCV
- matplotlib

---

## 🙌 Acknowledgments

- 🐶 Dataset: [Dogs vs Cats - Kaggle](https://www.kaggle.com/c/dogs-vs-cats)
- 🔥 Framework: PyTorch
- 🎥 Webcam: OpenCV

---

## 👤 Author

**Charlie Parker**  
[GitHub](https://github.com/notkilluaz)  
[Portfolio](https://notkilluaz.github.io/)
