# 📸 Task 3: Image Captioning with Attention Mechanism

### Project Overview

This project successfully implements an advanced **Image Captioning** system, integrating **Computer Vision** (CNN) and **Natural Language Processing (NLP)**. The core goal is to generate grammatically correct and semantically accurate descriptions (captions) for given images.

The architecture is based on the **Encoder-Decoder model**, enhanced significantly by the **Bahdanau Attention Mechanism**, which allows the model to achieve high-quality results by intelligently focusing on image regions relevant to the word being generated. 

### ✨ Key Features

* **Integrated Pipeline:** Combines a CNN and an RNN to create an end-to-end vision-to-language model.
* **Encoder (Feature Extraction):** Uses a pre-trained **ResNet101V2** model (Transfer Learning) to extract a rich, spatial feature map of the image.
* **Attention Mechanism:** Implements a custom **Bahdanau Attention Layer** to dynamically weigh image features based on the decoding step, leading to contextually relevant captions. 
* **Decoder (Sequence Generation):** A **Long Short-Term Memory (LSTM)** network generates the sequence of words.
* **Robust Inference:** The final script (`caption_generator.py`) includes robust path handling using the **Raw String** format for reliable testing on Windows systems.

### 🧠 System Architecture

| Component | Technology | Role |
| :--- | :--- | :--- |
| **Encoder** | ResNet101V2 (Pre-trained) | Extracts visual context (feature map) from the image. |
| **Attention** | Bahdanau Attention (Custom Layer) | Focuses the decoder on the most relevant image regions for predicting the next word. |
| **Decoder** | LSTM (RNN) | Generates the next word in the caption sequence based on the attended context. |

### 🛠️ Prerequisites

* Python 3.8+
* Required libraries: `tensorflow`, `keras`, `numpy`, `matplotlib`, etc.
* Trained model assets must be available: `model-weights/`, `tokenizer.pkl`, and `model_architecture.py`.

### 🚀 How to Run (Inference)

The project is designed for immediate testing using the `caption_generator.py` script.

#### 1. Configure the Test Image Path

Open the **`caption_generator.py`** file and update the `TEST_IMAGE_PATH` variable to the **exact absolute path** of the image you want to test.

```python
# In caption_generator.py (Configuration Section)
# Replace this path with your absolute path, ensuring correct filename and extension.
TEST_IMAGE_PATH = r"E:\Image_Captioning_Model\images\35506150_cbdb630f4f.jpg"