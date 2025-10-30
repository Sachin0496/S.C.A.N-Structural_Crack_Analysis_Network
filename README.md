#  SCAN: Structural Crack Analysis Network

### An AI-powered system for the detection, classification, and future-risk assessment of structural cracks.

---

## 🎯 Project Overview

**S.C.A.N ** is an advanced structural integrity monitoring system. It leverages the power of **Convolutional Neural Networks (CNNs)** to analyze images and video feeds of infrastructure.

This tool is designed to move beyond simple detection. S.C.A.N. **detects**, **classifies**, and most importantly, **assesses the potential future implications** of structural faults. It provides actionable intelligence to engineers and maintenance teams, enabling them to prioritize repairs and prevent catastrophic failures.

Our primary focus is on critical infrastructure, including:
* **Buildings:** Foundations, walls, and support columns.
* **Roads:** Pavement, highways, and bridge decks.

---

## ✨ Key Features

* **High-Accuracy Detection:** Utilizes a robust CNN model trained to identify cracks of various sizes and types, even in complex or low-light environments.
* **Intelligent Classification:** Automatically categorizes cracks (e.g., transverse, longitudinal, alligator, shear) and assigns a severity level (low, medium, high).
* **Predictive Risk Analysis:** This is the core of S.C.A.N. The system analyzes crack patterns, width, and location to provide a clear report on **potential future implications**, such as:
    * Risk of water ingress and rebar corrosion.
    * Potential for propagation and structural weakening.
    * Prioritized maintenance recommendations.
* **User-Friendly Web Interface:** A clean and simple dashboard for uploading images, viewing analysis results, and exporting reports.

---

## 💻 Technology Stack

This project is built using a modern technology stack to ensure performance and scalability:

* **Core Model:** Convolutional Neural Network (CNN)
* **Deep Learning Framework:** PyTorch 
* **Backend:** Python 
* **Frontend:**  Streamlit
* **Image Processing:** OpenCV, NumPy
* **LLM Assistant:** OpenAI GPT-3.5

---

## ⚙️ How It Works: The Analysis Pipeline

1.  **Input:** The user uploads an image of a structure (e.g., a building facade, a section of road).
2.  **Pre-processing:** The system segments the image and applies filters using OpenCV to enhance crack features.
3.  **CNN Analysis:** The processed image is fed into the trained S.C.A.N. model(feature extraction by VGG-16). The model identifies and localizes all visible cracks.
4.  **Classification & Assessment:** A secondary logic module analyzes the CNN's output. It measures crack dimensions (length, width) and classifies its type.
5.  **Report Generation:** The system generates a comprehensive report that visualizes the cracks on the original image and provides a plain-English summary of the **findings** and **future implications**.

---

## 🚀 Getting Started (Installation)

To get a local copy up and running, follow these simple steps.

### Prerequisites

* Python 3.8+
* pip

### Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/Sachin0496/S.C.A.N-Structural_Crack_Analysis_Network
    cd S.C.A.N-Structural_Crack_Analysis_Network
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the application:**
    ```bash
    python frontend.py
    ```

4.  **Access the web interface:**
    Open your browser and navigate to `http://localhost:8501/#scan-bot`
