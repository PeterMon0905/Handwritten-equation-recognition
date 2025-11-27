# ✍️ Handwritten Equation Recognition
## Introduction

This project implements a system for **Handwritten Equation Recognition** using **Deep Learning** techniques. The system is designed to accurately recognize mathematical symbols and the structural layout of a handwritten equation, then convert it into a digital format.

The primary goal is to provide an efficient tool for rapidly digitizing mathematical expressions noted by hand.

## Key Features

* **Symbol Recognition:** Accurate identification of handwritten digits (0-9), basic operators (+, -, *, /).
* **Output Generation:** Returns the computed numerical result of the recognized expression.
* **Graphical User Interface (GUI):** A simple interface (`gui_demo.py`) allowing users to draw equations directly and receive real-time recognition results.

## Technology Stack

The project is primarily built using Python, utilizing popular libraries for machine learning and image processing:

* **Language:** Python 3.x
* **Deep Learning Framework:** Keras
* **Data Processing:** NumPy, Pandas
* **Computer Vision:** OpenCV, PIL
* **GUI Framework:** Tkinter

## Installation

Follow these steps to set up your environment and run the project locally:

### 1. Clone the Repository

```bash
git clone https://github.com/PeterMon0905/Handwritten-equation-recognition.git
cd Handwritten-equation-recognition
```
### 2. Install Dependencies

```bash
pip install -r requirements.txt
```
## Usage
### Run the GUI Demo
``` bash
python gui_demo.py
```
You can draw the equation with your mouse or you can take a picture of the equation, and the system will attempt to recognize it instantly.
