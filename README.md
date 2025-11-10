# Detection-of-Depression-using-FMRI
Depression Detection via fMRI
A web-based application for detecting depression from fMRI scans using a deep learning CNN.

Table of Contents
About

Features

Demo

Installation

Usage

Project Structure

Contributing

License

About
This project leverages a convolutional neural network (CNN) to classify brain MRI (NIfTI) images as 'depressed' or 'control.' The model is trained on preprocessed fMRI data and offered to end users through a simple Flask-based web application. Users can upload an fMRI slice in .nii or .nii.gz format and receive an instant prediction.

Features
Upload fMRI images in .nii or .nii.gz format

Automatic preprocessing and normalization

Deep learning CNN model for binary classification

Responsive web interface built with Flask

Robust error handling for file types and input

Demo
<!-- Replace with your actual screenshot file path -->
Installation
Clone the repository:

bash
git clone https://github.com/yourusername/your-repo.git
cd your-repo
Install dependencies (preferably in a virtual environment):

bash
pip install -r requirements.txt
Place your pre-trained model weights (depression_cnn.pt) in the model/ directory.

Usage
Run the app:

bash
python app.py
or use the included script:

bash
bash run_and_open.sh  # On Linux/Mac/WSL
run_flask.bat         # On Windows
Visit: http://127.0.0.1:5000

Upload a .nii or .nii.gz file and view prediction results.

Project Structure
text
├── app.py             # Flask application
├── model/
│   ├── train.py       # Training script
│   └── depression_cnn.pt  # Trained weights
├── static/
│   ├── styles.css     # CSS for web UI
│   └── uploads/       # Uploaded MRI images
├── templates/
│   └── index.html     # Web interface HTML
├── requirements.txt   # Dependencies
└── README.md          # Project documentation
Contributing
Pull requests are welcome! For changes, please open an issue first to discuss what you would like to change. Make sure to update tests as appropriate.

License
Distributed under the MIT License. See LICENSE for more information.

Last updated: November 10, 2025
