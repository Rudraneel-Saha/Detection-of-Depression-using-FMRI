import os
from flask import Flask, render_template, request
from prediction import predict_nifti

UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'.nii', '.nii.gz'}
app = Flask(__name__)
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return any(filename.endswith(ext) for ext in ALLOWED_EXTENSIONS)

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    error = None
    if request.method == 'POST':
        file = request.files.get('file', None)
        if file and allowed_file(file.filename):
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(file_path)
            try:
                result = predict_nifti(file_path)
                prediction = 'Depressed' if result == 1 else 'Not Depressed'
            except Exception as e:
                error = f'Error in prediction: {str(e)}'
        else:
            error = 'Please upload a .nii or .nii.gz MRI file.'
    return render_template('index.html', prediction=prediction, error=error)

if __name__ == "__main__":
    app.run(debug=True)


