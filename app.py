import os
from flask import Flask,request,render_template,send_file
import torch
from werkzeug.utils import secure_filename
from model_revised import train_style_transfer,load_image,save_output_image

app=Flask(__name__)

UPLOAD_FOLDER='uploads'
OUTPUT_FOLDER='outputs'
ALLOWED_EXTENSIONS={'png','jpg','jpeg'}

app.config['UPLOAD_FOLDER']=UPLOAD_FOLDER
app.config['OUTPUT_FOLDER']=OUTPUT_FOLDER

os.makedirs(UPLOAD_FOLDER,exist_ok=True)
os.makedirs(OUTPUT_FOLDER,exist_ok=True)

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.',1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/',methods=['GET','POST'])
def upload_files():
    if request.method=='POST':
        if 'content' not in request.files or 'style' not in request.files:
            return 'Both content and style images are required',400
        
        content_file=request.files['content']
        style_file=request.files['style']
        
        if not content_file or not style_file:
            return 'No selected file',400
        
        if not (allowed_file(content_file.filename) and allowed_file(style_file.filename)):
            return 'Invalid file type. Only JPG, JPEG, and PNG are allowed',400
        
        content_path=os.path.join(app.config['UPLOAD_FOLDER'],secure_filename(content_file.filename))
        style_path=os.path.join(app.config['UPLOAD_FOLDER'],secure_filename(style_file.filename))
        
        content_file.save(content_path)
        style_file.save(style_path)
        
        output_filename=f'styled_{secure_filename(content_file.filename)}'
        output_path=os.path.join(app.config['OUTPUT_FOLDER'],output_filename)
        
        epochs=int(request.form.get('epochs',100))
        content_weight=float(request.form.get('content_weight',1.0))
        style_weight=float(request.form.get('style_weight',1e4))
        
        try:
            train_style_transfer(
                content_image_path=content_path,
                style_image_path=style_path,
                output_image_path=output_path,
                epochs=epochs,
                content_weight=content_weight,
                style_weight=style_weight
            )
            
            return send_file(output_path,mimetype='image/jpeg')
        
        except Exception as e:
            return f'Style transfer failed: {str(e)}',500

    return '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Style Transfer Application</title>
</head>
<body>
    <h1>Upload Content and Style Images</h1>
    <form method="post" enctype="multipart/form-data">
        <label for="content">Content Image:</label>
        <input type="file" id="content" name="content" required><br><br>
        <label for="style">Style Image:</label>
        <input type="file" id="style" name="style" required><br><br>
        <label for="epochs">Epochs:</label>
        <input type="number" id="epochs" name="epochs" min="1" value="100"><br><br>
        <label for="content_weight">Content Weight:</label>
        <input type="number" id="content_weight" name="content_weight" step="0.1" value="1.0"><br><br>
        <label for="style_weight">Style Weight:</label>
        <input type="number" id="style_weight" name="style_weight" step="0.1" value="10000"><br><br>
        <input type="submit" value="Upload">
    </form>
</body>
</html>
'''
if __name__=='__main__':
    app.run(debug=True)

