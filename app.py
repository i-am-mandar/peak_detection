#flask app

from flask import Flask, render_template, request
from flask_wtf import FlaskForm
from wtforms import FileField, SubmitField
from werkzeug.utils import secure_filename
import os
from wtforms.validators import InputRequired
import pandas as pd
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from pathlib import Path
import io
import base64
import numpy as np
from model_predict import predict
from scipy.signal import hilbert
from scipy.signal import find_peaks


app = Flask(__name__)
app.config['SECRET_KEY'] = 'supersecretkey'
app.config['UPLOAD_FOLDER'] = 'static/files'
app.config['SAVE_PLOT'] = 'static/plot'

class UploadFileForm(FlaskForm):
    file = FileField("File", validators=[InputRequired()])
    submit = SubmitField("Upload File")

@app.route('/', methods=['GET',"POST"])
@app.route('/home', methods=['GET',"POST"])
def home():

    form = UploadFileForm()
    
    if request.method == 'POST':
        if form.validate_on_submit():
            file = form.file.data # grab the file
            file.save(os.path.join(os.path.abspath(os.path.dirname(__file__)),app.config['UPLOAD_FOLDER'],secure_filename(file.filename))) # Then save the file
            
            df = pd.read_excel(file) # read the data
        
            df = df.iloc[0: ,17:]    # ignore the unecessary data
            f = df.iloc[0,0:] 
            n = f.size
            time=np.arange(n)
            predict_peak = predict(f)     #get this from cnn 
            
            analytical_signal = hilbert(f)
            env = np.abs(analytical_signal)
            x, _ = find_peaks(env, distance=n)
            
            predict_peak_image = render_image(time, f, predict_peak, env)
            find_peak_image = render_image(time, f, x, env)

        
            return render_template('plot.html', predict_peak = predict_peak, predict_peak_image = predict_peak_image, find_peak = x, find_peak_image = find_peak_image)
    else:
        return render_template('index.html', form=form)


def render_image(time, signal, peak, envelope):
    fig = Figure()
    axis = fig.add_subplot(1, 1, 1)
    axis.set_title("Detecting Peak")
    axis.set_xlabel("Time")
    axis.set_ylabel("Amplitude")
    axis.set_xlim(time[0], time[-1])
    axis.grid()
    axis.plot(time, signal)
    #axis.plot(time, env)
    axis.plot(peak, envelope[peak], "x")
    
    pngImage = io.BytesIO()
    FigureCanvas(fig).print_png(pngImage)
    
    # Encode PNG image to base64 string
    pngImageB64String = "data:image/png;base64,"
    pngImageB64String += base64.b64encode(pngImage.getvalue()).decode('utf8')
    
    return pngImageB64String

if __name__ == '__main__':
    app.run(debug=True)