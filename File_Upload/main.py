from flask import Flask, render_template
from flask_wtf import FlaskForm
from wtforms import FileField, SubmitField
from werkzeug.utils import secure_filename
import os
from wtforms.validators import InputRequired
import pandas as pd
import matplotlib.pyplot as plt

app = Flask(__name__)
app.config['SECRET_KEY'] = 'supersecretkey'
app.config['UPLOAD_FOLDER'] = 'static/files'

class UploadFileForm(FlaskForm):
    file = FileField("File", validators=[InputRequired()])
    submit = SubmitField("Upload File")

@app.route('/', methods=['GET',"POST"])
@app.route('/home', methods=['GET',"POST"])
def home():
    form = UploadFileForm()
    input = []
    amp = []
    if form.validate_on_submit():
        file = form.file.data # First grab the file
        file.save(os.path.join(os.path.abspath(os.path.dirname(__file__)),app.config['UPLOAD_FOLDER'],secure_filename(file.filename))) # Then save the file
        data = pd.read_excel(file)

        for i, row in data.iterrows(): # read each row from excel
            #for j, column in row.iteritems():
                print(row.input, row.result)
                input.append(row.input) 
                amp.append(row.result) 

        plt.bar(input, amp) # plot based on input and result
        
        plt.title('Input Vs peak')
        plt.xlabel('Input')
        plt.ylabel('Peak')
        plt.show()

        return "File has been uploaded."
    return render_template('index.html', form=form)

if __name__ == '__main__':
    app.run(debug=True)

