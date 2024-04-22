from distutils.log import debug 
from fileinput import filename 
from flask import * 

from source_code.utils import ML_model

ml_model=ML_model()
app = Flask(__name__)   
  
@app.route('/')   
def main():   
    return render_template("index.html")   
  
@app.route('/success', methods = ['POST'])   
def success():   
    if request.method == 'POST':   
        f = request.files['file'] 
        f.save(r"C:\Users\srava\OneDrive\Documents\iitj\ProjectWork\car_crash\car_crash\input_files"+f.filename)  
        out_response=ml_model.get_prediction(r"C:\Users\srava\OneDrive\Documents\iitj\ProjectWork\car_crash\car_crash\input_files"+f.filename) 

        return render_template("Acknowledgement.html", name = out_response)  
         
  
if __name__ == '__main__':   
    app.run(debug=True)