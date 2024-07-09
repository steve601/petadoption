from flask import Flask,request,render_template
from source.logger import logging
from source.main_project.pipeline.predict_pipeline import PredicPipeline,UserData

app = Flask(__name__)

@app.route('/')
def homepage():
    return render_template('pet.html')

@app.route('/predict',methods = ['POST'])
def do_prediction():
    form_inp = UserData(
        pettype=request.form.get('pet'),
        breed=request.form.get('breed'),
        agemonths=request.form.get('age'),
        color=request.form.get('color'),
        size=request.form.get('size'),
        weightkg=request.form.get('weight'),
        vaccinated=request.form.get('vac'),
        healthcondition=request.form.get('hth'),
        adoptionfee=request.form.get('fee'),
        previousowner=request.form.get('own')
    )
    user_df = form_inp.get_data_as_df()
    
    predictpipe = PredicPipeline()
    y_pred = predictpipe.predict(user_df)
    
    msg = 'Pet is likely to be adopted' if y_pred == 1 else 'Pet is unlikely to be adopted'
    
    return render_template('pet.html',text=msg)

if __name__ == "__main__":
    app.run(host="0.0.0.0")