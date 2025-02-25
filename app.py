from flask import Flask, render_template, request
from src.pipeline.prediction_pipeline import CustomData, Prediction_pipeline
from src.pipeline.train import training_pipeline
app = Flask(__name__)

@app.route('/train2')
def home_page():
    training_pipeline()
    return 'training completed'


@app.route('/predict', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('form.html')
    else:
        data = CustomData(
            carat=float(request.form.get('carat')),
            depth=float(request.form.get('depth')),
            table=float(request.form.get('table')),
            x=float(request.form.get('x')),
            y=float(request.form.get('y')),
            z=float(request.form.get('z')),
            cut=request.form.get('cut'),
            color=request.form.get('color'),
            clarity=request.form.get('clarity')
        )
        final_data = data.get_data_as_data_frame()
        
        predict_pipeline = Prediction_pipeline()
        pred = predict_pipeline.predict(final_data)
        
        results = round(pred[0], 2)
        
        return render_template('form.html', final_result=results)

if __name__=="__main__":

    app.run(host="0.0.0.0", port=5000, debug= False)
            


            
     
            
            
            
            
            
            
        
            
    