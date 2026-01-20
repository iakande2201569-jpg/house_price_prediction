from flask import Flask, render_template, request
import joblib
import pandas as pd
import os

app = Flask(__name__)

# Load the trained model
model_path = os.path.join('model', 'house_price_model.pkl')
model = joblib.load(model_path)

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction_text = None
    
    if request.method == 'POST':
        try:
            # Get data from form
            overall_qual = int(request.form['OverallQual'])
            gr_liv_area = float(request.form['GrLivArea'])
            garage_cars = int(request.form['GarageCars'])
            full_bath = int(request.form['FullBath'])
            year_built = int(request.form['YearBuilt'])
            neighborhood = request.form['Neighborhood']

            # Create DataFrame for model input
            # Must match the columns used during training
            input_data = pd.DataFrame({
                'OverallQual': [overall_qual],
                'GrLivArea': [gr_liv_area],
                'GarageCars': [garage_cars],
                'FullBath': [full_bath],
                'YearBuilt': [year_built],
                'Neighborhood': [neighborhood]
            })

            # Make prediction
            prediction = model.predict(input_data)[0]
            prediction_text = f"Estimated House Price: ${prediction:,.2f}"
            
        except Exception as e:
            prediction_text = f"Error: {str(e)}"

    # List of Neighborhoods for the dropdown (from Ames dataset)
    neighborhoods = [
        "CollgCr", "Veenker", "Crawfor", "NoRidge", "Mitchel", "Somerst", 
        "NWAmes", "OldTown", "BrkSide", "Sawyer", "NridgHt", "NAmes", 
        "SawyerW", "IDOTRR", "MeadowV", "Edwards", "Timber", "Gilbert", 
        "StoneBr", "ClearCr", "NPkVill", "Blmngtn", "BrDale", "SWISU", "Blueste"
    ]
    neighborhoods.sort()

    return render_template('index.html', prediction=prediction_text, neighborhoods=neighborhoods)

if __name__ == "__main__":
    app.run(debug=True)