from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd
import os
import traceback

app = Flask(__name__)

# --- CONFIG ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'model_saham_regresi.pkl')
CSV_PATH = os.path.join(BASE_DIR, 'data_grafik_bbri.csv')

# --- LOAD MODEL ---
model = None
try:
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
except Exception as e:
    print(f"Error loading model: {e}")

# --- LOAD CSV GRAFIK ---
chart_labels = []
chart_values = []
try:
    if os.path.exists(CSV_PATH):
        df = pd.read_csv(CSV_PATH)
        chart_labels = df['DateStr'].astype(str).tolist()
        chart_values = df['Close'].tolist()
except Exception:
    pass

@app.route('/')
def home():
    return render_template('index.html', labels=chart_labels, values=chart_values)

@app.route('/predict', methods=['POST'])
def predict():
    # Default params biar gak error
    common_params = {
        'labels': chart_labels, 
        'values': chart_values, 
        'input_val': request.form.get('harga_open', '')
    }

    if model is None:
        return render_template('index.html', prediction_text='Error: Model belum siap.', **common_params)

    try:
        # 1. Ambil Input
        raw_val = request.form.get('harga_open', '').strip()
        if not raw_val: raise ValueError("Input kosong.")
        harga_open = float(raw_val)

        # 2. Prediksi
        prediksi = model.predict(np.array([[harga_open]], dtype=float))
        harga_close = float(np.ravel(prediksi)[0])

        # 3. Ambil Rumus (Intercept & Slope)
        a = float(model.intercept_)
        b = float(model.coef_.flatten()[0])

        # --- LOGIKA BARU: ANALISA TREND (BULLISH/BEARISH) ---
        selisih = harga_close - harga_open
        trend_status = ""
        trend_color = ""
        penjelasan_trend = ""

        if selisih > 0:
            trend_status = "BULLISH (Naik) 🚀"
            trend_color = "text-success" # Hijau Bootstrap
            penjelasan_trend = "Berdasarkan data historis, saham ini memiliki kecenderungan naik dari pagi ke sore hari (Intraday Gain)."
        elif selisih < 0:
            trend_status = "BEARISH (Turun) 🔻"
            trend_color = "text-danger" # Merah Bootstrap
            penjelasan_trend = "Berdasarkan data historis, saham ini memiliki kecenderungan koreksi/turun menjelang penutupan pasar (Profit Taking)."
        else:
            trend_status = "SIDEWAYS (Tetap) ➖"
            trend_color = "text-muted"
            penjelasan_trend = "Harga penutupan diprediksi sama dengan harga pembukaan."

        # Return ke HTML
        return render_template('index.html',
                               prediction_text=f"Rp {harga_close:,.2f}",
                               input_val=harga_open,
                               labels=chart_labels,
                               values=chart_values,
                               show_calc=True,
                               nilai_a=a,
                               nilai_b=b,
                               # Kirim data trend baru ini:
                               trend_status=trend_status,
                               trend_color=trend_color,
                               penjelasan_trend=penjelasan_trend,
                               selisih=selisih)

    except Exception as e:
        return render_template('index.html', prediction_text=f'Error: {str(e)}', **common_params)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)