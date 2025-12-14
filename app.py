from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd
import os
import datetime as dt

app = Flask(__name__)

# --- KONFIGURASI PATH ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'model_saham_date.pkl')
CSV_PATH = os.path.join(BASE_DIR, 'data_grafik_bbri.csv')

# --- LOAD MODEL (Cukup sekali di awal gapapa, kecuali kamu sering ganti model) ---
model = None
try:
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    print(f"✅ Model berhasil dimuat: {MODEL_PATH}")
except Exception as e:
    print(f"❌ Error memuat model: {e}")


# --- FUNGSI BANTUAN: BACA DATA TERBARU ---
# Kita bikin fungsi biar bisa dipanggil berulang kali
def get_latest_data():
    labels = []
    values = []
    last_p = 0.0
    
    try:
        if os.path.exists(CSV_PATH):
            # Selalu baca file CSV secara fresh dari disk
            df = pd.read_csv(CSV_PATH)
            
            if 'DateStr' in df.columns and 'Close' in df.columns:
                labels = df['DateStr'].astype(str).tolist()
                # Paksa jadi numeric
                df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
                df = df.dropna(subset=['Close'])
                values = df['Close'].tolist()
                
                if len(values) > 0:
                    last_p = float(values[-1])
        else:
            print("⚠️ CSV tidak ditemukan saat reload.")
            
    except Exception as e:
        print(f"❌ Error reload CSV: {e}")
        
    return labels, values, last_p


# --- ROUTES ---
@app.route('/')
def home():
    # Panggil fungsi reload data setiap kali halaman dibuka
    chart_labels, chart_values, _ = get_latest_data()
    
    return render_template('index.html', 
                           labels=chart_labels, 
                           values=chart_values)

@app.route('/predict', methods=['POST'])
def predict():
    # Panggil fungsi reload data juga saat mau prediksi
    chart_labels, chart_values, last_price = get_latest_data()
    
    error_context = {
        'labels': chart_labels,
        'values': chart_values,
        'input_val': request.form.get('tanggal_target', ''),
        'selisih': 0.0,
        'trend_status': 'Error',
        'trend_color': 'text-muted',
        'show_calc': False
    }

    if model is None:
        return render_template('index.html', prediction_text="Error: Model belum dimuat.", **error_context)

    try:
        input_date_str = request.form.get('tanggal_target', '')
        if not input_date_str:
            raise ValueError("Tanggal belum dipilih.")

        date_obj = dt.datetime.strptime(input_date_str, '%Y-%m-%d')
        date_ordinal = date_obj.toordinal()

        prediksi = model.predict([[date_ordinal]])
        harga_hasil = float(prediksi[0])

        # Hitung selisih pakai last_price yang BARU DI-LOAD tadi
        selisih = harga_hasil - float(last_price)
        
        if selisih > 0:
            trend_status = "NAIK (Bullish) 🚀"
            trend_color = "text-success"
        elif selisih < 0:
            trend_status = "TURUN (Bearish) 🔻"
            trend_color = "text-danger"
        else:
            trend_status = "STABIL ➖"
            trend_color = "text-muted"

        nilai_a = float(model.intercept_)
        nilai_b = float(model.coef_[0])

        return render_template('index.html',
                               prediction_text=f"Rp {harga_hasil:,.2f}",
                               tanggal_hasil=date_obj.strftime('%d %B %Y'),
                               input_val=input_date_str,
                               labels=chart_labels,
                               values=chart_values,
                               show_calc=True,
                               nilai_a=nilai_a,
                               nilai_b=nilai_b,
                               trend_status=trend_status,
                               trend_color=trend_color,
                               selisih=selisih,
                               ordinal_val=date_ordinal,
                               last_price=last_price)

    except Exception as e:
        print(f"ERROR: {e}")
        return render_template('index.html', 
                               prediction_text=f"Terjadi Kesalahan: {str(e)}", 
                               **error_context)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)