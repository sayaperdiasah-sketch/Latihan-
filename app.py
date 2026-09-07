import os
import sys
from flask import Flask, request, jsonify, render_template
from openai import OpenAI

app = Flask(__name__)

# 1. Mengambil token Hugging Face secara aman
hf_token = os.environ.get("HF_TOKEN")
if not hf_token:
    print("❌ EROR: Token 'HF_TOKEN' tidak terdeteksi di server environment!")
    sys.exit(1)

# 2. Inisialisasi OpenAI SDK Client untuk Router API Hugging Face
client = OpenAI(
    base_url="https://huggingface.co",
    api_key=hf_token,
)

# 3. Routing Halaman Utama (Sekarang memanggil file terpisah index.html secara bersih)
@app.route('/')
def home():
    return render_template('index.html')

# 4. API End-point untuk Memproses Jawaban AI Medis
@app.route('/get_response', methods=['POST'])
def get_response():
    data = request.json
    user_message = data.get("message", "")

    try:
        completion = client.chat.completions.create(
            model="OpenMedZoo/MedGo:featherless-ai",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Anda adalah asisten virtual kesehatan AI medis. Jawablah menggunakan Bahasa Indonesia yang ramah. "
                        "Jika pengguna berkonsultasi mengenai gejala penyakit fisik atau mental, Anda wajib memberikan minimal 3 kemungkinan penyebab masalah kesehatan tersebut sebagai edukasi umum. "
                        "Di akhir paragraf, Anda wajib mengingatkan pengguna secara tegas untuk melakukan verifikasi pada label kemasan fisik obat dan berkonsultasi langsung ke dokter nyata demi mendapatkan diagnosis klinis yang akurat."
                    )
                },
                {"role": "user", "content": user_message}
            ],
            temperature=0.3,
            max_tokens=600
        )
        ai_reply = completion.choices.message.content
    except Exception as e:
        ai_reply = f"Gagal mendapatkan respon dari AI Server: {str(e)}"

    return jsonify({"reply": ai_reply})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)

