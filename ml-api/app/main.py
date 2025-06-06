# Import FastAPI dan modul terkait
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse                    # Untuk mengembalikan respons HTML
from fastapi.templating import Jinja2Templates                # Untuk merender template HTML dengan Jinja2
from fastapi.staticfiles import StaticFiles                   # Untuk menyajikan file statis seperti CSS/JS

# Import instance FastAPI dari file lain, misalnya `load_model.py` (yang berisi endpoint ML)
from app.model.load_model import app as ml_app

# Inisialisasi aplikasi FastAPI utama
app = FastAPI()

# Mount atau gabungkan rute dari model ML ke path "/predict"
# Misalnya, akses POST /predict dari frontend akan diarahkan ke ml_app
app.mount("/predict", ml_app)

# Konfigurasi direktori template dan file statis
templates = Jinja2Templates(directory="templates")            # Lokasi template HTML
app.mount("/static", StaticFiles(directory="static"), name="static")  # File CSS, JS, gambar, dll.

# Endpoint utama untuk root URL ("/")
@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    # Render file templates/index.html saat user mengakses halaman utama
    return templates.TemplateResponse("index.html", {"request": request})
