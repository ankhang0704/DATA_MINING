from django.apps import AppConfig
import joblib
from pathlib import Path
from django.conf import settings # IMPORT DÒNG NÀY

# Khai báo biến toàn cục
model = None
model_columns = None

class PredictorConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'predictor'

    # def ready(self):
    #     global model
    #     global model_columns
        
    #     # Tạo đường dẫn tuyệt đối đến các file mô hình
    #     MODEL_PATH = settings.BASE_DIR / 'assets' / 'models' / 'adult_model.joblib'
    #     COLUMNS_PATH = settings.BASE_DIR / 'assets' / 'models' / 'model_columns.joblib'

    #     try:
    #         # Tải mô hình và các cột khi ứng dụng đã sẵn sàng
    #         model = joblib.load(MODEL_PATH)
    #         model_columns = joblib.load(COLUMNS_PATH)
    #         print("✅ [DJANGO ML] Mô hình Adult Income đã được tải thành công!")
    #     except FileNotFoundError:
    #         print(f"❌ [DJANGO ML] LỖI: Không tìm thấy file mô hình tại {MODEL_PATH} hoặc {COLUMNS_PATH}.")
    #         print("Vui lòng đảm bảo bạn đã chạy 'train_model.py' và đặt các file joblib vào thư mục assets/models.")
    #     except Exception as e:
    #         print(f"❌ [DJANGO ML] Lỗi không xác định khi tải mô hình: {e}")

    def ready(self):
        global model
        global model_columns
        global scaler
        
        # Tạo đường dẫn tuyệt đối đến các file mô hình
        MODEL_PATH = settings.BASE_DIR / 'assets' / 'models_logistic' / 'adult_model.joblib'
        COLUMNS_PATH = settings.BASE_DIR / 'assets' / 'models_logistic' / 'model_columns.joblib'
        SCALER_PATH = settings.BASE_DIR / 'assets' / 'models_logistic' / 'scaler.joblib'

        try:
            # Tải mô hình, các cột và scaler khi ứng dụng đã sẵn sàng
            model = joblib.load(MODEL_PATH)
            model_columns = joblib.load(COLUMNS_PATH)
            scaler = joblib.load(SCALER_PATH)
            print("✅ [DJANGO ML] Mô hình Adult Income và Scaler đã được tải thành công!")
        except FileNotFoundError:
            print(f"❌ [DJANGO ML] LỖI: Không tìm thấy file mô hình tại {MODEL_PATH}, {COLUMNS_PATH} hoặc {SCALER_PATH}.")
            print("Vui lòng đảm bảo bạn đã chạy 'train_model.py' và đặt các file joblib vào thư mục assets/models.")
        except Exception as e:
            print(f"❌ [DJANGO ML] Lỗi không xác định khi tải mô hình: {e}")