from django.apps import AppConfig
import joblib
from pathlib import Path
from django.conf import settings # IMPORT DÒNG NÀY

# Khai báo biến toàn cục
model = None
model_columns = None

#Biến mô hình dự đoán thất nghiệp
model_unemployment = None
model_columns_unemployment = None
scaler_unemployment = None

class PredictorConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'predictor'

    def ready(self):
        # Biến mô hình dự đoán thu nhập
        global model
        global model_columns
        global scaler
        
        # Biến số mô hình dự đoán thất nghiệp
        global model_unemployment
        global model_columns_unemployment
        global scaler_unemployment

        # Tạo đường dẫn tuyệt đối đến các file mô hình thu nhập
        MODEL_PATH = settings.BASE_DIR / 'assets' / 'model_income' / 'adult_model.joblib'
        COLUMNS_PATH = settings.BASE_DIR / 'assets' / 'model_income' / 'model_columns.joblib'
        SCALER_PATH = settings.BASE_DIR / 'assets' / 'model_income' / 'scaler.joblib'

        # Tạo đường dẫn tuyệt đối đến các file mô hình thất nghiệp
        MODEL_UNEMPLOYMENT_PATH = settings.BASE_DIR / 'assets' / 'model_unemployment' / 'unemployment_model.joblib'
        COLUMNS_UNEMPLOYMENT_PATH = settings.BASE_DIR / 'assets' / 'model_unemployment' / 'unemployment_columns.joblib'
        SCALER_UNEMPLOYMENT_PATH = settings.BASE_DIR / 'assets' / 'model_unemployment' / 'unemployment_scaler.joblib'   

        try:
            # Tải mô hình, các cột và scaler khi ứng dụng đã sẵn sàng (Thu nhập)
            model = joblib.load(MODEL_PATH)
            model_columns = joblib.load(COLUMNS_PATH)
            scaler = joblib.load(SCALER_PATH)
            print("✅ [DJANGO ML] Mô hình Adult Income và Scaler đã được tải thành công!")

            # Tải mô hình, các cột và scaler khi ứng dụng đã sẵn sàng (Thất nghiệp)
            model_unemployment = joblib.load(MODEL_UNEMPLOYMENT_PATH)
            model_columns_unemployment = joblib.load(COLUMNS_UNEMPLOYMENT_PATH)
            scaler_unemployment = joblib.load(SCALER_UNEMPLOYMENT_PATH)
            print("✅ [DJANGO ML] Mô hình Unemployment và Scaler đã được tải thành công!")
            
        except FileNotFoundError as e:
            # SỬA LỖI: Báo cáo chính xác file nào bị thiếu
            print(f"❌ [DJANGO ML] LỖI FileNotFoundError: Không tìm thấy file. Chi tiết: {e}")
            print("Vui lòng kiểm tra lại đường dẫn trong 'assets' và chạy lại file train.")
        except Exception as e:
            print(f"❌ [DJANGO ML] Lỗi không xác định khi tải mô hình: {e}")