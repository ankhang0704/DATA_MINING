from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import pandas as pd
import json
from .models import PredictionHistory, UnemploymentHistory
import numpy as np
# Import các biến model và model_columns từ apps.py
from .apps import model, model_columns, scaler
from .apps import model_unemployment, model_columns_unemployment, scaler_unemployment

import traceback

# Trang dự đoán thu nhập
def predict_page(request):
    """Render trang chủ (form nhập liệu)"""
    return render(request, "predictor/predict_page.html")

# Trang dự đoán thất nghiệp
def predict_unemployment_page(request):
    """Render trang dự đoán thất nghiệp (form nhập liệu)"""
    return render(request, "predictor/predict_unemployment_page.html")

# Trang sử lý yêu cầu dự đoán thu nhập
@csrf_exempt
def predict_request(request):
    """Xử lý yêu cầu dự đoán POST, trả về kết quả và xác suất."""
    if request.method == 'POST':
        try:
            # 1. Nhận và xử lý dữ liệu
            data_dict = json.loads(request.body.decode('utf-8'))
            query_df = pd.DataFrame([data_dict]) 
            query_processed = pd.get_dummies(query_df)
            query_aligned = query_processed.reindex(columns=model_columns, fill_value=0)

            # CHUẨN HÓA DỮ LIỆU ĐẦU VÀO
            # Dùng scaler đã được huấn luyện (fit) từ trước
            query_scaled = scaler.transform(query_aligned)

            # 2. Dự đoán và Trích xuất Xác suất (Nâng cao)
            # prediction[0] là kết quả phân loại (0 hoặc 1)
            prediction = model.predict(query_scaled)[0]
            
            # predict_proba trả về xác suất cho TẤT CẢ các lớp: [[P(<=50K), P(>50K)]]
            proba = model.predict_proba(query_scaled)[0]
            
            # Xác suất cho lớp >50K (Lớp có index 1)
            prob_over_50k = float(proba[1])
            
            # Làm tròn xác suất và chuyển sang phần trăm
            prob_percent = np.round(prob_over_50k * 100, 2)
            
            # Chuyển kết quả (0 hoặc 1) thành dạng chữ
            result_label = '>50K' if prediction == 1 else '<=50K'
            
            # 3. LƯU LỊCH SỬ DỰ ĐOÁN (Cho tính năng thống kê)
            print(f"DEBUG: User đã đăng nhập: {request.user.is_authenticated}")
            if request.user.is_authenticated:
                print(f"DEBUG: Tên người dùng đang lưu: {request.user.username}")
                PredictionHistory.objects.create(
                    user=request.user,
                    input_data=data_dict, # Lưu trữ input dưới dạng dict
                    output_result=result_label,
                    probability=prob_percent,
                )

            # 4. Trả kết quả JSON về Frontend
            return JsonResponse({
                'prediction': result_label,
                'probability': prob_percent, # <--- GỬI XÁC SUẤT VỀ FRONTEND
                'status': 'success'
            })

        except Exception as e:
            # ... (Xử lý lỗi) ...
            return JsonResponse({'error': f'Lỗi xử lý dữ liệu: {str(e)}', 'status': 'error'}, status=400)
    
    return JsonResponse({'error': 'Chỉ chấp nhận phương thức POST'}, status=405)

# Trang sử lý yêu cầu dự đoán thất nghiệp
@csrf_exempt
def predict_unemployment_request(request):
    """Xử lý yêu cầu dự đoán THẤT NGHIỆP (POST) và trả về kết quả JSON."""
    if request.method == 'POST':
        try:
            data_dict = json.loads(request.body.decode('utf-8'))
            
            # --- 1. Kỹ thuật Đặc trưng (Giống hệt mô hình thu nhập) ---
            # (Gộp 'capital-gain'/'capital-loss' thành 'net_capital')
            data_dict['net_capital'] = data_dict.get('capital-gain', 0) - data_dict.get('capital-loss', 0)
            
            # (Rời rạc hóa (Binning) 'age')
            age = int(data_dict.get('age', 0))
            age_bins = [16, 25, 40, 55, 65, 100]
            age_labels = ['Age_17-25', 'Age_26-40', 'Age_41-55', 'Age_56-65', 'Age_66+']
            data_dict['age_group'] = pd.cut([age], bins=age_bins, labels=age_labels, right=True)[0]

            # Xóa các cột gốc
            data_dict.pop('age', None)
            data_dict.pop('capital-gain', None)
            data_dict.pop('capital-loss', None)
            # (Không cần xóa 'workclass' vì nó đã bị loại khỏi model_columns_unemployment)
            
            # --- 2. Xử lý OHE và Căn chỉnh (Dùng cột của mô hình THẤT NGHIỆP) ---
            query_df = pd.DataFrame([data_dict]) 
            query_processed = pd.get_dummies(query_df)
            query_aligned = query_processed.reindex(columns=model_columns_unemployment, fill_value=0)

            # --- 3. CHUẨN HÓA DỮ LIỆU (Dùng scaler của mô hình THẤT NGHIỆP) ---
            query_scaled = scaler_unemployment.transform(query_aligned)

            # --- 4. Dự đoán (Dùng mô hình THẤT NGHIỆP) ---
            prediction = model_unemployment.predict(query_scaled)[0] 
            proba = model_unemployment.predict_proba(query_scaled)[0]
            
            # Tính xác suất (Lớp 1 là 'Thất nghiệp')
            # --- SỬA LOGIC TÍNH XÁC SUẤT HIỂN THỊ ---
            if prediction == 1: # Nếu dự đoán là Thất nghiệp (lớp 1)
                prob_for_prediction = float(proba[1]) # Lấy xác suất của lớp 1
            else: # Nếu dự đoán là Có việc làm (lớp 0)
                prob_for_prediction = float(proba[0]) # Lấy xác suất của lớp 0

            prob_percent = np.round(prob_for_prediction * 100, 2)
            
            result_label = 'Thất nghiệp' if prediction == 1 else 'Có việc làm'

            # (Tùy chọn: Lưu lịch sử này vào một Model khác hoặc Model chung)
            if request.user.is_authenticated:
                UnemploymentHistory.objects.create(
                    user=request.user,
                    input_data=data_dict, # Lưu trữ input dưới dạng dict
                    output_result=result_label,
                    probability=prob_percent,
                )
            # --- 5. Trả kết quả JSON ---
            return JsonResponse({
                'prediction': result_label,
                'probability': prob_percent, 
                'status': 'success'
            })

        except Exception as e:
            # --- BẮT ĐẦU DEBUG ---
            # In lỗi chi tiết ra Terminal
            print("===================================================================")
            print(f"❌ LỖI NGHIÊM TRỌNG TRONG predict_unemployment_request:")
            print(f"Lỗi: {e}")
            print(f"Loại Lỗi: {type(e)}")
            print("--- Chi tiết Traceback ---")
            traceback.print_exc() # In ra toàn bộ Traceback
            print("===================================================================")
            # --- KẾT THÚC DEBUG ---
            return JsonResponse({'error': f'Lỗi xử lý dữ liệu (Thất nghiệp): {str(e)}', 'status': 'error'}, status=400)
    
    return JsonResponse({'error': 'Chỉ chấp nhận phương thức POST'}, status=405)