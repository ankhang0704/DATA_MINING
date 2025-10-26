from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import pandas as pd
import json
from .models import PredictionHistory
import numpy as np
# Import các biến model và model_columns từ apps.py
from .apps import model, model_columns, scaler

def predict_page(request):
    """Render trang chủ (form nhập liệu)"""
    return render(request, "predictor/predict_page.html")

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
# @csrf_exempt
# def predict_request(request):
#     """Xử lý yêu cầu dự đoán POST"""
#     if request.method == 'POST':
#         try:
#             # 1. KIỂM TRA: Đảm bảo dữ liệu nhận được là chuỗi JSON hợp lệ
#             raw_data = request.body.decode('utf-8')
#             print(f"DEBUG: Chuỗi JSON nhận được: {raw_data}") # <--- KIỂM TRA DÒNG NÀY

#             # 2. Phân tích chuỗi JSON
#             data = json.loads(raw_data)
#             print(f"DEBUG: Dữ liệu đã chuyển thành Dict: {data}") # <--- KIỂM TRA DÒNG NÀY
#             # 1. Nhận dữ liệu từ body của request
#             data = json.loads(request.body.decode('utf-8'))
            
#             # 2. Chuyển dữ liệu JSON thành DataFrame
#             query_df = pd.DataFrame([data]) # Dữ liệu nhận từ form là 1 dictionary, cần chuyển thành list chứa dictionary

#             # 3. Tiền xử lý dữ liệu (One-Hot Encoding)
#             query_processed = pd.get_dummies(query_df)

#             # 4. Căn chỉnh các cột để khớp với mô hình (Điền 0 cho cột thiếu)
#             query_aligned = query_processed.reindex(columns=model_columns, fill_value=0)

#             # 5. Dự đoán
#             prediction = model.predict(query_aligned)
            
#             # 6. Chuyển kết quả (0 hoặc 1) thành dạng chữ
#             result = '>50K' if prediction[0] == 1 else '<=50K'

#             return JsonResponse({'prediction': result, 'status': 'success'})

#         except Exception as e:
#             return JsonResponse({'error': f'Lỗi xử lý dữ liệu: {str(e)}', 'status': 'error'}, status=400)
    
#     return JsonResponse({'error': 'Chỉ chấp nhận phương thức POST'}, status=405)