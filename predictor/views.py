from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import pandas as pd
import json

# Import các biến model và model_columns từ apps.py
from .apps import model, model_columns

def predict_page(request):
    """Render trang chủ (form nhập liệu)"""
    return render(request, "predictor/predict_page.html")

@csrf_exempt
def predict_request(request):
    """Xử lý yêu cầu dự đoán POST"""
    if request.method == 'POST':
        try:
            # 1. KIỂM TRA: Đảm bảo dữ liệu nhận được là chuỗi JSON hợp lệ
            raw_data = request.body.decode('utf-8')
            print(f"DEBUG: Chuỗi JSON nhận được: {raw_data}") # <--- KIỂM TRA DÒNG NÀY

            # 2. Phân tích chuỗi JSON
            data = json.loads(raw_data)
            print(f"DEBUG: Dữ liệu đã chuyển thành Dict: {data}") # <--- KIỂM TRA DÒNG NÀY
            # 1. Nhận dữ liệu từ body của request
            data = json.loads(request.body.decode('utf-8'))
            
            # 2. Chuyển dữ liệu JSON thành DataFrame
            query_df = pd.DataFrame([data]) # Dữ liệu nhận từ form là 1 dictionary, cần chuyển thành list chứa dictionary

            # 3. Tiền xử lý dữ liệu (One-Hot Encoding)
            query_processed = pd.get_dummies(query_df)

            # 4. Căn chỉnh các cột để khớp với mô hình (Điền 0 cho cột thiếu)
            query_aligned = query_processed.reindex(columns=model_columns, fill_value=0)

            # 5. Dự đoán
            prediction = model.predict(query_aligned)
            
            # 6. Chuyển kết quả (0 hoặc 1) thành dạng chữ
            result = '>50K' if prediction[0] == 1 else '<=50K'

            return JsonResponse({'prediction': result, 'status': 'success'})

        except Exception as e:
            return JsonResponse({'error': f'Lỗi xử lý dữ liệu: {str(e)}', 'status': 'error'}, status=400)
    
    return JsonResponse({'error': 'Chỉ chấp nhận phương thức POST'}, status=405)