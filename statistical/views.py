from django.shortcuts import render
from django.contrib.auth.decorators import login_required
from django.db.models import Count 
from django.apps import apps 
from collections import Counter

# Create your views here.
@login_required
def statistical_page(request):
    # Render the statistical page template

    #Yêu cầu 1: Hiển thị độ chính xác của mô hình dự đoán
    model_accuracy = 85.78
    # model_accuracy_unemployment = 78.45 # Thêm nếu muốn
    model_accuracy_unemployment = 93.00 # Thêm nếu muốn

    #Yêu cầu 2: Biểu đô Feature Importance
    chart_path = 'model_income/feature_importance.png'
    # chart_path_unemployment = 'model_unemployment/feature_importance.png' # Thêm nếu muốn
    chart_path_unemployment = 'model_unemployment/unemployment_feature_importance.png' # Thêm nếu muốn
    # 2. DỮ LIỆU ĐỘNG (Phân tích Lịch sử)
        #Lấy id của user hiện tại
    current_user_id = request.user.id
    #Lấy model từ apps.py
    PredictionHistory = apps.get_model('predictor', 'PredictionHistory')
    UnemploymentHistory = apps.get_model('predictor', 'UnemploymentHistory')
    # Lấy tất cả lịch sử của người dùng hiện tại
    income_history = PredictionHistory.objects.filter(user__id=current_user_id).order_by('-prediction_date')
    total_predictions = income_history.count()

    positive_rate = 0.0

    if total_predictions > 0:
        # Đếm số lần dự đoán là '>50K'
        positive_count = income_history.filter(output_result='>50K').count()
        
        # Yêu cầu 3: Tính tỷ lệ dự đoán >50K
        positive_rate = round((positive_count / total_predictions) * 100, 2)
        
    # PHÂN TÍCH LỊCH SỬ THẤT NGHIỆP (UNEMPLOYMENT) - MỚI
    # -----------------------------------------------
    unemployment_history = UnemploymentHistory.objects.filter(user__id=current_user_id)
    total_unemployment_predictions = unemployment_history.count()
    unemployed_rate = 0.0
    if total_unemployment_predictions > 0:
        unemployed_count = unemployment_history.filter(output_result='Thất nghiệp').count()
        unemployed_rate = round((unemployed_count / total_unemployment_predictions) * 100, 2) # Tỷ lệ dự đoán thất nghiệp

    all_inputs = []
    for record in income_history:
        all_inputs.append(record.input_data)
    for record in unemployment_history:
        all_inputs.append(record.input_data)
    
    occupation_counts = Counter(item.get('occupation', 'N/A') for item in all_inputs if 'occupation' in item)
    age_group_counts = Counter(item.get('age_group', 'N/A') for item in all_inputs if 'age_group' in item)

    top_occupations = occupation_counts.most_common(3)
    top_age_groups = age_group_counts.most_common(3)
    # -----------------------------------------------
    
    context = {
        # Income Stats
        'model_accuracy': model_accuracy,
        'chart_path': chart_path,
        'total_predictions': total_predictions,
        'positive_rate': positive_rate,

        # Unemployment Stats - MỚI
        'total_unemployment_predictions': total_unemployment_predictions,
        'unemployed_rate': unemployed_rate,
        'chart_path_unemployment': chart_path_unemployment,
        'model_accuracy_unemployment': model_accuracy_unemployment,


        # Input Analysis - MỚI
        'top_occupations': top_occupations,
        'top_age_groups': top_age_groups,
        'has_history': total_predictions > 0 or total_unemployment_predictions > 0, # Cờ để kiểm tra có lịch sử không
    }

    return render(request, "statistical/statistical_page.html", context)