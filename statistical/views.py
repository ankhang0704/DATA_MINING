from django.shortcuts import render
from django.contrib.auth.decorators import login_required
from django.db.models import Count 
from django.apps import apps 

# Create your views here.
@login_required
def statistical_page(request):
    # Render the statistical page template

    #Yêu cầu 1: Hiển thị độ chính xác của mô hình dự đoán
    model_accuracy = 85.78

    #Yêu cầu 2: Biểu đô Feature Importance
    chart_path = 'models_logistic/feature_importance.png'

    # 2. DỮ LIỆU ĐỘNG (Phân tích Lịch sử)

        #Lấy id của user hiện tại
    current_user_id = request.user.id
    #Lấy model từ apps.py
    PredictionHistory = apps.get_model('predictor', 'PredictionHistory')
    # Lấy tất cả lịch sử của người dùng hiện tại
    user_history = PredictionHistory.objects.filter(user__id=current_user_id).order_by('-prediction_date')
    total_predictions = user_history.count()

    positive_rate = 0.0

    if total_predictions > 0:
        # Đếm số lần dự đoán là '>50K'
        positive_count = user_history.filter(output_result='>50K').count()
        
        # Yêu cầu 3: Tính tỷ lệ dự đoán >50K
        positive_rate = round((positive_count / total_predictions) * 100, 2)
        

    context = {
        'model_accuracy': model_accuracy,
        'chart_path': chart_path,
        'total_predictions': total_predictions,
        'positive_rate': positive_rate,
    }

    return render(request, "statistical/statistical_page.html", context)