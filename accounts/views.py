import re
import os
import json
import logging
from django.utils.timezone import datetime
from django.http import HttpResponse
from django.shortcuts import render
from django.contrib.auth import login
from django.contrib.auth.decorators import login_required
from django.conf import settings
from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth.decorators import login_required
# Create your views here.
from django.apps import apps # Cần import

@login_required
def activity_history_views(request):

    #Lấy id của user hiện tại
    current_user_id = request.user.id
    #Lấy model từ apps.py
    PredictionHistory = apps.get_model('predictor', 'PredictionHistory')
    #  Truy vấn Lịch sử: Lấy tất cả bản ghi của user hiện tại, sắp xếp mới nhất lên trước
    user_history = PredictionHistory.objects.filter(user__id=current_user_id).order_by('-prediction_date')

    context = {
        'history': user_history,
        'count': user_history.count()
    }

    return render(request, 'users/activity_history.html',  context)