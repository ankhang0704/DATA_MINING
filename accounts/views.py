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

@login_required
def profile_view(request):
    user = request.user
    google_account = None  # Khởi tạo biến

    # Tìm tài khoản google của user
    if user.is_authenticated:
        try:
            # Lấy tài khoản social có provider là 'google'
            google_account = user.socialaccount_set.get(provider='google')
        except:
            # Bỏ qua nếu không tìm thấy
            google_account = None
            
    # Các logic khác để lấy thông tin (URL đã quét, tỷ lệ an toàn,...)
    # scanned_urls = ...
    # safe_rate = ...

    context = {
        'user': user,
        'google_account': google_account, # Truyền biến này vào template
        # 'scanned_urls': scanned_urls,
        # 'safe_rate': safe_rate,
    }

    return render(request, 'users/profile.html', context)

@login_required
def settings_view(request):
    return render(request, 'users/settings.html')
@login_required
def activity_history_views(request):
    return render(request, 'users/activity_history.html')