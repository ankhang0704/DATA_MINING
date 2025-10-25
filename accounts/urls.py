from django.urls import path, include
from accounts import views

urlpatterns = [
    path("profile/", views.profile_view, name="profile"),
    path("settings/", views.settings_view, name="settings"),
    path("history/", views.activity_history_views, name="history")
]