from django.urls import path
from api.views import HomeView

urlpatterns = [
    path('', HomeView.as_view())
]