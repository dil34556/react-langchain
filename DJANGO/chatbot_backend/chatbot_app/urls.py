from django.urls import path
from .views import chatbot

urlpatterns = [
    path('api/chatbot/', chatbot, name='chatbot'),
]
