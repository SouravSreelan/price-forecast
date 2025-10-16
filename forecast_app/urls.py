from django.urls import path
from . import views

urlpatterns = [
    path('forecast_plot/', views.forecast_plot_view, name='forecast_plot')
]
