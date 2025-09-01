from django.urls import path
from .views import sensor_heart_rate, student, session, activity, data_index, \
    student_activities_page, sensor_heart_rate_activities_page, hrv_metrics_page, hrv_interpretation_page

urlpatterns = [

    ##### get all simple
    path('sensor-heart-rate', sensor_heart_rate),
    path('student', student),
    path('session', session),
    path('activity', activity),

    ### pages
    path('', data_index, name='data_index'),
    path('student-activities/', student_activities_page, name='student_activities_page'),
    path('sensor-heart-rate-activities/', sensor_heart_rate_activities_page, name='sensor_heart_rate_activities_page'),
    path('hrv-metrics/', hrv_metrics_page, name='hrv_metrics_page'),
    path('hrv-interpretation/', hrv_interpretation_page, name='hrv_interpretation_page'),
]
