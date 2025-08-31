from django.contrib import admin
from .models import SensorHeartRate, Activity, Session, Student


admin.site.register(SensorHeartRate)
admin.site.register(Student)
admin.site.register(Session)
admin.site.register(Activity)
