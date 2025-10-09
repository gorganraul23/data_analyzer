from rest_framework import serializers
from .models import SensorHeartRate, Student, Session, Activity
from .utils.date_converter import EpochMsDateTimeField

class SensorHeartRateSerializer(serializers.ModelSerializer):

    timestamp = EpochMsDateTimeField(read_only=True)

    class Meta:
        model = SensorHeartRate
        fields = ('id',
                  'session',
                  'activity',
                  'student',
                  'timestamp',
                  'value_heart_rate',
                  'status_heart_rate',
                  'value_ibi_0',
                  'status_ibi_0',
                  'value_ibi_1',
                  'status_ibi_1',
                  'value_ibi_2',
                  'status_ibi_2',
                  'value_ibi_3',
                  'status_ibi_3',
                  "value_ibi_depr",
                  "status_ibi_depr",
                  )

class StudentSerializer(serializers.ModelSerializer):
    class Meta:
        model = Student
        fields = ('id',
                  'nickname',
                  'yearBirth',
                  'sex',
                  'handUsed'
                  )

class SessionSerializer(serializers.ModelSerializer):

    tInitSession = EpochMsDateTimeField(read_only=True)
    tEndSession = EpochMsDateTimeField(read_only=True)

    class Meta:
        model = Session
        fields = ('id',
                  'sessionName',
                  'student',
                  'tInitSession',
                  'tEndSession',
                  'done',
                  )

class ActivitySerializer(serializers.ModelSerializer):

    tInitActivity = EpochMsDateTimeField(read_only=True)
    tEndActivity = EpochMsDateTimeField(read_only=True)
    performance = serializers.FloatField(read_only=True, allow_null=True, required=False)

    class Meta:
        model = Activity
        fields = ('id',
                  'activityType',
                  'session',
                  'tInitActivity',
                  'tEndActivity',
                  'done',
                  'level',
                  'performance'
                  )

class StudentActivitiesSerializer(serializers.ModelSerializer):
    #student_id = serializers.IntegerField(source="session.student_id", read_only=True)
    student_id = serializers.IntegerField(source="session__student_id", read_only=True)
    session_id = serializers.IntegerField(read_only=True)
    activity_id = serializers.IntegerField(source="id", read_only=True)
    done = serializers.BooleanField(read_only=True)
    row_count = serializers.IntegerField(read_only=True)
    performance = serializers.FloatField(read_only=True, allow_null=True, required=False)

    class Meta:
        model = Activity
        fields = ("student_id",
                  "session_id",
                  "activity_id",
                  "activityType",
                  "done",
                  "level",
                  "performance",
                  "row_count")

class SensorHeartRateActivitiesSerializer(serializers.ModelSerializer):
    student_id = serializers.IntegerField(read_only=True)
    session_id = serializers.IntegerField(read_only=True)
    activity_id = serializers.IntegerField(read_only=True)
    timestamp = EpochMsDateTimeField(read_only=True)
   # performance = serializers.IntegerField(source="activity__performance", read_only=True, required=False)

    class Meta:
        model  = SensorHeartRate
        fields = ("id",
                  "activity_id",
                  "session_id",
                  "student_id",
                  "value_heart_rate",
                  "status_heart_rate",
                  "value_ibi_0",
                  "status_ibi_0",
                  "value_ibi_1",
                  "status_ibi_1",
                  "value_ibi_2",
                  "status_ibi_2",
                  "value_ibi_3",
                  "status_ibi_3",
                  "value_ibi_depr",
                  "status_ibi_depr",
                  'timestamp',
                  #'performance'
                  )
