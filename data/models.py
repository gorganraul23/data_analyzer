from django.db import models


class Student(models.Model):

    nickname = models.CharField(db_column='nickname', max_length=10, blank=True)
    yearBirth = models.CharField(db_column='yearBirth', max_length=4, blank=True)
    sex = models.CharField(db_column='sex', max_length=6, blank=True)
    handUsed = models.CharField(db_column='handUsed', max_length=5, blank=True)

    class Meta:
        db_table = 'student'
        managed = False


class Session(models.Model):

    sessionName = models.CharField(db_column='sessionName', max_length=20, blank=True)
    tInitSession = models.BigIntegerField(db_column='tInitSession', null=True, blank=True)
    tEndSession = models.BigIntegerField(db_column='tEndSession', null=True, blank=True)
    done = models.BooleanField(db_column='done', default=False)
    student = models.ForeignKey(
        Student, on_delete=models.DO_NOTHING,
        db_column='studentID', related_name='sessions',
        db_constraint=False
    )

    class Meta:
        db_table = 'session'
        managed = False


class Activity(models.Model):

    activityType = models.CharField(db_column='activitytype', max_length=20, blank=True)
    tInitActivity = models.BigIntegerField(db_column='tinitactivity', null=True, blank=True)
    tEndActivity = models.BigIntegerField(db_column='tendactivity', null=True, blank=True)
    done = models.BooleanField(db_column='done', default=False)
    level = models.IntegerField(db_column='level', null=True, blank=True)
    session = models.ForeignKey(
        Session, on_delete=models.DO_NOTHING,
        db_column='sessionID', related_name='activities',
        db_constraint=False
    )

    class Meta:
        db_table = 'activity'
        managed = False

class SensorHeartRate(models.Model):

    value_heart_rate = models.IntegerField(db_column='value_heart_rate', null=True, blank=True)
    status_heart_rate = models.SmallIntegerField(db_column='status_heart_rate', null=True, blank=True)
    value_ibi_0 = models.IntegerField(db_column='value_ibi_0', null=True, blank=True)
    status_ibi_0 = models.SmallIntegerField(db_column='status_ibi_0', null=True, blank=True)
    value_ibi_1 = models.IntegerField(db_column='value_ibi_1', null=True, blank=True)
    status_ibi_1 = models.SmallIntegerField(db_column='status_ibi_1', null=True, blank=True)
    value_ibi_2 = models.IntegerField(db_column='value_ibi_2', null=True, blank=True)
    status_ibi_2 = models.SmallIntegerField(db_column='status_ibi_2', null=True, blank=True)
    value_ibi_3 = models.IntegerField(db_column='value_ibi_3', null=True, blank=True)
    status_ibi_3 = models.SmallIntegerField(db_column='status_ibi_3', null=True, blank=True)
    timestamp = models.BigIntegerField(db_column='timestamp', null=True, blank=True)

    session = models.ForeignKey(
        Session, on_delete=models.DO_NOTHING,
        db_column='id_session', related_name='heart_rates',
        db_constraint=False
    )
    activity = models.ForeignKey(
        Activity, on_delete=models.DO_NOTHING,
        db_column='id_activity', related_name='heart_rates',
        db_constraint=False
    )
    student = models.ForeignKey(
        Student, on_delete=models.DO_NOTHING,
        db_column='id_student', related_name='heart_rates',
        db_constraint=False
    )

    class Meta:
        db_table = 'sensor_heart_rate'
        managed = False
