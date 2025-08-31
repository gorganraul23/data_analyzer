from datetime import datetime, timezone
from django.utils.dateparse import parse_datetime
from rest_framework import serializers

try:
    from zoneinfo import ZoneInfo
    def get_local_tz():
        from django.conf import settings
        return ZoneInfo(getattr(settings, "TIME_ZONE", "UTC"))
except Exception:
    import pytz
    def get_local_tz():
        from django.conf import settings
        return pytz.timezone(getattr(settings, "TIME_ZONE", "UTC"))

class EpochMsDateTimeField(serializers.Field):

    def to_representation(self, value):
        if value in (None, "", 0):
            return None
        try:
            ms = int(value)
        except (TypeError, ValueError):
            return None
        # assume ms since Unix epoch in UTC
        dt_utc = datetime.fromtimestamp(ms / 1000.0, tz=timezone.utc)
        dt_local = dt_utc.astimezone(get_local_tz())
        return dt_local.isoformat()

        # if value in (None, "", 0):
        #     return None
        # try:
        #     ms = int(value)
        # except (TypeError, ValueError):
        #     return None
        #
        # dt_utc = datetime.fromtimestamp(ms / 1000.0, tz=timezone.utc)
        # dt_local = dt_utc.astimezone(get_local_tz())
        #
        # # Return "dd-mm-yyyy hh:mm:ss.mmmm"
        # milli = int(dt_local.microsecond / 1000)  # 0..999
        #
        # return dt_local.strftime("%d-%m-%Y %H:%M:%S") + f".{milli:03d}"

    def to_internal_value(self, data):

        if data in (None, ""):
            return None

        try:
            return int(data)
        except (TypeError, ValueError):
            pass

        dt = parse_datetime(str(data))
        if dt is None:
            raise serializers.ValidationError("Invalid datetime format.")
        if dt.tzinfo is None:
            dt = get_local_tz().localize(dt) if hasattr(get_local_tz(), "localize") else dt.replace(tzinfo=get_local_tz())
        dt_utc = dt.astimezone(timezone.utc)

        return int(dt_utc.timestamp() * 1000)