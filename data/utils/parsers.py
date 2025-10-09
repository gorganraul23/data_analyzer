from django.conf import settings
from django.db import connection

def parse_bool(v: str | None):
    if v is None:
        return None
    s = str(v).strip().lower()
    if s in ('1', 'true', 't', 'yes', 'y', 'on'):
        return True
    if s in ('0', 'false', 'f', 'no', 'n', 'off'):
        return False

    return None

def excel_text(s):
    if s is None:
        return ""
    s = str(s)

    return "" + s

def column_exists(table_name, column_name):
    with connection.cursor() as cursor:
        db_name = settings.DATABASES["default"]["NAME"]
        cursor.execute("""
            SELECT column_name 
            FROM information_schema.columns 
            WHERE table_schema = %s AND table_name = %s AND column_name = %s
        """, [db_name, table_name, column_name])
        return cursor.fetchone() is not None