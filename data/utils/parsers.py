from django.conf import settings
from django.db import connection
import matplotlib.colors as mcolors

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

def performance_color(perf):
    cmap = mcolors.LinearSegmentedColormap.from_list("perf_scale", ["#ff0000", "#ffff00", "#00b050"])
    rgba = cmap(perf / 100)
    hex_color = mcolors.to_hex(rgba)
    return f"background-color: {hex_color};"