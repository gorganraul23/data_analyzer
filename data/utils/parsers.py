
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