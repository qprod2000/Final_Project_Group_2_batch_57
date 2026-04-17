def normalize_stops(x):
    x = str(x).lower().strip()
    if x in ["0 stops", "zero", "non-stop"]:
        return "Langsung"
    elif x in ["1 stop", "one"]:
        return "1 Transit"
    else:
        return "2 Transit"


def format_duration(x):
    h = int(x)
    m = int((x - h) * 60)
    return f"{h} jam {m} menit"


def format_inr(x):
    return f"₹ {int(x):,}"