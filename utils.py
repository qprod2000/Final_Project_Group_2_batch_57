# ============================================================
# utils.py — Helper functions untuk formatting & display
# ============================================================


def format_duration(hours: float) -> str:
    """Konversi jam desimal → '2j 15m'."""
    h = int(hours)
    m = round((hours - h) * 60)
    if m == 0:
        return f"{h}j"
    return f"{h}j {m}m"


def format_inr(amount: float) -> str:
    """Format angka ke Rupee India: ₹5,953."""
    return f"₹{int(round(amount)):,}"


def stops_color(stops_raw: str) -> str:
    """Kembalikan warna hex sesuai jumlah transit."""
    return {
        "zero":        "#1D9E75",   # hijau — non-stop
        "one":         "#E8C84A",   # kuning — 1 henti
        "two_or_more": "#E24B4A",   # merah  — 2+ henti
    }.get(stops_raw, "#9593A0")


def class_color(flight_class: str) -> str:
    """Kembalikan warna hex sesuai kelas penerbangan."""
    return {
        "Economy":  "#1D9E75",
        "Business": "#7F77DD",
    }.get(flight_class, "#9593A0")