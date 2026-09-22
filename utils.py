def grade_from_score(score):
    if score >= 40: return 'A'
    elif score >= 35: return 'B'
    elif score >= 25: return 'C'
    elif score >= 20: return 'D'
    else: return 'F'


def normalize(value, max_value):
    return value / max_value


# -----------------------------------
# STAGE 1 LCI (CAT1 → QUIZ1)
# FIXED: removed positive bias trend
# -----------------------------------
def calculate_lci_stage1(cat1, quiz1):
    cat1_n = cat1 / 50
    quiz1_n = quiz1 / 20

    # FIXED TREND: balanced (no upward bias)
    trend_n = (cat1_n - quiz1_n)

    return (
        0.55 * cat1_n +
        0.35 * quiz1_n +
        0.10 * (1 - abs(trend_n))   # consistency reward
    )


# -----------------------------------
# STAGE 2 LCI (CAT2 → FAT)
# FIXED: reduced attendance dominance + unbiased trend
# -----------------------------------
def calculate_lci_stage2(cat2_score, quiz2, quiz3, attendance, cat2_uncertainty=None):

    cat2_n = cat2_score / 50
    q2_n = quiz2 / 20
    q3_n = quiz3 / 20
    att_n = attendance / 100

    quiz_avg = (q2_n + q3_n) / 2

    # FIXED TREND (no artificial positive shift)
    trend_n = (cat2_n - quiz_avg)

    # consistency factor (penalizes large mismatch)
    trend_factor = 1 - abs(trend_n)

    # uncertainty control (optional safety)
    uncertainty_factor = 1.0
    if cat2_uncertainty is not None:
        uncertainty_factor = max(0.7, 1 - cat2_uncertainty * 0.2)

    return (
        uncertainty_factor * (
            0.45 * cat2_n +
            0.25 * quiz_avg +
            0.20 * att_n +
            0.10 * trend_factor
        )
    )