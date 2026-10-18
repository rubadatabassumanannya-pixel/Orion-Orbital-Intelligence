import numpy as np

# ── SETUP ─────────────────────────────────────────────────────────────────────
rng = np.random.default_rng(42)
N = 200_000

# True miss distances — most are safe, ~0.23% are truly dangerous
true_miss = rng.exponential(scale=5000, size=N)  # metres
dangerous = true_miss < 1000  # true collision risk under 1km
print(f"Total encounters: {N}")
print(f"Truly dangerous: {dangerous.sum()} ({100*dangerous.mean():.2f}%)")

# TLE ages between 0.5 and 5 days
age = rng.uniform(0.5, 5.0, size=N)

# Uncertainty per axis (metres) — RTN model
sT = (1.0 + 1.5 * age) * 1000   # along-track dominant
sN = (0.3 + 0.1 * age) * 1000   # cross-track

# Noisy estimated miss (what the system sees)
noise_T = rng.normal(0, 1, N) * sT
noise_N = rng.normal(0, 1, N) * sN
est_miss = np.sqrt((true_miss + noise_T)**2 + noise_N**2)

# Pc estimate
R = 10.0  # hard-body radius metres
pc = (R**2 / (2 * sT * sN)) * np.exp(
    -0.5 * ((est_miss / sT)**2 + (noise_N / sN)**2)
)

# ── COMPARE RULES ─────────────────────────────────────────────────────────────
def evaluate(alerts, dangerous):
    tp = (alerts & dangerous).sum()
    precision = tp / alerts.sum() if alerts.sum() > 0 else 0
    recall = tp / dangerous.sum() if dangerous.sum() > 0 else 0
    return alerts.sum(), tp, round(precision*100, 1), round(recall*100, 1)

# Rule 1: raw miss distance < 2km
alert_dist = est_miss < 2000

# Rule 2: Pc threshold matched to same recall as Rule 1
_, _, _, target_recall = evaluate(alert_dist, dangerous)
# Find Pc threshold that gives same recall
thresholds = np.percentile(pc[dangerous], 100 - target_recall)
alert_pc = pc > thresholds

# Rule 3: Pc > 1e-6 (higher recall)
alert_pc2 = pc > 1e-6

# Rule 4: Standard 1e-4 threshold
alert_pc3 = pc > 1e-4

print("\n── FALSE ALARM COMPARISON ──────────────────────────────")
print(f"{'Rule':<30} {'Alerts':>8} {'True Hits':>10} {'Precision':>10} {'Recall':>8}")
print("-" * 70)

for name, alert in [
    ("Miss distance < 2km", alert_dist),
    (f"Pc > {thresholds:.1e} (matched recall)", alert_pc),
    ("Pc > 1e-6 (high recall)", alert_pc2),
    ("Pc > 1e-4 (standard)", alert_pc3),
]:
    a, tp, prec, rec = evaluate(alert, dangerous)
    print(f"{name:<30} {a:>8,} {tp:>10,} {prec:>9.1f}% {rec:>7.1f}%")

print("\n── KEY FINDING ─────────────────────────────────────────")
a_dist, tp_dist, prec_dist, rec_dist = evaluate(alert_dist, dangerous)
a_pc, tp_pc, prec_pc, rec_pc = evaluate(alert_pc, dangerous)
reduction = round(100 * (a_dist - a_pc) / a_dist, 1)
print(f"At matched recall ({rec_dist}%):")
print(f"  Miss distance alerts: {a_dist:,}")
print(f"  Pc alerts:            {a_pc:,}")
print(f"  False alarm reduction: {reduction}%")
print(f"  Precision improvement: {prec_dist}% → {prec_pc}%")