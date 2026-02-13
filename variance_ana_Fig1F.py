import numpy as np

# Data
inter_bouton = np.array([
    -15.85585586, -17.53753754, -4.084084084, -0.48048048,
    7.207207207207219, 12.972972972972983, 15.37537537537537, 18.01801801801801
])

bouton = np.array([
    -74.23423423, -73.51351351, -12.73273273, 13.453453453453449,
    16.096096096096105, 21.86186186186187, 28.34834834834834, 83.36336336336336
])

n = len(inter_bouton)

# Observed variance ratio
observed_var_inter = np.var(inter_bouton, ddof=1)
observed_var_bouton = np.var(bouton, ddof=1)
observed_var_ratio = observed_var_bouton / observed_var_inter

print("="*70)
print("PERMUTATION TEST FOR VARIANCE RATIO")
print("="*70)
print(f"\nObserved variance (Inter-Bouton): {observed_var_inter:.3f}")
print(f"Observed variance (Bouton):       {observed_var_bouton:.3f}")
print(f"Variance ratio (Bouton/Inter):    {observed_var_ratio:.3f}")

# Permutation test for variance ratio
n_permutations = 100000  # More permutations for precise p-value
np.random.seed(42)

pooled_data = np.concatenate([inter_bouton, bouton])
permuted_var_ratios = []

for _ in range(n_permutations):
    np.random.shuffle(pooled_data)
    perm_inter = pooled_data[:n]
    perm_bouton = pooled_data[n:]
    
    var_inter = np.var(perm_inter, ddof=1)
    var_bouton = np.var(perm_bouton, ddof=1)
    
    if var_inter > 0:  # Avoid division by zero
        permuted_var_ratios.append(var_bouton / var_inter)

permuted_var_ratios = np.array(permuted_var_ratios)

# One-tailed p-value (testing if Bouton variance > Inter-Bouton variance)
p_value_variance = np.mean(permuted_var_ratios >= observed_var_ratio)

print(f"\nP-value (one-tailed, Bouton > Inter): {p_value_variance:.6f}")
print(f"P-value (scientific notation):        {p_value_variance:.2e}")

if p_value_variance < 0.001:
    print(f"\nResult: HIGHLY SIGNIFICANT (p < 0.001)")
elif p_value_variance < 0.01:
    print(f"\nResult: VERY SIGNIFICANT (p < 0.01)")
elif p_value_variance < 0.05:
    print(f"\nResult: SIGNIFICANT (p < 0.05)")
else:
    print(f"\nResult: NOT SIGNIFICANT (p ≥ 0.05)")

print("\n" + "="*70)
print("METHODS PARAGRAPH")
print("="*70)
print("""
Statistical analysis was performed using permutation tests appropriate 
for small sample sizes (n=8 paired measurements). Central tendency was 
assessed using a paired permutation test on median differences (10,000 
permutations), while variance differences between compartments were 
evaluated using a permutation test on variance ratios (100,000 
permutations). All analyses were conducted in Python using NumPy and SciPy.
""")
