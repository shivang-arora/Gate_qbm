/- Cyclic Fourier IQP sampling on an integer register. Lean/mathlib v4.19.0. -/
import FixedIQP
import Mathlib.Analysis.Fourier.ZMod
import Mathlib.Tactic.LinearCombination

noncomputable section
open scoped BigOperators
open FixedIQP
namespace IntegerFourier
variable {d : ℕ} [NeZero d]

/-- The first work QFT on zero and last inverse QFT each supply 1/√d.
The sign convention for mathlib's unnormalized dft is negative. -/
def rowAmp (θ : ZMod d → ℝ) (x : ZMod d) : ℂ :=
  ZMod.dft (fun z => Complex.exp ((θ z : ℂ) * Complex.I)) x / d

/-- Born marginal after the final Hadamards on m hidden qubits. -/
def hiddenBorn (m : ℕ) (A : Word m → ℂ) : ℝ :=
  ∑ y, ((transform m (fun h => (A h).re) y / size m)^2 +
        (transform m (fun h => (A h).im) y / size m)^2)

theorem hiddenBorn_mean (m : ℕ) (A : Word m → ℂ) :
    hiddenBorn m A = (∑ h, Complex.normSq (A h)) / size m := by
  have h (f : Word m → ℝ) :
      (∑ y, (transform m f y / size m)^2) = (∑ y, (f y)^2) / size m := by
    simp only [div_pow, div_eq_mul_inv, mul_pow]
    rw [← Finset.sum_mul,parseval]
    field_simp [ne_of_gt (size_pos m)]
    all_goals ring
  unfold hiddenBorn
  rw [Finset.sum_add_distrib, h, h, ← add_div, ← Finset.sum_add_distrib]
  simp only [Complex.normSq_apply, sq]

def out (m : ℕ) (θ : Word m → ZMod d → ℝ) (x : ZMod d) : ℝ :=
  hiddenBorn m (fun h => rowAmp (θ h) x)

theorem marginal (m : ℕ) (θ : Word m → ZMod d → ℝ) (x : ZMod d) :
    out m θ x = (∑ h, Complex.normSq (rowAmp (θ h) x)) / size m := hiddenBorn_mean m _

/-- A phase ramp indexed by the desired output label. -/
def ramp (a z : ZMod d) : ℝ := (ZMod.stdAddChar (a*z)).arg

theorem ramp_phase (a z : ZMod d) :
    Complex.exp ((ramp a z : ℂ) * Complex.I) = ZMod.stdAddChar (a*z) := by
  have h := Complex.norm_mul_exp_arg_mul_I (ZMod.stdAddChar (a*z))
  simpa [ramp, ZMod.stdAddChar_apply] using h

theorem ramp_decodes (a x : ZMod d) :
    rowAmp (ramp a) x = if a = x then 1 else 0 := by
  unfold rowAmp
  simp only [ramp_phase, ZMod.dft_apply, smul_eq_mul, ← AddChar.map_add_eq_mul]
  have shift (z : ZMod d) : -(z*x) + a*z = (a-x)*z := by ring
  simp_rw [shift]
  by_cases hh : a = x
  · subst x; simp [NeZero.ne (d : ℂ)]
  · have h0 := AddChar.sum_eq_zero_of_ne_one
        (ZMod.isPrimitive_stdAddChar d (sub_ne_zero.mpr hh))
    simp [hh, show (∑ z : ZMod d, ZMod.stdAddChar ((a-x)*z)) = 0 from h0]

/-- A layout with B addresses can realize any prescribed histogram of B integers. -/
theorem histogram (m : ℕ) (k : ZMod d → ℕ) (hk : ∑ x, k x = 2^m) :
    ∃ θ, ∀ x, out m θ x = k x / size m := by
  classical
  have card : Fintype.card (Word m) = Fintype.card ((x : ZMod d) × Fin (k x)) := by
    simp [card_word, hk]
  let e := Fintype.equivOfCardEq card
  refine ⟨fun h => ramp (e h).1, ?_⟩
  intro x
  rw [marginal]
  simp only [ramp_decodes]
  congr 1
  calc
    _ = ∑ h : Word m, (if (e h).1 = x then (1:ℝ) else 0) := by
      apply Finset.sum_congr rfl; intro h _; split_ifs <;> simp
    _ = ∑ b : (j : ZMod d) × Fin (k j), (if b.1 = x then (1:ℝ) else 0) :=
      Fintype.sum_equiv e _ _ (fun _ => rfl)
    _ = _ := by
      rw [Fintype.sum_sigma]
      have term (j : ZMod d) : (∑ _ : Fin (k j), (if j = x then (1:ℝ) else 0)) =
          if j = x then (k x : ℝ) else 0 := by split_ifs with hj <;> simp_all
      simp_rw [term]; simp

def TV (p q : ZMod d → ℝ) : ℝ := (∑ x, |p x - q x|) / 2

/-- Round all bin counts down and put the remaining addresses in bin zero. -/
theorem rounded (m : ℕ) (p : ZMod d → ℝ) (hp : ∀ x, 0 ≤ p x) (h1 : ∑ x, p x = 1) :
    ∃ k : ZMod d → ℕ, (∑ x, k x = 2^m) ∧ TV (fun x => k x / size m) p ≤ d / size m := by
  classical
  let B : ℕ := 2^m
  have hB : (B:ℝ) = size m := by simp [B,size]
  have pos := size_pos m
  let b := fun x => ⌊size m * p x⌋₊
  have lo (x) : (b x : ℝ) ≤ size m * p x := Nat.floor_le (mul_nonneg pos.le (hp x))
  have hi (x) : size m * p x < (b x : ℝ) + 1 := Nat.lt_floor_add_one _
  have sumP : (∑ x, size m * p x) = size m := by rw [← Finset.mul_sum,h1,mul_one]
  have sub : ∑ x, b x ≤ B := by
    have le := Finset.sum_le_sum (fun x (_ : x ∈ (Finset.univ : Finset (ZMod d))) => lo x)
    rw [sumP, ← hB] at le
    exact_mod_cast le
  let R := B - ∑ x, b x
  have eq : R + (∑ x, b x) = B := Nat.sub_add_cancel sub
  have eqR : (R:ℝ) + (∑ x, (b x : ℝ)) = size m := by rw [← hB]; exact_mod_cast eq
  let k := fun x => b x + if x = 0 then R else 0
  refine ⟨k, ?_, ?_⟩
  · simpa [k, Finset.sum_add_distrib, Nat.add_comm, B] using eq
  · let err := fun x => size m * p x - b x
    have sumE : (∑ x, err x) = R := by
      simp only [err,Finset.sum_sub_distrib,sumP]; linarith
    have Rbound : (R:ℝ) ≤ d := by
      rw [← sumE]
      calc
        _ ≤ ∑ _ : ZMod d, (1:ℝ) := Finset.sum_le_sum fun x _ => by dsimp [err]; linarith [hi x]
        _ = _ := by simp [ZMod.card]
    have bound (x) : |(k x : ℝ) - size m * p x| ≤ err x + (if x=0 then (R:ℝ) else 0) := by
      dsimp [k,err]
      by_cases hx : x=0 <;> simp only [hx, if_true, if_false, Nat.cast_add, Nat.cast_zero]
      all_goals apply abs_le.mpr; constructor <;> linarith [lo x, lo 0, Nat.cast_nonneg (α := ℝ) R]
    have total := Finset.sum_le_sum (fun x (_ : x ∈ (Finset.univ : Finset (ZMod d))) => bound x)
    rw [Finset.sum_add_distrib,sumE] at total
    simp at total
    have point (x) : |(k x:ℝ) / size m - p x| = |(k x:ℝ) - size m * p x| * (size m)⁻¹ := by
      rw [← abs_of_pos (inv_pos.mpr pos), ← abs_mul]
      congr 1; field_simp [ne_of_gt pos]
    unfold TV
    simp_rw [point]
    rw [← Finset.sum_mul, div_eq_mul_inv (d:ℝ)]
    have hmul := mul_le_mul_of_nonneg_right (show (∑ x, |(k x:ℝ)-size m*p x|) ≤ 2*d by linarith)
      (inv_nonneg.mpr pos.le)
    linarith

/-- The hidden width depends only on d and the tolerance, and is fixed before p. -/
theorem fourier_universality (d : ℕ) [NeZero d] (ε : ℝ) (he : 0 < ε) :
    ∃ m : ℕ, ∀ p : ZMod d → ℝ, (∀ x, 0 ≤ p x) → (∑ x, p x = 1) →
      ∃ θ : Word m → ZMod d → ℝ, TV (out m θ) p < ε := by
  obtain ⟨m,hm⟩ := pow_unbounded_of_one_lt ((d:ℝ) / ε) (show (1:ℝ)<2 by norm_num)
  refine ⟨m, ?_⟩
  intro p hp h1
  obtain ⟨k,hk,hTV⟩ := rounded m p hp h1
  obtain ⟨θ,ht⟩ := histogram m k hk
  refine ⟨θ, ?_⟩
  rw [funext ht]
  apply lt_of_le_of_lt hTV
  apply (div_lt_iff₀ (size_pos m)).mpr
  have hh := (div_lt_iff₀ he).mp hm
  simpa [size,mul_comm] using hh

#print axioms histogram
#print axioms fourier_universality
open scoped ComplexConjugate

/-- For a three-level work register a phase-only row supported at 0 and 1
must select one of them deterministically. -/
theorem qutrit_row (θ : ZMod 3 → ℝ)
    (hs : ∀ x, x ≠ 0 → x ≠ 1 → rowAmp θ x = 0) :
    Complex.normSq (rowAmp θ 0) = 0 ∨ Complex.normSq (rowAmp θ 0) = 1 := by
  classical
  let f := fun z => Complex.exp ((θ z : ℂ) * Complex.I)
  let A := rowAmp θ 0
  let B := rowAmp θ 1
  let c := fun z : ZMod 3 => ZMod.stdAddChar z
  have cc (z : ZMod 3) : c z * c (-z) = 1 := by
    simp [c, ← AddChar.map_add_eq_mul]
  have cj (z : ZMod 3) : conj (c z) = c (-z) := by
    simp only [c, ZMod.stdAddChar_apply, AddChar.map_neg_eq_inv, Circle.coe_inv_eq_conj]
  have inv (z : ZMod 3) : f z = A + B * c z := by
    have h := congrFun (ZMod.dft_dft f) (-z)
    rw [ZMod.dft_apply] at h
    simp only [neg_neg, smul_eq_mul, mul_neg, Nat.cast_ofNat] at h
    have point (j : ZMod 3) : ZMod.stdAddChar (j*z) * ZMod.dft f j =
        (if j=0 then 3*A else 0) + (if j=1 then 3*B*c z else 0) := by
      by_cases h0 : j=0
      · subst j; simp [A,rowAmp,f]; ring
      by_cases h1 : j=1
      · subst j; simp [B,rowAmp,f,c]; ring
      have hv := hs j h0 h1
      have hf : ZMod.dft f j = 0 := by simpa [rowAmp,f] using hv
      simp [h0,h1,hf]
    simp_rw [point] at h
    simp [Finset.sum_add_distrib] at h
    linear_combination -h / 3
  have eqn (z : ZMod 3) :
      c (-z) = (A*conj A+B*conj B)*c (-z) + A*conj B*(c (-z)*c (-z)) + B*conj A := by
    have un : f z * conj (f z) = 1 := by
      rw [Complex.mul_conj, ← Complex.sq_norm]
      simp [f, Complex.norm_exp_ofReal_mul_I]
    rw [inv, map_add, map_mul, cj] at un
    linear_combination -c (-z) * un + (B*conj A+B*conj B*c (-z)) * (cc z)
  have sumchar (t : ZMod 3) (ht : t ≠ 0) : (∑ z, c (t*z)) = 0 :=
    AddChar.sum_eq_zero_of_ne_one (ZMod.isPrimitive_stdAddChar 3 ht)
  have s1 : (∑ z : ZMod 3, c (-z)) = 0 := by simpa using sumchar (-1) (by decide)
  have s2 : (∑ z : ZMod 3, c (-z)*c (-z)) = 0 := by
    convert sumchar (-2) (by decide) using 1
    apply Finset.sum_congr rfl; intro z _
    simp only [c, ← AddChar.map_add_eq_mul]; congr 1; ring
  have combined := congrArg (fun g : ZMod 3 → ℂ => ∑ z, g z) (funext eqn)
  dsimp only at combined
  rw [s1,Finset.sum_add_distrib,Finset.sum_add_distrib,← Finset.mul_sum,← Finset.mul_sum,
      s1,s2] at combined
  have prod0 : B * conj A = 0 := by simpa [ZMod.card] using combined.symm
  rcases mul_eq_zero.mp prod0 with hB | hA
  · right
    have hh := congrArg norm (inv 0)
    have aa : ‖A‖=1 := by simpa [f,hB,Complex.norm_exp_ofReal_mul_I] using hh.symm
    change Complex.normSq A = 1
    rw [← Complex.sq_norm, aa]; norm_num
  · left
    have : A=0 := by simpa using congrArg (fun z : ℂ => conj z) hA
    change Complex.normSq A = 0
    simp [this]

/-- Zero probability outside {0,1} forces an integer number of hidden rows
to output zero. In particular 1/3 cannot occur with 2^m uniform addresses. -/
theorem qutrit_edge (m : ℕ) (θ : Word m → ZMod 3 → ℝ)
    (support : ∀ x, x ≠ 0 → x ≠ 1 → out m θ x = 0) :
    ∃ k : ℕ, out m θ 0 = k / size m := by
  classical
  let v := fun h => Complex.normSq (rowAmp (θ h) 0)
  have dich (h) : v h = 0 ∨ v h = 1 := by
    apply qutrit_row
    intro x h0 h1
    have hh := support x h0 h1
    rw [marginal, div_eq_zero_iff] at hh
    have hz := hh.resolve_right (ne_of_gt (size_pos m))
    exact Complex.normSq_eq_zero.mp ((Finset.sum_eq_zero_iff_of_nonneg
      (fun i _ => Complex.normSq_nonneg (rowAmp (θ i) x))).mp hz h (Finset.mem_univ _))
  let S := Finset.univ.filter (fun h => v h = 1)
  refine ⟨S.card, ?_⟩
  rw [marginal]
  congr 1
  change (∑ h, v h) = _
  calc
    _ = ∑ h : Word m, (if v h = 1 then (1:ℝ) else 0) := by
      apply Finset.sum_congr rfl; intro h _
      rcases dich h with hh | hh <;> simp [hh]
    _ = _ := by simp [S]

theorem qutrit_thirds_impossible (m : ℕ) (θ : Word m → ZMod 3 → ℝ)
    (support : ∀ x, x ≠ 0 → x ≠ 1 → out m θ x = 0) :
    out m θ 0 ≠ 1/3 := by
  intro h0
  obtain ⟨k,hk⟩ := qutrit_edge m θ support
  rw [hk] at h0
  have hz := size_pos m
  field_simp [ne_of_gt hz] at h0
  have hreal : (2:ℝ)^m = 3*k := by unfold size at h0; linarith
  have hnat : 2^m = 3*k := by exact_mod_cast hreal
  have hdiv : 3 ∣ 2^m := ⟨k,hnat⟩
  exact (by decide : ¬ (3:ℕ) ∣ 2) (Nat.prime_three.dvd_of_dvd_pow hdiv)

#print axioms qutrit_thirds_impossible
#print axioms qutrit_edge

/-- Native digitwise QFT. The diagonal on each hidden address factorizes over
the digits, so its conditional output amplitude is a product of cyclic amplitudes.
Both outer transforms on the register are tensor products of the digit QFTs. -/
def stringOut (d k m : ℕ) [NeZero d] (θ : Word m → Fin k → ZMod d → ℝ)
    (x : Fin k → ZMod d) : ℝ :=
  hiddenBorn m (fun h => ∏ j, rowAmp (θ h j) (x j))

theorem string_histogram (d k m : ℕ) [NeZero d] (num : (Fin k → ZMod d) → ℕ)
    (sumN : ∑ x, num x = 2^m) :
    ∃ θ, ∀ x, stringOut d k m θ x = num x / size m := by
  classical
  have card : Fintype.card (Word m) = Fintype.card ((x : Fin k → ZMod d) × Fin (num x)) := by
    simp [card_word,sumN]
  let e := Fintype.equivOfCardEq card
  refine ⟨fun h j => ramp ((e h).1 j), ?_⟩
  intro x
  rw [stringOut,hiddenBorn_mean]
  have point (h : Word m) : (∏ j, rowAmp (ramp ((e h).1 j)) (x j)) =
      if (e h).1 = x then 1 else 0 := by
    by_cases eqx : (e h).1 = x
    · simp [eqx,ramp_decodes]
    · obtain ⟨j,hj⟩ : ∃ j, (e h).1 j ≠ x j := Function.ne_iff.mp eqx
      simp only [ramp_decodes,eqx,if_false]
      exact Finset.prod_eq_zero (Finset.mem_univ j) (if_neg hj)
  simp_rw [point]
  congr 1
  calc
    _ = ∑ h : Word m, (if (e h).1 = x then (1:ℝ) else 0) := by
      apply Finset.sum_congr rfl; intro h _; split_ifs <;> simp
    _ = ∑ b : (j : Fin k → ZMod d) × Fin (num j), (if b.1 = x then (1:ℝ) else 0) :=
      Fintype.sum_equiv e _ _ (fun _ => rfl)
    _ = _ := by
      rw [Fintype.sum_sigma]
      have term (j : Fin k → ZMod d) : (∑ _ : Fin (num j), (if j=x then (1:ℝ) else 0)) =
          if j=x then (num x:ℝ) else 0 := by split_ifs with hj <;> simp_all
      simp_rw [term]; simp

theorem string_universality (d k : ℕ) [NeZero d] (ε : ℝ) (he : 0 < ε) :
    ∃ m : ℕ, ∀ p : (Fin k → ZMod d) → ℝ, (∀ x, 0 ≤ p x) → (∑ x, p x = 1) →
      ∃ θ, (∑ x, |stringOut d k m θ x - p x|)/2 < ε := by
  classical
  let D := d^k
  haveI : NeZero D := ⟨pow_ne_zero _ (NeZero.ne d)⟩
  let e : ZMod D ≃ (Fin k → ZMod d) := Fintype.equivOfCardEq (by simp [D,ZMod.card])
  obtain ⟨m,hm⟩ := pow_unbounded_of_one_lt ((D:ℝ)/ε) (show (1:ℝ)<2 by norm_num)
  refine ⟨m, ?_⟩
  intro p hp h1
  have hsum : ∑ j, p (e j) = 1 := by rw [e.sum_comp p,h1]
  obtain ⟨c,hc,ht⟩ := rounded m (fun j => p (e j)) (fun j => hp (e j)) hsum
  obtain ⟨θ,hθ⟩ := string_histogram d k m (fun x => c (e.symm x)) (by rw [e.symm.sum_comp, hc])
  refine ⟨θ, ?_⟩
  simp_rw [hθ]
  have t : (∑ x, |(c (e.symm x):ℝ) / size m - p x|) =
           ∑ j, |(c j:ℝ)/size m - p (e j)| :=
    Fintype.sum_equiv e.symm _ _ (fun _ => by simp)
  rw [t]
  apply lt_of_le_of_lt ht
  apply (div_lt_iff₀ (size_pos m)).mpr
  simpa [size,mul_comm] using (div_lt_iff₀ he).mp hm

#print axioms string_universality

end IntegerFourier
