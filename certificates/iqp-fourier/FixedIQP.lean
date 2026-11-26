/- Exact universality of a fixed diagonal IQP circuit.
Lean 4.19.0, mathlib v4.19.0. Run: lake env lean FixedIQP.lean -/
import Mathlib.Analysis.SpecialFunctions.Complex.Arg
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.Positivity
import Mathlib.Tactic.Ring
import Mathlib.Tactic.FieldSimp
import Mathlib.Tactic.NormNum

noncomputable section
open scoped BigOperators
namespace FixedIQP

/-- Binary words, built by adding one qubit at a time. -/
def Word : ℕ → Type
  | 0 => Unit
  | n+1 => Bool × Word n

instance wordFinite (n : ℕ) : Fintype (Word n) := by
  induction n with
  | zero => exact inferInstanceAs (Fintype Unit)
  | succ n ih => exact @instFintypeProd Bool (Word n) _ ih

instance wordDec (n : ℕ) : DecidableEq (Word n) := by
  induction n with
  | zero => exact inferInstanceAs (DecidableEq Unit)
  | succ n ih => exact @instDecidableEqProd Bool (Word n) _ ih

instance wordInhabited (n : ℕ) : Inhabited (Word n) := by
  induction n with
  | zero => exact ⟨()⟩
  | succ n ih => exact ⟨(false, ih.default)⟩

theorem card_word (n : ℕ) : Fintype.card (Word n) = 2^n := by
  induction n with
  | zero => simp [Word]
  | succ n ih => change Fintype.card (Bool × Word n) = _; simp [ih, pow_succ, Nat.mul_comm]

def size (n : ℕ) : ℝ := 2^n
theorem size_pos (n : ℕ) : 0 < size n := by unfold size; positivity

/-- Unnormalized matrix entry of the n-fold Hadamard tensor. -/
def sign : (n : ℕ) → Word n → Word n → ℝ
  | 0, _, _ => 1
  | n+1, (a,x), (b,y) => (if a && b then -1 else 1) * sign n x y

theorem sign_symm (n : ℕ) (x y : Word n) : sign n x y = sign n y x := by
  induction n with
  | zero => rfl
  | succ n ih =>
    obtain ⟨a,x⟩ := x; obtain ⟨b,y⟩ := y
    simp only [sign, Bool.and_comm a b, ih x y]

theorem sign_sq (n : ℕ) (x y : Word n) : sign n x y ^ 2 = 1 := by
  induction n with
  | zero => simp [sign]
  | succ n ih =>
    obtain ⟨a,x⟩ := x; obtain ⟨b,y⟩ := y
    cases a <;> cases b <;> simp [sign, mul_pow, ih]

theorem orthogonal (n : ℕ) (x y : Word n) :
    ∑ z, sign n x z * sign n y z = if x = y then size n else 0 := by
  classical
  induction n with
  | zero => have : x = y := Subsingleton.elim (α := Unit) x y; simp [this, Word, sign, size]
  | succ n ih =>
    obtain ⟨a,x⟩ := x; obtain ⟨b,y⟩ := y
    change (∑ z : Bool × Word n, _) = (if (a,x) = (b,y) then size (n+1) else 0)
    cases a <;> cases b <;> by_cases he : x = y <;>
      simp [Fintype.sum_prod_type, Fintype.sum_bool, sign, ih, he, size, pow_succ,
        Finset.sum_neg_distrib, mul_two, two_mul]

def transform (n : ℕ) (f : Word n → ℝ) (x : Word n) := ∑ z, sign n x z * f z

theorem transform_sign (n : ℕ) (a x : Word n) (c : ℝ) :
    transform n (fun z => sign n a z * c) x = (if x = a then size n else 0) * c := by
  simp only [transform, ← mul_assoc, ← Finset.sum_mul, orthogonal]

theorem transform_twice (n : ℕ) (f : Word n → ℝ) (x : Word n) :
    transform n (transform n f) x = size n * f x := by
  classical
  unfold transform
  simp only [Finset.mul_sum, ← mul_assoc]
  rw [Finset.sum_comm]
  simp_rw [← Finset.sum_mul]
  have h (z : Word n) :
      (∑ y, sign n x y * sign n y z) = if x = z then size n else 0 := by
    simpa only [sign_symm n _ z] using orthogonal n x z
  simp_rw [h, ite_mul, zero_mul]
  simp

theorem parseval (n : ℕ) (f : Word n → ℝ) :
    (∑ x, (transform n f x)^2) = size n * ∑ x, (f x)^2 := by
  calc
    _ = ∑ x, ∑ y, (sign n x y * f y) * transform n f x := by
      simp only [← Finset.sum_mul, pow_two]; rfl
    _ = ∑ y, f y * transform n (transform n f) y := by
      rw [Finset.sum_comm]
      apply Finset.sum_congr rfl
      intro y _
      unfold transform
      rw [Finset.mul_sum]
      apply Finset.sum_congr rfl
      intro x _
      rw [sign_symm n x y]; ring
    _ = _ := by
      simp only [transform_twice, Finset.mul_sum]
      apply Finset.sum_congr rfl; intro y _; ring

structure Row (ι : Type) where
  a : ι
  b : ι
  r : ℝ
  lo : 0 ≤ r
  hi : r ≤ 1

def Row.mass {ι : Type} [DecidableEq ι] (u : Row ι) (x : ι) : ℝ :=
  (if x = u.a then u.r else 0) + (if x = u.b then 1-u.r else 0)

/-- Allocate one unit-capacity bin per label; store the extracted row at a. -/
theorem allocate {ι : Type} [DecidableEq ι] [Inhabited ι] (s : Finset ι) (w : ι → ℝ)
    (hn : ∀ x ∈ s, 0 ≤ w x) (hs : ∑ x ∈ s, w x = s.card) :
    ∃ rows : ι → Row ι, ∀ x, (∑ h ∈ s, (rows h).mass x) = if x ∈ s then w x else 0 := by
  classical
  induction s using Finset.strongInductionOn generalizing w
  rename_i s ih
  by_cases empty : s = ∅
  · subst s; exact ⟨fun _ => ⟨default, default, 1, by norm_num, by norm_num⟩, by simp⟩
  have ne := Finset.nonempty_iff_ne_empty.mpr empty
  have small : ∃ a ∈ s, w a ≤ 1 := by
    by_contra! con
    have hh := Finset.sum_lt_sum_of_nonempty ne con
    simp at hh; linarith
  have large : ∃ b ∈ s, 1 ≤ w b := by
    by_contra! con
    have hh := Finset.sum_lt_sum_of_nonempty ne con
    simp at hh; linarith
  obtain ⟨a,ha,low⟩ := small
  obtain ⟨b,hb,high⟩ := large
  let w' := fun x => w x - (if x = b then 1 - w a else 0)
  have nn : ∀ x ∈ s.erase a, 0 ≤ w' x := by
    intro x hx
    obtain ⟨_,hx⟩ := Finset.mem_erase.mp hx
    dsimp [w']; split_ifs with e
    · subst x; linarith [hn a ha]
    · simpa using hn x hx
  have tot : ∑ x ∈ s.erase a, w' x = ((s.erase a).card : ℝ) := by
    have hw := Finset.sum_erase_add _ w ha
    have hc := Finset.card_erase_add_one ha
    have hc' : ((s.erase a).card : ℝ) + 1 = s.card := by exact_mod_cast hc
    simp only [w', Finset.sum_sub_distrib, Finset.sum_ite_eq', Finset.mem_erase]
    split_ifs <;> simp_all
    all_goals linarith
  obtain ⟨rs,hrs⟩ := ih (s.erase a) (Finset.erase_ssubset ha) w' nn tot
  let u : Row ι := ⟨a,b,w a,hn a ha,low⟩
  refine ⟨Function.update rs a u, ?_⟩
  intro x
  rw [← Finset.sum_erase_add _ _ ha]
  have hh : (∑ h ∈ s.erase a, (Function.update rs a u h).mass x) =
      ∑ h ∈ s.erase a, (rs h).mass x := by
    apply Finset.sum_congr rfl
    intro h mem
    rw [Function.update_of_ne (Finset.mem_erase.mp mem).1]
  rw [hh, hrs, Function.update_self]
  simp only [Finset.mem_erase, u, Row.mass, w']
  split_ifs <;> simp_all
  all_goals linarith

def rowPhase {n : ℕ} (u : Row (Word n)) (z : Word n) : ℂ :=
  ⟨sign n u.a z * Real.sqrt u.r, sign n u.b z * Real.sqrt (1-u.r)⟩

theorem row_norm {n : ℕ} (u : Row (Word n)) (z : Word n) : ‖rowPhase u z‖ = 1 := by
  have sq : ‖rowPhase u z‖ ^ 2 = 1 := by
    rw [Complex.sq_norm, Complex.normSq_apply]
    change (_ * _) * (_ * _) + (_ * _) * (_ * _) = (1:ℝ)
    simp only [← sq, mul_pow, sign_sq, one_mul, Real.sq_sqrt u.lo,
      Real.sq_sqrt (sub_nonneg.mpr u.hi)]
    ring
  nlinarith [norm_nonneg (rowPhase u z)]

def rowAngle {n : ℕ} (u : Row (Word n)) (z : Word n) := (rowPhase u z).arg

theorem row_cos {n : ℕ} (u : Row (Word n)) (z : Word n) :
    Real.cos (rowAngle u z) = sign n u.a z * Real.sqrt u.r := by
  simpa only [rowAngle, row_norm, one_mul] using Complex.norm_mul_cos_arg (rowPhase u z)

theorem row_sin {n : ℕ} (u : Row (Word n)) (z : Word n) :
    Real.sin (rowAngle u z) = sign n u.b z * Real.sqrt (1-u.r) := by
  simpa only [rowAngle, row_norm, one_mul] using Complex.norm_mul_sin_arg (rowPhase u z)

/-- A layout is the set of addressed, diagonal phase gates between the Hadamards. -/
abbrev Layout (n : ℕ) := Finset (Word n × Word n)
abbrev Angles (n : ℕ) := Word n → Word n → ℝ

def fullLayout (n : ℕ) : Layout n := Finset.univ
def gated {n : ℕ} (C : Layout n) (θ : Angles n) (h z : Word n) : ℝ :=
  if (h,z) ∈ C then θ h z else 0

def amplitude {n : ℕ} (f : Word n → Word n → ℝ) (y x : Word n) : ℝ :=
  transform n (fun h => transform n (f h) x) y / (size n)^2

def zeroWord : (n : ℕ) → Word n
  | 0 => ()
  | n+1 => (false, zeroWord n)

theorem sign_zero (n : ℕ) (x : Word n) : sign n x (zeroWord n) = 1 := by
  induction n with
  | zero => rfl
  | succ n ih => obtain ⟨b,y⟩ := x; simp [zeroWord, sign, ih]

/-- The normalized final Hadamard tensor on all 2n qubits. -/
def allHadamards {n : ℕ} (f : Angles n) (y x : Word n) : ℝ :=
  transform n (fun h => transform n (f h) x) y / size n

/-- Each addressed gate contributes its angle only on its selected basis word. -/
def phaseSum {n : ℕ} (C : Layout n) (θ : Angles n) (h z : Word n) : ℝ :=
  ∑ k ∈ C, if (h,z) = k then θ k.1 k.2 else 0

theorem phaseSum_eq {n : ℕ} (C : Layout n) (θ : Angles n) (h z : Word n) :
    phaseSum C θ h z = gated C θ h z := by simp [phaseSum, gated]

/-- First all-wire Hadamard column on |0,0>, addressed phases, final all-wire
Hadamards. Evaluate with cos and sin for the two Cartesian coordinates. -/
def run {n : ℕ} (C : Layout n) (θ : Angles n) (part : ℝ → ℝ) : Angles n :=
  allHadamards (fun h z =>
    ((sign n h (zeroWord n) * sign n z (zeroWord n)) / size n) *
    part (phaseSum C θ h z))

theorem transform_scale (n : ℕ) (f : Word n → ℝ) (r : ℝ) (x : Word n) :
    transform n (fun z => r * f z) x = r * transform n f x := by
  unfold transform
  rw [Finset.mul_sum]
  apply Finset.sum_congr rfl; intro y _; ring

theorem run_eq {n : ℕ} (C : Layout n) (θ : Angles n) (part : ℝ → ℝ) (y x : Word n) :
    run C θ part y x = amplitude (fun h z => part (gated C θ h z)) y x := by
  simp only [run, allHadamards, amplitude, sign_zero, one_mul, phaseSum_eq, transform_scale]
  ring

/-- Born probability on the work register after tracing out the hidden register. -/
def workProb {n : ℕ} (C : Layout n) (θ : Angles n) (x : Word n) : ℝ :=
  ∑ y, ((run C θ Real.cos y x)^2 + (run C θ Real.sin y x)^2)

theorem coord_marginal {n : ℕ} (f : Word n → Word n → ℝ) (x : Word n) :
    (∑ y, (amplitude f y x)^2) =
      (∑ h, (transform n (f h) x)^2) / (size n)^3 := by
  simp only [amplitude, div_pow, div_eq_mul_inv, mul_pow]
  rw [← Finset.sum_mul, parseval]
  field_simp [ne_of_gt (size_pos n)]
  all_goals ring

theorem one_row {n : ℕ} (u : Row (Word n)) (x : Word n) :
    (transform n (fun z => Real.cos (rowAngle u z)) x)^2 +
    (transform n (fun z => Real.sin (rowAngle u z)) x)^2 =
    (size n)^2 * u.mass x := by
  simp only [row_cos, row_sin, transform_sign, mul_pow, Row.mass,
    Real.sq_sqrt u.lo, Real.sq_sqrt (sub_nonneg.mpr u.hi)]
  split_ifs <;> ring

theorem row_mixture {n : ℕ} (rows : Word n → Row (Word n)) (x : Word n) :
    workProb (fullLayout n) (fun h z => rowAngle (rows h) z) x =
    (∑ h, (rows h).mass x) / size n := by
  unfold workProb
  simp only [run_eq, gated, fullLayout, Finset.mem_univ, if_true]
  rw [Finset.sum_add_distrib, coord_marginal, coord_marginal, ← add_div, ← Finset.sum_add_distrib]
  simp only [one_row, ← Finset.mul_sum]
  field_simp [ne_of_gt (size_pos n)]
  all_goals ring

/-- The fully specified fixed circuit represents every target exactly. -/
theorem full_exact (n : ℕ) (p : Word n → ℝ) (hp : ∀ x, 0 ≤ p x)
    (h1 : ∑ x, p x = 1) :
    ∃ θ : Angles n, ∀ x, workProb (fullLayout n) θ x = p x := by
  have tot : (∑ x, size n * p x) = ((Finset.univ : Finset (Word n)).card : ℝ) := by
    rw [← Finset.mul_sum, h1, mul_one, Finset.card_univ, card_word]
    simp [size]
  obtain ⟨rows, hr⟩ := allocate (Finset.univ : Finset (Word n)) (fun x => size n * p x)
    (fun x _ => mul_nonneg (size_pos n).le (hp x)) tot
  refine ⟨fun h z => rowAngle (rows h) z, fun x => ?_⟩
  rw [row_mixture, hr]; simp [ne_of_gt (size_pos n)]

/-- Choose the gate architecture before the target, with exact equality. -/
theorem fixed_universality (n : ℕ) :
    ∃ C : Layout n, C = fullLayout n ∧
      ∀ p : Word n → ℝ, (∀ x, 0 ≤ p x) → (∑ x, p x = 1) →
      ∃ θ : Angles n, ∀ x, workProb C θ x = p x :=
  ⟨fullLayout n, rfl, full_exact n⟩

#print axioms full_exact
#print axioms fixed_universality
end FixedIQP
