/-
Selection and reconstruction guarantees for the SVD-OMP family
(plain, whitened and causal-metric SVD-OMP). Lab notebook entry T1.

P1  `sparse_error_eq`, `topk_optimal`: in any real inner product space, with an
    orthonormal dictionary, the error of a k-sparse approximation is the
    coefficient error on the support plus the dropped energy, so keeping the
    k largest coefficients (with their own values) is optimal.
P2  `weighted_svd_reconstruction`: the weighted factorisation maps back to W exactly.
P3  `write_metric_orthonormal`, `metric_topk_optimal`: the write vectors are
    orthonormal in the metric M = L_out L_outᵀ, so per-token top-k is exactly
    optimal for the M-weighted output error.
P4  `weighted_error_identity`: the sample-averaged M-weighted error equals the
    Frobenius norm of L_outᵀ E L_in.
P5  `fisher_identity`, `fisher_pullback`: the second moment of e_y - p under
    y ~ p is diag p - p pᵀ, and pulling back through a linear map gives the
    Gauss-Newton metric.
-/
import Mathlib

open Finset Matrix
open scoped InnerProductSpace RealInnerProductSpace

namespace SvdOmp

/-! ## Top-k exchange lemma -/

/-- If every element of `T` beats every element outside `T` under `f`, then `T` has
the largest `f`-sum among sets of its size. -/
theorem sum_le_sum_of_top {ι : Type*} [DecidableEq ι] (f : ι → ℝ) (S T : Finset ι)
    (hcard : S.card = T.card) (htop : ∀ i ∈ T, ∀ j ∉ T, f j ≤ f i) :
    ∑ c ∈ S, f c ≤ ∑ c ∈ T, f c := by
  have hS := Finset.sum_inter_add_sum_sdiff S T f
  have hT := Finset.sum_inter_add_sum_sdiff T S f
  rw [Finset.inter_comm] at hT
  have hc : (S \ T).card = (T \ S).card := Finset.card_sdiff_comm hcard
  suffices h : ∑ c ∈ S \ T, f c ≤ ∑ c ∈ T \ S, f c by linarith
  rcases (T \ S).eq_empty_or_nonempty with he | hne
  · have : S \ T = ∅ := by
      rw [← Finset.card_eq_zero, hc, he, Finset.card_empty]
    simp [this, he]
  · obtain ⟨i0, hi0, hmin⟩ := (T \ S).exists_min_image f hne
    have hi0T : i0 ∈ T := (Finset.mem_sdiff.mp hi0).1
    calc ∑ c ∈ S \ T, f c ≤ (S \ T).card • f i0 := by
          apply Finset.sum_le_card_nsmul
          intro j hj
          exact htop i0 hi0T j (Finset.mem_sdiff.mp hj).2
      _ = (T \ S).card • f i0 := by rw [hc]
      _ ≤ ∑ c ∈ T \ S, f c := Finset.card_nsmul_le_sum _ _ _ hmin

/-- `T` holds `k` coefficients of largest magnitude (ties allowed). -/
def IsTopK {ι : Type*} (a : ι → ℝ) (T : Finset ι) (k : ℕ) : Prop :=
  T.card = k ∧ ∀ i ∈ T, ∀ j ∉ T, a j ^ 2 ≤ a i ^ 2

/-- The energy left outside a top-k set is the smallest possible among k-sets. -/
theorem dropped_energy_le {ι : Type*} [Fintype ι] [DecidableEq ι] (a : ι → ℝ)
    (S T : Finset ι) (k : ℕ) (hS : S.card = k) (hT : IsTopK a T k) :
    ∑ c ∈ Tᶜ, a c ^ 2 ≤ ∑ c ∈ Sᶜ, a c ^ 2 := by
  have h := sum_le_sum_of_top (fun c => a c ^ 2) S T (by rw [hS, hT.1]) hT.2
  have e1 := Finset.sum_add_sum_compl S (fun c => a c ^ 2)
  have e2 := Finset.sum_add_sum_compl T (fun c => a c ^ 2)
  linarith

/-- Coordinate form of P1: zero the coordinates outside `S`, replace those inside by `x`. -/
theorem coord_error_eq {ι : Type*} [Fintype ι] [DecidableEq ι] (a x : ι → ℝ)
    (S : Finset ι) :
    ∑ c, (a c - if c ∈ S then x c else 0) ^ 2
      = ∑ c ∈ S, (a c - x c) ^ 2 + ∑ c ∈ Sᶜ, a c ^ 2 := by
  rw [← Finset.sum_add_sum_compl S]
  congr 1
  · exact Finset.sum_congr rfl fun c hc => by simp [hc]
  · exact Finset.sum_congr rfl fun c hc => by simp [Finset.mem_compl.mp hc]

/-! ## P1: orthonormal dictionaries in any real inner product space -/

section InnerProduct

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E]
variable {ι : Type*} [Fintype ι] [DecidableEq ι]

/-- P1, error identity. `y = ∑ a_c w_c`; keep support `S` with coefficients `x`. -/
theorem sparse_error_eq (w : ι → E) (hw : Orthonormal ℝ w) (a x : ι → ℝ) (S : Finset ι) :
    ‖∑ c, a c • w c - ∑ c ∈ S, x c • w c‖ ^ 2
      = ∑ c ∈ S, (a c - x c) ^ 2 + ∑ c ∈ Sᶜ, a c ^ 2 := by
  set b : ι → ℝ := fun c => a c - if c ∈ S then x c else 0
  have hsplit : ∑ c, a c • w c - ∑ c ∈ S, x c • w c = ∑ c, b c • w c := by
    have : ∑ c ∈ S, x c • w c = ∑ c, (if c ∈ S then x c else 0) • w c := by
      simp_rw [ite_smul, zero_smul]
      rw [Finset.sum_ite_mem, Finset.univ_inter]
    rw [this, ← Finset.sum_sub_distrib]
    exact Finset.sum_congr rfl fun c _ => by simp [b, sub_smul]
  rw [hsplit, ← real_inner_self_eq_norm_sq, hw.inner_sum b b Finset.univ]
  simp only [conj_trivial]
  rw [← coord_error_eq a x S]
  exact Finset.sum_congr rfl fun c _ => by ring

/-- P1, optimality: a top-k set with its own coefficients is the best k-sparse
approximation in the dictionary, over every support of size k and every coefficient choice. -/
theorem topk_optimal (w : ι → E) (hw : Orthonormal ℝ w) (a : ι → ℝ) (k : ℕ)
    (T : Finset ι) (hT : IsTopK a T k) (S : Finset ι) (hS : S.card = k) (x : ι → ℝ) :
    ‖∑ c, a c • w c - ∑ c ∈ T, a c • w c‖ ^ 2
      ≤ ‖∑ c, a c • w c - ∑ c ∈ S, x c • w c‖ ^ 2 := by
  rw [sparse_error_eq w hw a a T, sparse_error_eq w hw a x S]
  have h0 : ∑ c ∈ T, (a c - a c) ^ 2 = 0 := by simp
  have h1 : 0 ≤ ∑ c ∈ S, (a c - x c) ^ 2 := Finset.sum_nonneg fun _ _ => sq_nonneg _
  have h2 := dropped_energy_le a S T k hS hT
  linarith

end InnerProduct

/-! ## P2, P3: the weighted factorisation (whitened and causal-metric SVD-OMP) -/

section Weighted

variable {m n r : Type*} [Fintype m] [Fintype n] [Fintype r]
  [DecidableEq m] [DecidableEq n] [DecidableEq r]

/-- P2. If `L_outᵀ W L_in = P diag(s) Qᵀ` with invertible weights, then
`write diag(s) readᵀ = W` for `write = L_outᵀ⁻¹ P`, `read = L_inᵀ⁻¹ Q`.
No orthogonality of `P` or `Q` is needed. -/
theorem weighted_svd_reconstruction (W : Matrix m n ℝ) (Lin : Matrix n n ℝ)
    (Lout : Matrix m m ℝ) (P : Matrix m r ℝ) (s : r → ℝ) (Q : Matrix n r ℝ)
    (hin : IsUnit Lin.det) (hout : IsUnit Lout.det)
    (hsvd : Loutᵀ * W * Lin = P * diagonal s * Qᵀ) :
    ((Loutᵀ)⁻¹ * P) * diagonal s * ((Linᵀ)⁻¹ * Q)ᵀ = W := by
  have houtT : IsUnit Loutᵀ.det := by rwa [det_transpose]
  have hQ : ((Linᵀ)⁻¹ * Q)ᵀ = Qᵀ * Lin⁻¹ := by
    rw [transpose_mul, ← transpose_nonsing_inv, transpose_transpose]
  calc ((Loutᵀ)⁻¹ * P) * diagonal s * ((Linᵀ)⁻¹ * Q)ᵀ
        = (Loutᵀ)⁻¹ * (P * diagonal s * Qᵀ) * Lin⁻¹ := by
          rw [hQ]; simp only [Matrix.mul_assoc]
    _ = (Loutᵀ)⁻¹ * (Loutᵀ * W * Lin) * Lin⁻¹ := by rw [hsvd]
    _ = W := by
          simp only [Matrix.mul_assoc]
          rw [mul_nonsing_inv _ hin, Matrix.mul_one, ← Matrix.mul_assoc,
            nonsing_inv_mul _ houtT, Matrix.one_mul]

/-- P3. The write vectors are orthonormal in the metric `M = L_out L_outᵀ`. -/
theorem write_metric_orthonormal (Lout : Matrix m m ℝ) (hout : IsUnit Lout.det)
    (P : Matrix m r ℝ) (hP : Pᵀ * P = 1) :
    ((Loutᵀ)⁻¹ * P)ᵀ * (Lout * Loutᵀ) * ((Loutᵀ)⁻¹ * P) = 1 := by
  have houtT : IsUnit Loutᵀ.det := by rwa [det_transpose]
  have hT : ((Loutᵀ)⁻¹ * P)ᵀ = Pᵀ * Lout⁻¹ := by
    rw [transpose_mul, ← transpose_nonsing_inv, transpose_transpose]
  rw [hT]
  calc Pᵀ * Lout⁻¹ * (Lout * Loutᵀ) * ((Loutᵀ)⁻¹ * P)
        = Pᵀ * (Lout⁻¹ * Lout) * (Loutᵀ * (Loutᵀ)⁻¹) * P := by
          simp only [Matrix.mul_assoc]
    _ = Pᵀ * P := by
          rw [nonsing_inv_mul _ hout, mul_nonsing_inv _ houtT, Matrix.mul_one,
            Matrix.mul_one]
    _ = 1 := hP

/-- Moving a matrix across the dot product. -/
theorem mulVec_dotProduct {p q : Type*} [Fintype p] [Fintype q] (A : Matrix p q ℝ)
    (x : q → ℝ) (y : p → ℝ) : (A *ᵥ x) ⬝ᵥ y = x ⬝ᵥ (Aᵀ *ᵥ y) := by
  rw [dotProduct_mulVec, vecMul_transpose]

/-- The M-weighted squared norm of `V b` equals `‖b‖²` when `Vᵀ M V = 1`. -/
theorem metric_norm_eq (M : Matrix m m ℝ) (V : Matrix m r ℝ) (hV : Vᵀ * M * V = 1)
    (b : r → ℝ) : (V *ᵥ b) ⬝ᵥ (M *ᵥ (V *ᵥ b)) = b ⬝ᵥ b := by
  rw [mulVec_dotProduct, mulVec_mulVec, mulVec_mulVec, hV, one_mulVec]

/-- P1 + P2 + P3, the per-token statement for whitened and causal-metric SVD-OMP.
With `a = diag(s) readᵀ φ`, keeping support `S` with coefficients `x` leaves
M-weighted output error `∑_S (a - x)² + ∑_{Sᶜ} a²`, and a top-k set with its own
coefficients is optimal among all k-sparse choices. Taking `L_out = 1` gives
whitened SVD-OMP; also `L_in = 1` gives plain SVD-OMP. -/
theorem metric_topk_optimal (W : Matrix m n ℝ) (Lin : Matrix n n ℝ) (Lout : Matrix m m ℝ)
    (P : Matrix m r ℝ) (s : r → ℝ) (Q : Matrix n r ℝ)
    (hin : IsUnit Lin.det) (hout : IsUnit Lout.det) (hP : Pᵀ * P = 1)
    (hsvd : Loutᵀ * W * Lin = P * diagonal s * Qᵀ) (φ : n → ℝ) (k : ℕ)
    (T : Finset r) (hT : IsTopK ((diagonal s * ((Linᵀ)⁻¹ * Q)ᵀ) *ᵥ φ) T k)
    (S : Finset r) (hS : S.card = k) (x : r → ℝ) :
    let V := (Loutᵀ)⁻¹ * P
    let M := Lout * Loutᵀ
    let a := (diagonal s * ((Linᵀ)⁻¹ * Q)ᵀ) *ᵥ φ
    let err := fun (S : Finset r) (x : r → ℝ) =>
      let e := W *ᵥ φ - V *ᵥ (fun c => if c ∈ S then x c else 0)
      e ⬝ᵥ (M *ᵥ e)
    err S x = ∑ c ∈ S, (a c - x c) ^ 2 + ∑ c ∈ Sᶜ, a c ^ 2 ∧ err T a ≤ err S x := by
  intro V M a err
  have hV : Vᵀ * M * V = 1 := write_metric_orthonormal Lout hout P hP
  have hW : W *ᵥ φ = V *ᵥ a := by
    simp only [V, a, mulVec_mulVec]
    rw [← Matrix.mul_assoc, weighted_svd_reconstruction W Lin Lout P s Q hin hout hsvd]
  have key : ∀ (S : Finset r) (x : r → ℝ),
      err S x = ∑ c ∈ S, (a c - x c) ^ 2 + ∑ c ∈ Sᶜ, a c ^ 2 := by
    intro S x
    simp only [err]
    rw [hW, ← mulVec_sub, metric_norm_eq M V hV, ← coord_error_eq a x S]
    simp [dotProduct, Pi.sub_apply, sq]
  refine ⟨key S x, ?_⟩
  rw [key T a, key S x]
  have h0 : ∑ c ∈ T, (a c - a c) ^ 2 = 0 := by simp
  have h1 : 0 ≤ ∑ c ∈ S, (a c - x c) ^ 2 := Finset.sum_nonneg fun _ _ => sq_nonneg _
  have h2 := dropped_energy_le a S T k hS hT
  linarith

/-- Paper Proposition "complement cannot conspire": keep `S` at full strength and scale each
unselected atom by a mask `m_c ∈ [0, 1]`. The deviation is `-V b` with `b_c = (1 - m_c) a_c` off
`S` and `0` on `S`; its M-weighted size is `∑_{Sᶜ} (1 - m_c)² a_c²`, never more than removing the
whole complement. So no partial ablation of the complement does more damage, in this metric,
than removing it outright. -/
theorem complement_mask_bound (M : Matrix m m ℝ) (V : Matrix m r ℝ) (hV : Vᵀ * M * V = 1)
    (a : r → ℝ) (S : Finset r) (mask : r → ℝ) (hm : ∀ c, 0 ≤ mask c ∧ mask c ≤ 1) :
    let δ := V *ᵥ (fun c => if c ∈ S then 0 else (1 - mask c) * a c)
    δ ⬝ᵥ (M *ᵥ δ) = ∑ c ∈ Sᶜ, (1 - mask c) ^ 2 * a c ^ 2 ∧
      δ ⬝ᵥ (M *ᵥ δ) ≤ ∑ c ∈ Sᶜ, a c ^ 2 := by
  intro δ
  have hδ : δ ⬝ᵥ (M *ᵥ δ) = ∑ c ∈ Sᶜ, (1 - mask c) ^ 2 * a c ^ 2 := by
    simp only [δ]
    rw [metric_norm_eq M V hV, dotProduct, ← Finset.sum_add_sum_compl S]
    have h0 : ∑ c ∈ S, (if c ∈ S then 0 else (1 - mask c) * a c)
        * (if c ∈ S then 0 else (1 - mask c) * a c) = 0 :=
      Finset.sum_eq_zero fun c hc => by simp [hc]
    rw [h0, zero_add]
    exact Finset.sum_congr rfl fun c hc => by
      simp [Finset.mem_compl.mp hc]; ring
  refine ⟨hδ, ?_⟩
  rw [hδ]
  refine Finset.sum_le_sum fun c _ => ?_
  have h1 : (1 - mask c) ^ 2 ≤ 1 := by nlinarith [hm c]
  nlinarith [sq_nonneg (a c)]

/-! ## P4: sample-averaged weighted error -/

theorem trace_mul_vecMulVec (B : Matrix n n ℝ) (u : n → ℝ) :
    trace (B * vecMulVec u u) = u ⬝ᵥ (B *ᵥ u) := by
  simp only [Matrix.trace, Matrix.diag, Matrix.mul_apply, Matrix.vecMulVec_apply, dotProduct,
    Matrix.mulVec, Finset.mul_sum]
  refine Finset.sum_congr rfl fun i _ => Finset.sum_congr rfl fun j _ => by ring

/-- P4. With sample Gram `G = (1/N) ∑ φᵢ φᵢᵀ = L_in L_inᵀ` and `M = L_out L_outᵀ`,
the average M-weighted error of `E` equals `‖L_outᵀ E L_in‖_F²`. -/
theorem weighted_error_identity {N : ℕ} (φ : Fin N → n → ℝ) (E : Matrix m n ℝ)
    (Lin : Matrix n n ℝ) (Lout : Matrix m m ℝ)
    (hG : (1 / (N : ℝ)) • ∑ i, vecMulVec (φ i) (φ i) = Lin * Linᵀ) :
    (1 / (N : ℝ)) * ∑ i, (E *ᵥ φ i) ⬝ᵥ ((Lout * Loutᵀ) *ᵥ (E *ᵥ φ i))
      = ∑ i, ∑ j, (Loutᵀ * E * Lin) i j ^ 2 := by
  have hterm : ∀ i, (E *ᵥ φ i) ⬝ᵥ ((Lout * Loutᵀ) *ᵥ (E *ᵥ φ i))
      = trace ((Eᵀ * (Lout * Loutᵀ) * E) * vecMulVec (φ i) (φ i)) := by
    intro i
    rw [trace_mul_vecMulVec, mulVec_dotProduct, mulVec_mulVec, mulVec_mulVec]
  have hfrob : ∀ A : Matrix m n ℝ, ∑ i, ∑ j, A i j ^ 2 = trace (Aᵀ * A) := by
    intro A
    simp only [Matrix.trace, Matrix.diag, Matrix.mul_apply, Matrix.transpose_apply, sq]
    exact Finset.sum_comm
  calc (1 / (N : ℝ)) * ∑ i, (E *ᵥ φ i) ⬝ᵥ ((Lout * Loutᵀ) *ᵥ (E *ᵥ φ i))
        = (1 / (N : ℝ)) * ∑ i, trace ((Eᵀ * (Lout * Loutᵀ) * E) * vecMulVec (φ i) (φ i)) := by
          simp_rw [hterm]
    _ = trace ((Eᵀ * (Lout * Loutᵀ) * E) * ((1 / (N : ℝ)) • ∑ i, vecMulVec (φ i) (φ i))) := by
          rw [Matrix.mul_smul, trace_smul, Matrix.mul_sum, trace_sum, smul_eq_mul]
    _ = trace ((Eᵀ * (Lout * Loutᵀ) * E) * (Lin * Linᵀ)) := by rw [hG]
    _ = trace (Linᵀ * ((Eᵀ * (Lout * Loutᵀ) * E) * Lin)) := by
          rw [← Matrix.mul_assoc, trace_mul_comm]
    _ = trace ((Loutᵀ * E * Lin)ᵀ * (Loutᵀ * E * Lin)) := by
          simp only [Matrix.transpose_mul, Matrix.transpose_transpose, Matrix.mul_assoc]
    _ = ∑ i, ∑ j, (Loutᵀ * E * Lin) i j ^ 2 := (hfrob _).symm

end Weighted

/-! ## P5: the sampled-label gradient metric is the Fisher / Gauss-Newton metric -/

section Fisher

variable {κ : Type*} [Fintype κ] [DecidableEq κ]

/-- P5. For `y ~ p` (`∑ p = 1`), `E[(e_y - p)(e_y - p)ᵀ] = diag p - p pᵀ`.
`e_y - p` is the gradient of `log softmax(z)_y` with respect to the logits `z`. -/
theorem fisher_identity (p : κ → ℝ) (hp : ∑ y, p y = 1) :
    ∑ y, p y • vecMulVec (Pi.single y 1 - p) (Pi.single y 1 - p)
      = diagonal p - vecMulVec p p := by
  ext i j
  simp only [Matrix.sum_apply, Matrix.smul_apply, vecMulVec_apply, Pi.sub_apply, smul_eq_mul,
    Matrix.sub_apply, diagonal_apply]
  have expand : ∀ y, p y * (((Pi.single y (1 : ℝ) : κ → ℝ) i - p i)
        * ((Pi.single y (1 : ℝ) : κ → ℝ) j - p j))
      = p y * ((Pi.single y (1 : ℝ) : κ → ℝ) i * (Pi.single y (1 : ℝ) : κ → ℝ) j)
        - p j * (p y * (Pi.single y (1 : ℝ) : κ → ℝ) i)
        - p i * (p y * (Pi.single y (1 : ℝ) : κ → ℝ) j)
        + p i * p j * p y := by
    intro y; ring
  simp_rw [expand, Finset.sum_add_distrib, Finset.sum_sub_distrib, ← Finset.mul_sum, hp]
  have hi : ∑ y, p y * (Pi.single y (1 : ℝ) : κ → ℝ) i = p i := by
    simp [Pi.single_apply]
  have hj : ∑ y, p y * (Pi.single y (1 : ℝ) : κ → ℝ) j = p j := by
    simp [Pi.single_apply]
  have hij : ∑ y, p y * ((Pi.single y (1 : ℝ) : κ → ℝ) i * (Pi.single y (1 : ℝ) : κ → ℝ) j)
      = if i = j then p i else 0 := by
    by_cases h : i = j
    · subst h; simp [Pi.single_apply]
    · simp [Pi.single_apply, h]
  rw [hi, hj, hij]
  ring

/-- P5, pulled back to a layer output: with `g_y = Jᵀ (e_y - p)`,
`E[g gᵀ] = Jᵀ (diag p - p pᵀ) J`. -/
theorem fisher_pullback {d : Type*} [Fintype d] (J : Matrix κ d ℝ) (p : κ → ℝ)
    (hp : ∑ y, p y = 1) :
    ∑ y, p y • vecMulVec (Jᵀ *ᵥ (Pi.single y 1 - p)) (Jᵀ *ᵥ (Pi.single y 1 - p))
      = Jᵀ * (diagonal p - vecMulVec p p) * J := by
  have h : ∀ u : κ → ℝ, vecMulVec (Jᵀ *ᵥ u) (Jᵀ *ᵥ u) = Jᵀ * vecMulVec u u * J := by
    intro u
    ext a b
    simp only [vecMulVec_apply, mulVec, dotProduct, transpose_apply, mul_apply,
      Finset.sum_mul, Finset.mul_sum]
    refine Finset.sum_congr rfl fun x _ => Finset.sum_congr rfl fun y _ => by ring
  simp_rw [h, ← fisher_identity p hp, Matrix.mul_sum, Matrix.sum_mul, Matrix.mul_smul,
    Matrix.smul_mul]

end Fisher

end SvdOmp
