"""Smoke test for distillation loss with entropy focus.

Validates:
1. entropy_focus_weights correctly up-weights high-entropy teacher tokens
2. Loss values are numerically stable across entropy levels
3. Gradients flow correctly and focus on high-entropy positions
4. Temperature scaling effect on peaked teacher distributions
5. Edge cases: all-zero, single-token, uniform teachers
6. build_teacher_topk produces valid distributions at different temperatures
7. End-to-end: chunked backward with entropy focus
"""
import sys
import math
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from src.distillation.loss import (
    build_teacher_topk,
    skew_kld_loss,
    compute_loss_chunked_backward,
)
from src.distillation.qead import (
    entropy_focus_weights,
    teacher_confidence_weights,
    fused_qead_kld_pass,
)

PASS = 0
FAIL = 0


def check(name, condition, detail=""):
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  PASS  {name}")
    else:
        FAIL += 1
        print(f"  FAIL  {name}  {detail}")


def test_entropy_focus_weights():
    print("\n=== Test 1: entropy_focus_weights direction ===")
    B, S, K = 2, 64, 20

    teacher_ids = torch.randint(0, 1000, (B, S, K))
    teacher_probs = torch.zeros(B, S, K)

    for b in range(B):
        for t in range(S):
            if t < S // 2:
                teacher_probs[b, t, 0] = 0.90
                remaining = 0.10 / (K - 1)
                teacher_probs[b, t, 1:] = remaining
            else:
                teacher_probs[b, t] = 1.0 / K

    ew = entropy_focus_weights(teacher_probs)
    peaked_mean = ew[:, : S // 2].mean().item()
    spread_mean = ew[:, S // 2 :].mean().item()

    check(
        "High-entropy tokens get MORE weight than peaked tokens",
        spread_mean > peaked_mean,
        f"spread={spread_mean:.4f} peaked={peaked_mean:.4f}",
    )
    check(
        "Ratio spread/peaked >= 2x",
        spread_mean / peaked_mean >= 2.0,
        f"ratio={spread_mean / peaked_mean:.2f}x",
    )

    cw = teacher_confidence_weights(teacher_probs)
    peaked_conf = cw[:, : S // 2].mean().item()
    spread_conf = cw[:, S // 2 :].mean().item()
    check(
        "Old confidence_weights is INVERTED (peaked > spread)",
        peaked_conf > spread_conf,
        f"peaked={peaked_conf:.4f} spread={spread_conf:.4f}",
    )

    print(f"\n  Summary: entropy_focus peaked={peaked_mean:.4f}, spread={spread_mean:.4f}")
    print(f"           old confidence  peaked={peaked_conf:.4f}, spread={spread_conf:.4f}")


def test_entropy_focus_edge_cases():
    print("\n=== Test 2: Edge cases ===")
    K = 20

    zero_probs = torch.zeros(1, 4, K)
    ew_zero = entropy_focus_weights(zero_probs)
    check("All-zero teacher → zero weights", ew_zero.sum().item() == 0.0)

    single_probs = torch.zeros(1, 4, K)
    single_probs[:, :, 0] = 1.0
    ew_single = entropy_focus_weights(single_probs, floor=0.1)
    check(
        "Single-token (max peaked) → weight = floor (0.1)",
        abs(ew_single.mean().item() - 0.1) < 0.01,
        f"got {ew_single.mean().item():.4f}",
    )

    uniform_probs = torch.ones(1, 4, K) / K
    ew_uniform = entropy_focus_weights(uniform_probs, floor=0.1)
    check(
        "Perfectly uniform → weight = 1.0",
        abs(ew_uniform.mean().item() - 1.0) < 0.01,
        f"got {ew_uniform.mean().item():.4f}",
    )

    floor_probs = torch.zeros(1, 4, K)
    floor_probs[:, :, 0] = 0.95
    floor_probs[:, :, 1] = 0.05
    ew_floor = entropy_focus_weights(floor_probs, floor=0.1)
    check(
        "Floor prevents zero-weight on peaked tokens",
        ew_floor.min().item() >= 0.1,
        f"min={ew_floor.min().item():.4f}",
    )

    big = torch.ones(4, 2048, K) / K
    ew_big = entropy_focus_weights(big)
    check(
        "Large tensor doesn't OOM or NaN",
        not torch.isnan(ew_big).any() and not torch.isinf(ew_big).any(),
    )


def test_temperature_scaling():
    print("\n=== Test 3: Temperature scaling on peaked teacher ===")
    logprobs = [-0.26, -3.0, -4.6, -5.3] + [-6.0] * 16
    K = len(logprobs)

    results = {}
    for temp in [1.0, 3.0, 5.0, 8.0]:
        inv_t = 1.0 / temp
        probs = [math.exp(lp * inv_t) for lp in logprobs]
        total = sum(probs)
        probs = [p / total for p in probs]
        ent = -sum(p * math.log(p + 1e-30) for p in probs)
        results[temp] = {"top1": probs[0], "entropy": ent}
        print(f"  T={temp:.0f}: top1={probs[0]:.3f}, top2={probs[1]:.3f}, entropy={ent:.3f}")

    check(
        "Higher T spreads distribution (T5 top1 < T3 top1)",
        results[5.0]["top1"] < results[3.0]["top1"],
    )
    check(
        "Higher T increases entropy (T5 > T3)",
        results[5.0]["entropy"] > results[3.0]["entropy"],
    )
    check(
        "T=5 reduces top1 below 40%",
        results[5.0]["top1"] < 0.40,
        f"top1={results[5.0]['top1']:.3f}",
    )


def test_build_teacher_topk_temperatures():
    print("\n=== Test 4: build_teacher_topk at different temperatures ===")
    K = 20
    V = 1000
    top_k_ids = [list(range(K)) for _ in range(8)]
    top_k_vals = [[-0.26] + [-3.0 - i * 0.5 for i in range(K - 1)] for _ in range(8)]

    for temp in [3.0, 5.0]:
        ids, probs = build_teacher_topk(top_k_ids, top_k_vals, V, K, torch.device("cpu"), temp)
        check(
            f"T={temp}: probs sum to ~1.0 per position",
            (probs.sum(dim=-1) - 1.0).abs().max().item() < 0.01,
            f"max_dev={( probs.sum(dim=-1) - 1.0).abs().max().item():.4f}",
        )
        check(f"T={temp}: no NaN/Inf", not torch.isnan(probs).any() and not torch.isinf(probs).any())
        top1_mean = probs[:, 0].mean().item()
        print(f"  T={temp}: mean top1 prob = {top1_mean:.4f}")

    ids3, probs3 = build_teacher_topk(top_k_ids, top_k_vals, V, K, torch.device("cpu"), 3.0)
    ids5, probs5 = build_teacher_topk(top_k_ids, top_k_vals, V, K, torch.device("cpu"), 5.0)
    check(
        "T=5 spreads more than T=3 (lower top1 mean)",
        probs5[:, 0].mean().item() < probs3[:, 0].mean().item(),
    )


def test_loss_with_entropy_focus():
    print("\n=== Test 5: Skew-KLD loss with entropy focus weights ===")
    torch.manual_seed(42)
    B, S, V, K = 2, 32, 1000, 20

    student_logits = torch.randn(B, S, V)
    teacher_ids = torch.randint(0, V, (B, S, K))
    teacher_probs = torch.zeros(B, S, K)
    for b in range(B):
        for t in range(S):
            if t < S // 2:
                teacher_probs[b, t, 0] = 0.90
                teacher_probs[b, t, 1:] = 0.10 / (K - 1)
            else:
                teacher_probs[b, t] = 1.0 / K

    uniform_w = torch.ones(B, S) / (B * S)
    loss_uniform = skew_kld_loss(student_logits, teacher_ids, teacher_probs, uniform_w, 0.2, 5.0)
    check("Loss with uniform weights is finite", torch.isfinite(loss_uniform))
    check("Loss with uniform weights is non-negative", loss_uniform.item() >= 0)

    ew = entropy_focus_weights(teacher_probs)
    ew_norm = ew / ew.sum().clamp(min=1e-8)
    loss_entropy = skew_kld_loss(student_logits, teacher_ids, teacher_probs, ew_norm, 0.2, 5.0)
    check("Loss with entropy focus is finite", torch.isfinite(loss_entropy))
    check("Loss with entropy focus is non-negative", loss_entropy.item() >= 0)

    print(f"\n  Loss uniform_weights: {loss_uniform.item():.6f}")
    print(f"  Loss entropy_focus:   {loss_entropy.item():.6f}")


def test_gradient_flow():
    print("\n=== Test 6: Gradient flow with entropy focus ===")
    torch.manual_seed(42)
    B, S, V, K = 1, 32, 1000, 20

    teacher_ids = torch.randint(0, V, (B, S, K))
    teacher_probs = torch.zeros(B, S, K)
    for t in range(S):
        if t < S // 2:
            teacher_probs[0, t, 0] = 0.90
            teacher_probs[0, t, 1:] = 0.10 / (K - 1)
        else:
            teacher_probs[0, t] = 1.0 / K

    student_logits = torch.randn(B, S, V, requires_grad=True)
    ew = entropy_focus_weights(teacher_probs)
    ew_norm = ew / ew.sum().clamp(min=1e-8)

    loss = skew_kld_loss(student_logits, teacher_ids, teacher_probs, ew_norm, 0.2, 5.0)
    loss.backward()

    grad = student_logits.grad
    check("Gradients exist", grad is not None)
    check("No NaN gradients", not torch.isnan(grad).any())
    check("No Inf gradients", not torch.isinf(grad).any())

    grad_peaked = grad[:, : S // 2].abs().mean().item()
    grad_spread = grad[:, S // 2 :].abs().mean().item()
    print(f"\n  Grad magnitude peaked tokens: {grad_peaked:.6f}")
    print(f"  Grad magnitude spread tokens: {grad_spread:.6f}")
    print(f"  Ratio spread/peaked: {grad_spread / max(grad_peaked, 1e-10):.2f}x")

    check(
        "Entropy focus directs MORE gradient to high-entropy tokens",
        grad_spread > grad_peaked,
        f"spread={grad_spread:.6f} peaked={grad_peaked:.6f}",
    )


def test_chunked_backward_with_entropy():
    print("\n=== Test 7: compute_loss_chunked_backward with entropy focus ===")
    torch.manual_seed(42)
    B, S, H, V, K = 1, 64, 128, 1000, 20

    hidden = torch.randn(B, S, H, requires_grad=True)
    lm_head = torch.randn(V, H)
    teacher_ids = torch.randint(0, V, (B, S, K))
    teacher_probs = torch.zeros(B, S, K)
    for t in range(S):
        teacher_probs[0, t] = 1.0 / K

    ew = entropy_focus_weights(teacher_probs)
    ew_norm = ew / ew.sum(dim=-1, keepdim=True).clamp(min=1e-8)

    labels = torch.full((B, S), -100, dtype=torch.long)
    labels[0, 10:] = torch.randint(0, V, (S - 10,))

    total, distill, task = compute_loss_chunked_backward(
        hidden, lm_head, teacher_ids, teacher_probs, ew_norm, labels,
        alpha=0.7, skew_lambda=0.2, temperature=5.0, chunk_size=32,
    )

    check("Chunked backward total loss is finite", math.isfinite(total))
    check("Chunked backward distill loss is finite", math.isfinite(distill))
    check("Chunked backward task loss is finite", math.isfinite(task))
    check("Hidden has gradients after chunked backward", hidden.grad is not None)
    if hidden.grad is not None:
        check("Hidden gradients are finite", torch.isfinite(hidden.grad).all())
    print(f"\n  total={total:.4f}, distill={distill:.4f}, task={task:.4f}")


def test_fused_qead_with_entropy():
    print("\n=== Test 8: fused_qead_kld_pass integration ===")
    torch.manual_seed(42)
    B, S, H, V, K = 1, 32, 128, 1000, 20

    hidden = torch.randn(B, S, H)
    lm_head = torch.randn(V, H)
    response_mask = torch.zeros(B, S, dtype=torch.bool)
    response_mask[0, 8:] = True
    teacher_ids = torch.randint(0, V, (B, S, K))
    teacher_probs = torch.zeros(B, S, K)
    for t in range(S):
        if response_mask[0, t]:
            teacher_probs[0, t] = 1.0 / K

    qead_w, kld = fused_qead_kld_pass(
        hidden, lm_head, response_mask, teacher_ids, teacher_probs,
        temperature=5.0, compute_kld=True,
    )

    check("QEAD weights shape matches", qead_w.shape == (B, S))
    check("QEAD weights are finite", torch.isfinite(qead_w).all())
    check("KLD is finite", torch.isfinite(kld).all())

    valid_teacher = teacher_probs.sum(dim=-1) > 1e-8
    qead_w = qead_w * valid_teacher.float()
    ew = entropy_focus_weights(teacher_probs)
    qead_w = qead_w * ew
    w_sums = qead_w.sum(dim=-1, keepdim=True).clamp(min=1e-8)
    qead_w = qead_w / w_sums

    check("Final weights sum to ~1.0 per sample", abs(qead_w.sum().item() - 1.0) < 0.01)
    check("Prompt tokens have zero weight", qead_w[0, :8].sum().item() == 0.0)
    check("Response tokens have positive weight", qead_w[0, 8:].sum().item() > 0.0)


def test_realistic_teacher_distribution():
    print("\n=== Test 9: Realistic OCR-Nemotron-7B distribution (77% peaked) ===")
    torch.manual_seed(42)
    B, S, K = 1, 100, 20

    teacher_probs = torch.zeros(B, S, K)
    for t in range(S):
        main_prob = 0.70 + 0.15 * torch.rand(1).item()
        remaining = 1.0 - main_prob
        teacher_probs[0, t, 0] = main_prob
        teacher_probs[0, t, 1:] = remaining / (K - 1)

    ew = entropy_focus_weights(teacher_probs, floor=0.1)
    mean_weight = ew.mean().item()
    min_weight = ew.min().item()
    max_weight = ew.max().item()

    print(f"  77%-peaked teacher: mean_w={mean_weight:.4f}, min={min_weight:.4f}, max={max_weight:.4f}")
    check("Mean weight is low (peaked teacher)", mean_weight < 0.5)
    check("Floor prevents zero weights", min_weight >= 0.1)

    informative = torch.zeros(B, S, K)
    for t in range(S):
        if t % 5 == 0:
            informative[0, t] = 1.0 / K
        else:
            informative[0, t, 0] = 0.77
            informative[0, t, 1:] = 0.23 / (K - 1)

    ew2 = entropy_focus_weights(informative, floor=0.1)
    informative_w = ew2[0, ::5].mean().item()
    peaked_w = ew2[0, 1::5].mean().item()
    print(f"  Mixed (20% informative): informative_w={informative_w:.4f}, peaked_w={peaked_w:.4f}")
    check(
        "Informative tokens weighted 3x+ more than peaked",
        informative_w / peaked_w >= 3.0,
        f"ratio={informative_w / peaked_w:.2f}x",
    )


def main():
    print("=" * 70)
    print("  Distillation Loss Smoke Test — Entropy Focus")
    print("=" * 70)

    test_entropy_focus_weights()
    test_entropy_focus_edge_cases()
    test_temperature_scaling()
    test_build_teacher_topk_temperatures()
    test_loss_with_entropy_focus()
    test_gradient_flow()
    test_chunked_backward_with_entropy()
    test_fused_qead_with_entropy()
    test_realistic_teacher_distribution()

    print("\n" + "=" * 70)
    total = PASS + FAIL
    if FAIL == 0:
        print(f"  ALL {total} TESTS PASSED")
    else:
        print(f"  {PASS}/{total} passed, {FAIL} FAILED")
    print("=" * 70)
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
