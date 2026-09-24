#!/usr/bin/env python3
"""T1 -- brute-force reference for the paper's forward-backward chain.

Validates the recursions of paper 1 Sections IIIC and IV-E, and the calibration
conditions of Section IV-E, against EXPLICIT ENUMERATION over a finite model.

The point of the construction is that enumeration and recursion are different
methods: the recursions are the thing under test, the enumeration is the ground
truth, so agreement is evidence rather than tautology. The chain is a finite
HMM on a small alphabet precisely so that every quantity below -- including the
expectations over observation sequences -- is exact, with no Monte Carlo.

Checked here (all exact, to floating-point tolerance):

  C1  log-space forward recursion   == alpha by enumeration
  C2  log-space backward recursion  == beta  by enumeration
  C3  smoothed marginal, and the PAIRWISE smoothing distribution, == enumeration
  C4  sum_b gamma_t(b) = 1 ; sum_a gamma_t(a,b) = gamma_t(b) ;
      gamma_T == the filtered posterior at the final epoch (property 1 of IIIC)
  C5  the transition matrix is row-normalised
  C6  model correctly specified  =>  reported score == true posterior, so the
      estimator of Eq. (40) is unbiased EXACTLY, at every observation sequence
      (not merely in expectation over the data)
  C7  model misspecified  =>  the bias is non-zero and equals the
      distance-weighted calibration error of Eq. (41), computed through an
      independent code path
  C8  a score miscalibrated at EVERY epoch can still satisfy Eq. (41): the
      weighted sum cancels, so per-epoch calibration is sufficient but not
      necessary for unbiasedness
  C9  two pair distributions with IDENTICAL marginals give different expected
      billed distance and different expected cost -- the reason the money form
      needs the pair (Eq. (42)) and the marginals are not enough

C1 and C2 compare UN-NORMALISED quantities on purpose. The normalised check C3
would still pass if alpha and beta carried a common-mode scale error, since it
cancels in alpha*beta/sum(alpha*beta); C1 and C2 are what catch that. On the
first run of this file it was the enumeration, not the recursions, that was
wrong -- alpha was summed over full paths (giving P(z_1:T, x_t=b)) and beta was
summed over full paths rather than suffixes (giving a path count at t=T-1) --
and C3 passed throughout. Fixing the reference made C1 and C2 agree to 1e-16,
which is the sense in which the recursions are now validated rather than
merely self-consistent.

THIS FILE DELIBERATELY DOES NOT IMPORT ANYTHING FROM THE PROJECT. It is a
self-contained reference; no C++ build is required to run it. The companion
test that compares the C++ output against these numbers comes after the
backward recursion lands in src/mm/cmm/.

Run:  python tests/python/test_smoothing_bruteforce.py
Exit: 0 on all checks, 1 on the first failure.
"""

import itertools
import math
import sys

import numpy as np

TOL = 1e-10
FAILURES = []


# ----------------------------------------------------------------------------- helpers

def check(name, ok, detail=""):
    print(f"  [{'ok ' if ok else 'FAIL'}] {name}{('  ' + detail) if detail else ''}")
    if not ok:
        FAILURES.append(name)
    return ok


def logsumexp(x):
    x = np.asarray(x, dtype=float)
    m = np.max(x)
    if not np.isfinite(m):
        return m
    return m + math.log(float(np.sum(np.exp(x - m))))


def log0(x):
    """log with log(0) = -inf and no warning: a forbidden transition is -inf."""
    x = np.asarray(x, dtype=float)
    out = np.full(x.shape, -np.inf)
    np.log(x, out=out, where=x > 0)
    return out


def normalise_rows(m):
    """Row-normalise a matrix, so it is a proper conditional distribution."""
    m = np.asarray(m, dtype=float)
    return m / m.sum(axis=1, keepdims=True)


# ----------------------------------------------------------------------------- the chain

class Chain:
    """A finite HMM: candidates per epoch, transitions, and emissions on an alphabet.

    ep[t][s, b] = P(z_t = s | x_t = b)
    tp[t][a, b] = P(x_t = b | x_{t-1} = a), row-normalised
    ell[t][a, b] = the along-road step distance for the pair (a, b), Section IV-A
    """

    def __init__(self, pi, tp, ep, ell):
        self.pi = np.asarray(pi, dtype=float)
        self.tp = [np.asarray(x, dtype=float) for x in tp]
        self.ep = [np.asarray(x, dtype=float) for x in ep]
        self.ell = [np.asarray(x, dtype=float) for x in ell]
        self.T = len(self.ep)
        self.K = [e.shape[1] for e in self.ep]

    # -- enumeration -----------------------------------------------------------

    def paths(self):
        return itertools.product(*[range(k) for k in self.K])

    def joint(self, path, z):
        p = self.pi[path[0]] * self.ep[0][z[0], path[0]]
        for t in range(1, self.T):
            p *= self.tp[t - 1][path[t - 1], path[t]] * self.ep[t][z[t], path[t]]
        return p

    def p_obs(self, z):
        return sum(self.joint(p, z) for p in self.paths())

    def alpha_enum(self, z):
        """alpha_t(b) = P(z_{1:t}, x_t = b), by summing over PATH PREFIXES.

        Note the prefix: summing the full joint over paths through b would give
        P(z_{1:T}, x_t=b), which is alpha_t(b) * beta_t(b) and not alpha_t(b).
        """
        a = np.zeros((self.T, max(self.K)))
        for t in range(self.T):
            for path in itertools.product(*[range(k) for k in self.K[: t + 1]]):
                p = self.pi[path[0]] * self.ep[0][z[0], path[0]]
                for u in range(1, t + 1):
                    p *= self.tp[u - 1][path[u - 1], path[u]] * self.ep[u][z[u], path[u]]
                a[t, path[t]] += p
        return a

    def beta_enum(self, z):
        """beta_t(b) = P(z_{t+1:T} | x_t = b), by summing over PATH SUFFIXES.

        The suffix is the sequence x_{t+1..T-1} only, so at t = T-1 the sum has a
        single empty term and beta is 1 -- which is Eq. (beta_init), reached by
        enumeration rather than assumed.
        """
        b_ = np.zeros((self.T, max(self.K)))
        for t in range(self.T):
            for cand in range(self.K[t]):
                s = 0.0
                for suffix in itertools.product(*[range(k) for k in self.K[t + 1:]]):
                    p, prev = 1.0, cand
                    for u in range(t + 1, self.T):
                        nxt = suffix[u - t - 1]
                        p *= self.tp[u - 1][prev, nxt] * self.ep[u][z[u], nxt]
                        prev = nxt
                    s += p
                b_[t, cand] = s
        return b_

    def smoothing_enum(self, z):
        """gamma_t(b) and gamma_t(a,b) = P(x_{t-1}=a, x_t=b | z), by enumeration."""
        marg = np.zeros((self.T, max(self.K)))
        pair = [np.zeros((self.K[t - 1], self.K[t])) for t in range(1, self.T)]
        tot = self.p_obs(z)
        for path in self.paths():
            joint = self.joint(path, z) / tot
            for t in range(self.T):
                marg[t, path[t]] += joint
            for t in range(1, self.T):
                pair[t - 1][path[t - 1], path[t]] += joint
        return marg, pair

    # -- recursions (the thing under test) ------------------------------------

    def alpha_rec(self, z):
        log_a = np.full((self.T, max(self.K)), -np.inf)
        log_a[0, : self.K[0]] = log0(self.pi) + log0(self.ep[0][z[0], : self.K[0]])
        for t in range(1, self.T):
            prev = log_a[t - 1, : self.K[t - 1]]
            for b in range(self.K[t]):
                log_a[t, b] = logsumexp(prev + log0(self.tp[t - 1][:, b])) \
                    + math.log(self.ep[t][z[t], b])
        return log_a

    def beta_rec(self, z):
        log_b = np.full((self.T, max(self.K)), -np.inf)
        log_b[self.T - 1, : self.K[self.T - 1]] = 0.0  # Eq. (beta_init)
        for t in range(self.T - 2, -1, -1):
            for a in range(self.K[t]):
                log_b[t, a] = logsumexp(
                    log0(self.tp[t][a, : self.K[t + 1]])
                    + log0(self.ep[t + 1][z[t + 1], : self.K[t + 1]])
                    + log_b[t + 1, : self.K[t + 1]])
        return log_b

    def smoothing_rec(self, z):
        log_a, log_b = self.alpha_rec(z), self.beta_rec(z)
        marg = np.zeros((self.T, max(self.K)))
        pair = [np.zeros((self.K[t - 1], self.K[t])) for t in range(1, self.T)]
        for t in range(self.T):
            lg = log_a[t, : self.K[t]] + log_b[t, : self.K[t]]
            marg[t, : self.K[t]] = np.exp(lg - logsumexp(lg))
        for t in range(1, self.T):
            lg = (log_a[t - 1, : self.K[t - 1]][:, None]
                  + log0(self.tp[t - 1])
                  + log0(self.ep[t][z[t], : self.K[t]])[None, :]
                  + log_b[t, : self.K[t]][None, :])
            pair[t - 1] = np.exp(lg - logsumexp(lg.ravel()))
        return marg, pair

    def viterbi(self, z):
        """The reported path; Section IV holds the candidate fixed at this path."""
        log_d = np.full((self.T, max(self.K)), -np.inf)
        back = np.zeros((self.T, max(self.K)), dtype=int)
        log_d[0, : self.K[0]] = log0(self.pi) + log0(self.ep[0][z[0], : self.K[0]])
        for t in range(1, self.T):
            for b in range(self.K[t]):
                cand = log_d[t - 1, : self.K[t - 1]] + log0(self.tp[t - 1][:, b])
                back[t, b] = int(np.argmax(cand))
                log_d[t, b] = np.max(cand) + math.log(self.ep[t][z[t], b])
        path = [int(np.argmax(log_d[self.T - 1, : self.K[self.T - 1]]))]
        for t in range(self.T - 1, 0, -1):
            path.append(int(back[t, path[-1]]))
        return list(reversed(path))


# ----------------------------------------------------------------------------- fixtures

def build_chain(seed=None):
    """A small deterministic chain, or a random one under a fixed seed.

    candidate counts vary per epoch so that index errors between layers show up.
    """
    if seed is None:
        pi = np.array([0.6, 0.4])
        tp = [normalise_rows([[0.7, 0.3, 0.0], [0.1, 0.6, 0.3]]),
              normalise_rows([[0.5, 0.5], [0.2, 0.8], [0.9, 0.1]]),
              normalise_rows([[0.4, 0.3, 0.3], [0.3, 0.3, 0.4]])]
        ep = [np.array([[0.9, 0.2], [0.1, 0.8]]),
              np.array([[0.8, 0.3, 0.1], [0.2, 0.7, 0.9]]),
              np.array([[0.6, 0.4], [0.4, 0.6]]),
              np.array([[0.7, 0.5, 0.2], [0.3, 0.5, 0.8]])]
        ell = [np.array([[2.0, 9.0, 0.0], [0.0, 2.5, 11.0]]),
               np.array([[3.0, 3.5], [4.0, 2.4], [5.0, 6.0]]),
               np.array([[1.5, 8.0, 7.5], [8.5, 1.2, 8.0]])]
        return Chain(pi, tp, ep, ell)

    rng = np.random.default_rng(seed)
    K = [2, 3, 2, 3]
    A = 2
    pi = rng.dirichlet(np.ones(K[0]))
    tp = [normalise_rows(rng.dirichlet(np.ones(K[t]), size=K[t - 1])) for t in range(1, 4)]
    ep = [rng.uniform(0.05, 1.0, size=(A, K[t])) for t in range(4)]
    ell = [rng.uniform(1.0, 12.0, size=(K[t - 1], K[t])) for t in range(1, 4)]
    return Chain(pi, tp, ep, ell)


def all_obs(chain):
    return list(itertools.product(*[range(chain.ep[t].shape[0]) for t in range(chain.T)]))


# ----------------------------------------------------------------------------- checks

def c1_c5(chain, label):
    """The recursions agree with enumeration, and the smoothing laws hold."""
    worst = {"alpha": 0.0, "beta": 0.0, "marg": 0.0, "pair": 0.0, "law": 0.0}
    prop1 = True
    for z in all_obs(chain):
        a_e, b_e = chain.alpha_enum(z), chain.beta_enum(z)
        a_r, b_r = chain.alpha_rec(z), chain.beta_rec(z)
        for t in range(chain.T):
            k = chain.K[t]
            worst["alpha"] = max(worst["alpha"], float(np.max(np.abs(
                np.exp(a_r[t, :k]) - a_e[t, :k]))))
            worst["beta"] = max(worst["beta"], float(np.max(np.abs(
                np.exp(b_r[t, :k]) - b_e[t, :k]))))
        m_e, p_e = chain.smoothing_enum(z)
        m_r, p_r = chain.smoothing_rec(z)
        for t in range(chain.T):
            worst["marg"] = max(worst["marg"], float(np.max(np.abs(
                m_r[t, : chain.K[t]] - m_e[t, : chain.K[t]]))))
            worst["law"] = max(worst["law"], abs(float(m_r[t, : chain.K[t]].sum()) - 1.0))
        for t in range(1, chain.T):
            worst["pair"] = max(worst["pair"], float(np.max(np.abs(p_r[t - 1] - p_e[t - 1]))))
            # property 2 of IIIC, transferred to the pair: summing over the origin
            # must give the marginal of the destination
            worst["law"] = max(worst["law"], float(np.max(np.abs(
                p_r[t - 1].sum(axis=0) - m_r[t, : chain.K[t]]))))
        # property 1: beta_T = 1 makes the smoothed score equal the filtered one
        fa = np.exp(a_r[chain.T - 1, : chain.K[-1]])
        fa = fa / fa.sum()
        prop1 &= bool(np.max(np.abs(fa - m_r[chain.T - 1, : chain.K[-1]])) < TOL)

    kind = "deterministic" if label == "det" else f"seed {label}"
    check(f"C1 alpha recursion == enumeration ({kind})", worst["alpha"] < TOL,
          f"max abs err {worst['alpha']:.2e}")
    check(f"C2 beta recursion == enumeration ({kind})", worst["beta"] < TOL,
          f"max abs err {worst['beta']:.2e}")
    check(f"C3a smoothed marginal == enumeration ({kind})", worst["marg"] < TOL,
          f"max abs err {worst['marg']:.2e}")
    check(f"C3b pairwise smoothing == enumeration ({kind})", worst["pair"] < TOL,
          f"max abs err {worst['pair']:.2e}")
    check(f"C4 normalisation and the marginal/pair identity ({kind})", worst["law"] < TOL,
          f"max abs err {worst['law']:.2e}")
    check(f"C4 gamma_T == filtered posterior (property 1) ({kind})", prop1)
    check(f"C5 transitions are row-normalised ({kind})",
          all(abs(float(m.sum(axis=1)[i]) - 1.0) < TOL
              for m in chain.tp for i in range(m.shape[0])))


def billed_steps(chain, path):
    """B_t of Section IV-A: the along-road length the bill places on each step."""
    return np.array([chain.ell[t][path[t], path[t + 1]] for t in range(chain.T - 1)])


def c6_c7(true_chain, reported_chain, label):
    """Calibration of the reported score <=> unbiasedness of the estimator.

    Both the estimand M of Eq. (37) and the estimator M-hat of Eq. (40) are
    computed exactly, by summing over every observation sequence with the TRUE
    probability of that sequence. Nothing is sampled.
    """
    bias, cond41 = 0.0, 0.0
    exact_at_every_z = True
    for z in all_obs(true_chain):
        pz = true_chain.p_obs(z)
        if pz <= 0.0:
            continue
        path = reported_chain.viterbi(z)
        b_steps = billed_steps(reported_chain, path)

        # true posterior of the reported candidate, given the data
        marg_true, _ = true_chain.smoothing_enum(z)
        p_true = np.array([marg_true[t, path[t]] for t in range(true_chain.T)])

        # reported score: the smoothed marginal at the same candidate
        marg_rep, _ = reported_chain.smoothing_rec(z)
        tw = np.array([marg_rep[t, path[t]] for t in range(reported_chain.T)])

        m_true = float(np.sum(b_steps * (1.0 - p_true[1:])))
        m_hat = float(np.sum(b_steps * (1.0 - tw[1:])))
        bias += pz * (m_hat - m_true)
        cond41 += pz * float(np.sum(b_steps * (tw[1:] - p_true[1:])))
        exact_at_every_z &= bool(np.max(np.abs(tw - p_true)) < TOL)

    kind = "correctly specified" if label == "same" else "misspecified"
    if label == "same":
        check(f"C6 reported score == true posterior at every observation ({kind})",
              exact_at_every_z)
        check(f"C6 estimator is exactly unbiased ({kind})", abs(bias) < TOL,
              f"bias {bias:.2e} m")
    else:
        check(f"C7 a misspecified model is biased ({kind})", abs(bias) > 1e-6,
              f"bias {bias:+.4f} m")
        # m_hat - m_true = sum B (p - tw), so the bias is MINUS the Eq. (41) sum
        check(f"C7 bias == minus the weighted calibration error of Eq. (41) ({kind})",
              abs(bias + cond41) < TOL, f"|bias + sum| = {abs(bias + cond41):.2e}")


def c8_cancellation():
    """Per-epoch calibration is sufficient for Eq. (41), not necessary."""
    chain = build_chain()
    z = (0, 1, 0, 1)
    marg_rep, _ = chain.smoothing_rec(z)
    path = chain.viterbi(z)
    b_steps = billed_steps(chain, path)
    p_true = np.array([marg_rep[t, path[t]] for t in range(chain.T)])[1:]
    b = b_steps

    # perturb two epochs by equal and opposite amounts, so the weighted sum
    # vanishes while every perturbed epoch is individually miscalibrated
    delta = 0.01                       # probability units, so the clip never fires
    d = delta * float(min(b[0], b[1]))
    tw = p_true.copy()
    tw[0] += d / b[0]
    tw[1] -= d / b[1]
    assert 0.0 < tw[0] < 1.0 and 0.0 < tw[1] < 1.0

    weighted = float(np.sum(b * (tw - p_true)))
    per_epoch_ok = bool(np.max(np.abs(tw - p_true)) < TOL)
    check("C8 the perturbation really is miscalibrated at every perturbed epoch",
          not per_epoch_ok, f"max |tw - p| = {np.max(np.abs(tw - p_true)):.3f}")
    check("C8 yet Eq. (41) holds exactly: the weighted sum cancels",
          abs(weighted) < TOL, f"weighted sum {weighted:.2e} m")


def c9_pair_beats_marginals():
    """Same marginals, different dependence, different expected billed distance."""
    marg = np.array([0.5, 0.5])
    indep = np.outer(marg, marg)                       # the vehicle jumps at random
    sticky = np.diag(marg)                             # the vehicle stays put
    ell = np.array([[1.0, 10.0], [10.0, 1.0]])         # a jump between roads is long
    rate = np.array([1.0, 0.0])                        # candidate 0 is the charged one

    check("C9 the two pair distributions have identical marginals",
          np.allclose(indep.sum(axis=0), marg) and np.allclose(indep.sum(axis=1), marg)
          and np.allclose(sticky.sum(axis=0), marg))
    d_indep = float(np.sum(indep * ell))
    d_sticky = float(np.sum(sticky * ell))
    check("C9 expected billed distance differs", abs(d_indep - d_sticky) > 1e-6,
          f"independent {d_indep:.2f} m vs sticky {d_sticky:.2f} m")
    cost_indep = float(np.sum(indep * ell * rate[None, :]))
    cost_sticky = float(np.sum(sticky * ell * rate[None, :]))
    check("C9 expected cost differs too, so the money form needs the pair",
          abs(cost_indep - cost_sticky) > 1e-6,
          f"independent {cost_indep:.2f} vs sticky {cost_sticky:.2f}")


def main():
    print("T1 -- brute-force reference for the forward-backward chain")
    print("(enumeration is ground truth; the recursions are what is being tested)\n")

    for label, chain in [("det", build_chain())] + \
                        [(str(s), build_chain(s)) for s in (0, 1, 2, 3, 4)]:
        c1_c5(chain, label)

    print()
    same = build_chain()
    c6_c7(same, same, "same")

    # misspecify only the reported transition matrix: the score is then the
    # posterior of the wrong model, which is what a wrong HMM gives in practice
    true_chain = build_chain()
    bad = Chain(true_chain.pi,
                [normalise_rows(m + 0.35) for m in true_chain.tp],
                true_chain.ep, true_chain.ell)
    c6_c7(true_chain, bad, "bad")

    print()
    c8_cancellation()
    c9_pair_beats_marginals()

    print()
    if FAILURES:
        print(f"FAILED: {len(FAILURES)} check(s): " + ", ".join(FAILURES))
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
