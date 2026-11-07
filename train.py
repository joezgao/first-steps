"""First steps: a tiny agent learns to reach a light without falling into a pit.

REINFORCE with a learned value baseline, in NumPy only. Every gradient below is written by hand
(no autograd), and checked against finite differences before training starts (gradcheck()).

    python3 train.py            # trains, writes ckpt/ep_XXXXX.npz, log.json
    python3 train.py --check    # gradient check only

The environment is duplicated exactly in index.html (stepEnv), so the page runs the same world.
"""
import json, os, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CKPT = os.path.join(HERE, "ckpt")

# ---------------------------------------------------------------- world
LIGHT = np.array([0.0, 0.72])     # where the light is
R_GOAL = 0.09                     # reaching it: within this distance
PIT_C = np.array([0.0, 0.04])     # the pit is an ellipse between the start band and the light
PIT_A = np.array([0.44, 0.22])    # semi-axes (x, y)
START_X = (-0.6, 0.6)             # each episode starts at a uniform random point in this band
START_Y = (-0.88, -0.62)
STEP = 0.06                       # one action moves the agent by at most this much per axis
WALL = 0.97                       # the world is the square [-WALL, WALL]^2; walls clamp
T_MAX = 100                       # steps before the episode times out
R_PROGRESS = 2.0                  # reward per unit of distance gained towards the light
R_TIME = -0.005                   # per step
R_REACH = 2.0
R_PIT = -1.0
GAMMA = 0.99


def in_pit(p):
    q = (p - PIT_C) / PIT_A
    return (q ** 2).sum(-1) < 1.0


def reset(rng, n):
    return np.stack([rng.uniform(*START_X, n), rng.uniform(*START_Y, n)], -1)


def step_env(p, a):
    """p: (n,2) positions, a: (n,2) raw actions. Returns new p, reward, outcome (0 running, 1 light, 2 pit)."""
    d0 = np.linalg.norm(p - LIGHT, axis=-1)
    p = np.clip(p + STEP * np.clip(a, -1.0, 1.0), -WALL, WALL)
    d1 = np.linalg.norm(p - LIGHT, axis=-1)
    r = R_PROGRESS * (d0 - d1) + R_TIME
    out = np.zeros(len(p), np.int8)
    reach = d1 < R_GOAL
    pit = in_pit(p) & ~reach
    out[reach] = 1
    out[pit] = 2
    r = r + R_REACH * reach + R_PIT * pit
    return p, r, out


# ---------------------------------------------------------------- tiny MLP with hand-written backprop
class MLP:
    """Layers: tanh hidden layers, output either tanh ('tanh') or linear ('lin')."""

    def __init__(self, sizes, out_act, rng, out_scale=1.0):
        self.sizes, self.out_act = sizes, out_act
        self.W, self.b = [], []
        for i, (m, n) in enumerate(zip(sizes[:-1], sizes[1:])):
            s = 1.0 / np.sqrt(m) * (out_scale if i == len(sizes) - 2 else 1.0)
            self.W.append(rng.normal(0, s, (m, n)))
            self.b.append(np.zeros(n))

    def params(self):
        return self.W + self.b

    def forward(self, x):
        self.hs = [x]
        h = x
        L = len(self.W)
        for i in range(L):
            z = h @ self.W[i] + self.b[i]
            h = np.tanh(z) if (i < L - 1 or self.out_act == "tanh") else z
            self.hs.append(h)
        return h

    def backward(self, dout):
        """dout: dLoss/d(output). Returns grads in the same order as params()."""
        L = len(self.W)
        gW, gb = [None] * L, [None] * L
        g = dout
        for i in reversed(range(L)):
            h = self.hs[i + 1]
            if i < L - 1 or self.out_act == "tanh":
                g = g * (1.0 - h * h)          # d tanh(z)/dz = 1 - tanh(z)^2
            gW[i] = self.hs[i].T @ g           # z = h_prev W + b
            gb[i] = g.sum(0)
            g = g @ self.W[i].T
        return gW + gb


class Adam:
    def __init__(self, params, lr, b1=0.9, b2=0.999, eps=1e-8):
        self.p, self.lr, self.b1, self.b2, self.eps = params, lr, b1, b2, eps
        self.m = [np.zeros_like(x) for x in params]
        self.v = [np.zeros_like(x) for x in params]
        self.t = 0

    def step(self, grads):
        self.t += 1
        for x, g, m, v in zip(self.p, grads, self.m, self.v):
            m *= self.b1; m += (1 - self.b1) * g
            v *= self.b2; v += (1 - self.b2) * g * g
            mh = m / (1 - self.b1 ** self.t)
            vh = v / (1 - self.b2 ** self.t)
            x -= self.lr * mh / (np.sqrt(vh) + self.eps)   # in place, so MLP arrays update


# ---------------------------------------------------------------- policy-gradient loss and its gradient
LOG_STD_MIN, LOG_STD_MAX = -2.3, 0.3


def policy_loss_and_grads(pi, log_std, S, A, Adv):
    """Loss = -mean_t[ Adv_t * log N(A_t ; mu(S_t), sigma^2) ].
    d logN/d mu = (a - mu) / sigma^2 ;  d logN/d log_sigma = (a - mu)^2 / sigma^2 - 1 (per action dim)."""
    N = len(S)
    mu = pi.forward(S)
    var = np.exp(2 * log_std)
    logp = (-0.5 * (A - mu) ** 2 / var - log_std - 0.5 * np.log(2 * np.pi)).sum(-1)
    loss = -(Adv * logp).mean()
    dmu = -(Adv[:, None] * (A - mu) / var) / N
    dlogstd = -(Adv[:, None] * ((A - mu) ** 2 / var - 1.0)).sum(0) / N
    return loss, pi.backward(dmu), dlogstd


def value_loss_and_grads(vf, S, G):
    v = vf.forward(S)[:, 0]
    loss = 0.5 * ((v - G) ** 2).mean()
    return loss, vf.backward(((v - G) / len(S))[:, None])


def gradcheck():
    rng = np.random.default_rng(0)
    pi = MLP([2, 8, 8, 2], "tanh", rng, 0.5)
    vf = MLP([2, 8, 1], "lin", rng)
    log_std = np.array([-0.3, 0.2])
    S = rng.uniform(-1, 1, (7, 2)); A = rng.normal(0, 1, (7, 2)); Adv = rng.normal(0, 1, 7); G = rng.normal(0, 1, 7)
    worst = 0.0

    def num(f, x, i):
        old = x.flat[i]; h = 1e-6
        x.flat[i] = old + h; lp = f()
        x.flat[i] = old - h; lm = f()
        x.flat[i] = old
        return (lp - lm) / (2 * h)

    _, gp, gs = policy_loss_and_grads(pi, log_std, S, A, Adv)
    f = lambda: policy_loss_and_grads(pi, log_std, S, A, Adv)[0]
    for x, g in list(zip(pi.params(), gp)) + [(log_std, gs)]:
        for i in range(x.size):
            n = num(f, x, i); worst = max(worst, abs(n - g.flat[i]) / max(1e-8, abs(n) + abs(g.flat[i])))
    _, gv = value_loss_and_grads(vf, S, G)
    f = lambda: value_loss_and_grads(vf, S, G)[0]
    for x, g in zip(vf.params(), gv):
        for i in range(x.size):
            n = num(f, x, i); worst = max(worst, abs(n - g.flat[i]) / max(1e-8, abs(n) + abs(g.flat[i])))
    return worst


# ---------------------------------------------------------------- rollouts
def rollout(pi, log_std, rng, n, greedy=False):
    """Runs n episodes in parallel. Returns per-step arrays (masked) and per-episode outcomes."""
    p = reset(rng, n)
    alive = np.ones(n, bool)
    S, A, R, M = [], [], [], []
    outcome = np.zeros(n, np.int8)
    steps = np.full(n, T_MAX)
    ret = np.zeros(n)
    for t in range(T_MAX):
        mu = pi.forward(p)
        a = mu if greedy else mu + np.exp(log_std) * rng.normal(size=mu.shape)
        p2, r, out = step_env(p, a)
        r = r * alive
        S.append(p.copy()); A.append(a); R.append(r); M.append(alive.copy())
        ret += r
        newly = alive & (out > 0)
        outcome[newly] = out[newly]
        steps[newly] = t + 1
        alive = alive & (out == 0)
        p = np.where(alive[:, None], p2, p)
        if not alive.any():
            break
    return dict(S=np.array(S), A=np.array(A), R=np.array(R), M=np.array(M),
                outcome=outcome, steps=steps, ret=ret)


def evaluate(pi, log_std, seed, n=1000):
    ro = rollout(pi, log_std, np.random.default_rng(seed), n)
    o = ro["outcome"]
    ok = o == 1
    return dict(light=float(ok.mean()), pit=float((o == 2).mean()), timeout=float((o == 0).mean()),
                steps_to_light=float(ro["steps"][ok].mean()) if ok.any() else None,
                ret=float(ro["ret"].mean()))


# ---------------------------------------------------------------- training
SEED = 7
HID = 48
BATCH = 32                         # episodes per policy update
EPISODES = 10240
LR_PI, LR_V, V_ITERS = 5e-3, 3e-3, 4
CKPT_EPISODES = [0, 32, 64, 128, 192, 256, 384, 512, 768, 1024, 1536, 2048, 3072, 4096, 5120, 6144, 7168, 8192, 9216, 10240]


def save_ckpt(ep, pi, vf, log_std, ev):
    os.makedirs(CKPT, exist_ok=True)
    d = {f"pi_W{i}": w for i, w in enumerate(pi.W)} | {f"pi_b{i}": b for i, b in enumerate(pi.b)}
    d |= {f"v_W{i}": w for i, w in enumerate(vf.W)} | {f"v_b{i}": b for i, b in enumerate(vf.b)}
    np.savez(os.path.join(CKPT, f"ep_{ep:05d}.npz"), log_std=log_std, eval=json.dumps(ev), **d)


def train():
    worst = gradcheck()
    print(f"gradcheck worst relative error {worst:.2e}")
    assert worst < 1e-5
    rng = np.random.default_rng(SEED)
    pi = MLP([2, HID, HID, 2], "tanh", rng, out_scale=0.5)
    vf = MLP([2, HID, HID, 1], "lin", rng)
    log_std = np.zeros(2)
    opt_pi = Adam(pi.params() + [log_std], LR_PI)
    opt_v = Adam(vf.params(), LR_V)
    log = dict(config=dict(seed=SEED, hidden=HID, batch=BATCH, episodes=EPISODES, lr_pi=LR_PI, lr_v=LR_V,
                           v_iters=V_ITERS, gamma=GAMMA, t_max=T_MAX, step=STEP,
                           params_policy=int(sum(x.size for x in pi.params()) + 2),
                           params_value=int(sum(x.size for x in vf.params()))),
               gradcheck=worst, curve=dict(episode=[], light=[], pit=[], ret=[]), ckpt=[])
    ep, t0 = 0, time.time()
    while True:
        if ep in CKPT_EPISODES:
            ev = evaluate(pi, log_std, seed=1234)
            ev["episode"] = ep
            ev["sigma"] = [float(x) for x in np.exp(log_std)]
            log["ckpt"].append(ev)
            save_ckpt(ep, pi, vf, log_std, ev)
            print(f"ep {ep:5d}  light {ev['light']:.3f}  pit {ev['pit']:.3f}  timeout {ev['timeout']:.3f}  "
                  f"steps {ev['steps_to_light']}  sigma {ev['sigma']}  ({time.time()-t0:.0f}s)", flush=True)
        if ep >= EPISODES:
            break
        ro = rollout(pi, log_std, rng, BATCH)
        T = len(ro["R"])
        G = np.zeros_like(ro["R"]); acc = np.zeros(BATCH)
        for t in reversed(range(T)):                       # discounted return-to-go
            acc = ro["R"][t] + GAMMA * acc * ro["M"][t]
            G[t] = acc
        m = ro["M"].reshape(-1)
        S = ro["S"].reshape(-1, 2)[m]; A = ro["A"].reshape(-1, 2)[m]; Gf = G.reshape(-1)[m]
        V = vf.forward(S)[:, 0]
        adv = Gf - V                                       # the baseline: subtract what the critic expected
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        _, gp, gs = policy_loss_and_grads(pi, log_std, S, A, adv)
        opt_pi.step(gp + [gs])
        np.clip(log_std, LOG_STD_MIN, LOG_STD_MAX, out=log_std)
        for _ in range(V_ITERS):
            _, gv = value_loss_and_grads(vf, S, Gf)
            opt_v.step(gv)
        ep += BATCH
        c = log["curve"]
        c["episode"].append(ep); c["light"].append(float((ro["outcome"] == 1).mean()))
        c["pit"].append(float((ro["outcome"] == 2).mean())); c["ret"].append(float(ro["ret"].mean()))
    log["seconds"] = time.time() - t0
    with open(os.path.join(HERE, "log.json"), "w") as f:
        json.dump(log, f)
    print(f"done in {log['seconds']:.0f}s")


if __name__ == "__main__":
    if "--check" in sys.argv:
        print("gradcheck worst relative error", gradcheck())
    else:
        train()
