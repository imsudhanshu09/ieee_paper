
import argparse, os, math, json, random
from typing import Tuple
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm

VARS = [
    "DHI","DNI","GHI","Wind_speed","Humidity","Temperature",
    "PV_production","Wind_production","Electric_demand"
]

def set_seed(seed: int = 42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def load_and_prepare(csv_path: str, resample_to_hour: bool=True) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if "Unnamed: 0" in df.columns: df = df.drop(columns=["Unnamed: 0"])
    df["Time"] = pd.to_datetime(df["Time"].str.replace("T"," "))
    df = df.sort_values("Time").set_index("Time")
    df = df[VARS].copy()
    if resample_to_hour:
        df = df.resample("1h").mean().interpolate()
    df = df.ffill().bfill()
    return df

class SeqDataset(Dataset):
    def __init__(self, data: np.ndarray, context_len: int, horizon_len: int, stride: int = 1):
        self.N = data.shape[1]
        self.T = data.shape[0]
        self.context_len = context_len
        self.horizon_len = horizon_len
        self.data = data.astype(np.float32)
        self.idxs = list(range(0, self.T - (context_len + horizon_len), stride))

    def __len__(self): return len(self.idxs)
    def __getitem__(self, i):
        s = self.idxs[i]; e = s + self.context_len; f = e + self.horizon_len
        ctx = self.data[s:e].T
        fut = self.data[e:f].T
        return torch.from_numpy(ctx), torch.from_numpy(fut)

# ---- Graph blocks ----

class GATLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.W = nn.Linear(in_dim, out_dim, bias=False)
        self.a = nn.Linear(2*out_dim, 1, bias=False)

    def forward(self, X): # (N,F)
        H = self.W(X); N = H.size(0)
        Hi = H.unsqueeze(1).repeat(1,N,1)
        Hj = H.unsqueeze(0).repeat(N,1,1)
        e = self.a(torch.cat([Hi,Hj], dim=-1)).squeeze(-1)
        A = F.softmax(F.leaky_relu(e,0.2), dim=-1)
        out = A @ H
        return out, A

class AdaptiveGraph(nn.Module):
    def __init__(self, in_dim, hidden=None):
        super().__init__()
        self.hidden = in_dim if hidden is None else hidden
        self.proj = nn.Linear(in_dim, self.hidden, bias=False)
    def forward(self, X): # (N,F)
        Z = self.proj(X) # (N,H)
        A = torch.relu(Z @ Z.T)
        A = A / (A.sum(dim=-1, keepdim=True) + 1e-6)
        out = A @ Z
        return out, A

# ---- Temporal conv ----

class DilatedTemporalBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, dilation=1):
        super().__init__()
        padding = (kernel_size-1)*dilation
        self.conv = nn.Conv1d(in_ch,out_ch,kernel_size,padding=padding,dilation=dilation)
        self.proj = nn.Conv1d(out_ch,out_ch,1)
        self.act = nn.ReLU()
    def forward(self,x):
        y = self.conv(x)
        trim = self.conv.padding[0]
        y = y[:,:,:-trim] if trim>0 else y
        y = self.act(y)
        return self.proj(y)

# ---- GNE variants ----

class GNE(nn.Module):
    def __init__(self, N, T, ht_dim=64, layers=4, mode="gat"):
        super().__init__()
        self.N=N; self.T=T; self.mode=mode
        self.temporal = nn.ModuleList([DilatedTemporalBlock(N,N,dilation=2**i) for i in range(layers)])
        if mode=="gat":
            self.graph = GATLayer(T,T)
        elif mode=="adaptive":
            self.graph = AdaptiveGraph(T,hidden=32)
        elif mode=="hybrid":
            self.gat = GATLayer(T,T)
            self.adapt = AdaptiveGraph(T, hidden=T)
        self.out = nn.Conv1d(N,N,1)
        self.ht_proj = nn.Linear(ht_dim, N)

    def forward(self, Xk, ht):
        B,N,T = Xk.shape
        y = Xk
        for block in self.temporal: y = y + block(y)
        outs, adjs = [], []
        for b in range(B):
            node_feats = y[b] # (N,T)
            if self.mode in ["gat","adaptive"]:
                H, A = self.graph(node_feats)
            else:
                Hg, Ag = self.gat(node_feats)
                Ha, Aa = self.adapt(node_feats)
                H = 0.5*(Hg+Ha); A = 0.5*(Ag+Aa)
            outs.append(H.unsqueeze(0)); adjs.append(A.unsqueeze(0))
        Y = torch.cat(outs,dim=0) # (B,N,T)
        cond = self.ht_proj(ht).unsqueeze(-1) # (B,N,1)
        Y = Y + cond
        return self.out(Y), torch.cat(adjs,dim=0)

# ---- DDPM Core with cosine beta + DDIM ----

def cosine_beta_schedule(T, s=0.008):
    steps = T
    t = torch.linspace(0, T, steps+1)
    f = torch.cos(((t/T + s)/(1+s)) * math.pi/2) ** 2
    alphas_bar = f/f[0]
    betas = 1 - (alphas_bar[1:]/alphas_bar[:-1])
    return torch.clip(betas, 1e-6, 0.999)

class GRUContext(nn.Module):
    def __init__(self, N, hidden=64):
        super().__init__()
        self.gru = nn.GRU(input_size=N, hidden_size=hidden, num_layers=1, batch_first=True)
    def forward(self, ctx): # (B,N,Tc)
        x = ctx.transpose(1,2)
        _, h = self.gru(x)
        return h.squeeze(0)

class Diffusion(nn.Module):

    def __init__(self, N, Tc, Tf, steps=100, graph_mode="gat", device="cpu"):
        super().__init__()
        self.N=N; self.Tf=Tf; self.K=steps; self.device=device
        betas = cosine_beta_schedule(steps)
        self.register_buffer("betas", betas)
        alphas = 1.0 - betas
        self.register_buffer("alphas", alphas)
        alphas_bar = torch.cumprod(alphas, dim=0)
        self.register_buffer("alphas_bar", alphas_bar)
        self.ht = GRUContext(N, hidden=64)
        self.gne = GNE(N, Tf, ht_dim=64, layers=4, mode=graph_mode)

    def q_sample(self, x0, k, noise=None):
        if noise is None: noise = torch.randn_like(x0)
        a_bar = self.alphas_bar[k].view(-1,1,1)
        return torch.sqrt(a_bar)*x0 + torch.sqrt(1-a_bar)*noise, noise

    def predict_eps(self, xk, ht, k):
        eps_hat, _ = self.gne(xk, ht)
        return eps_hat

    def forward(self, ctx, fut):
        B = ctx.size(0)
        ht = self.ht(ctx)
        k = torch.randint(0, self.K, (B,), device=ctx.device)
        xk, noise = self.q_sample(fut, k)
        eps_hat = self.predict_eps(xk, ht, k)
        return F.mse_loss(eps_hat, noise)

    @torch.no_grad()
    def sample_ddim(self, ctx, eta=0.0, n_steps=None):
        """ DDIM sampling; if eta=0 it's deterministic. """
        B = ctx.size(0)
        ht = self.ht(ctx)
        x = torch.randn(B, self.N, self.Tf, device=ctx.device)
        K = self.K if n_steps is None else n_steps
        idxs = torch.linspace(0, self.K-1, K).long().flip(0).to(ctx.device)
        for i, k in enumerate(idxs):
            a_bar = self.alphas_bar[k]
            eps = self.predict_eps(x, ht, k.expand(B))
            x0_hat = (x - torch.sqrt(1-a_bar)*eps) / torch.sqrt(a_bar)
            if i == K-1: x = x0_hat; break
            a_bar_prev = self.alphas_bar[idxs[i+1]]
            sigma = eta * torch.sqrt((1-a_bar_prev)/(1-a_bar)) * torch.sqrt(1-a_bar/a_bar_prev)
            dir_xt = torch.sqrt(a_bar_prev) * (x0_hat)
            noise = sigma * torch.randn_like(x)
            x = dir_xt + noise
        return x

# ---- EMA wrapper ----

class EMA:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {k: v.clone().detach() for k,v in model.state_dict().items()}
    def update(self, model):
        for k, v in model.state_dict().items():
            self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1.0-self.decay)
    def copy_to(self, model):
        model.load_state_dict(self.shadow, strict=False)

# ---- Metrics ----

def mae(y, yhat): return torch.mean(torch.abs(y - yhat)).item()
def rmse(y, yhat): return torch.sqrt(torch.mean((y - yhat)**2)).item()
def crps_empirical(samples, y_true):
    S = samples.size(0)
    y = y_true.unsqueeze(0).repeat(S,1,1,1)
    term1 = torch.mean(torch.abs(samples - y))
    term2 = 0.5*torch.mean(torch.abs(samples.unsqueeze(1) - samples.unsqueeze(0)))
    return (term1 - term2).item()

# ---- Train/Eval ----

def train(model, train_loader, val_loader, epochs, device, use_ema=True):
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=2)
    ema = EMA(model, decay=0.999) if use_ema else None
    best = 1e9; patience=5; bad=0
    for ep in range(epochs):
        model.train(); losses=[]
        for ctx, fut in tqdm(train_loader, desc=f"Epoch {ep+1}/{epochs}"):
            ctx, fut = ctx.to(device), fut.to(device)
            loss = model(ctx, fut)
            opt.zero_grad(); loss.backward(); opt.step()
            if ema: ema.update(model)
            losses.append(loss.item())
        model.eval()
        with torch.no_grad():
            vlosses=[]
            for ctx, fut in val_loader:
                ctx, fut = ctx.to(device), fut.to(device)
                vlosses.append(model(ctx,fut).item())
        v = float(np.mean(vlosses))
        sched.step(v)
        print(f"Epoch {ep+1}: train={np.mean(losses):.4f} val={v:.4f}")
        if v < best: best=v; bad=0
        else: bad+=1
        if bad>=patience: print("Early stopping."); break
    if ema: ema.copy_to(model)
    return model

@torch.no_grad()
@torch.no_grad()
def eval_model(model, loader, device, samples=5, use_ddim=True, max_batches=50):
    model.eval()
    maes=[]; rmses=[]; allS=[]; allY=[]
    for i, (ctx, fut) in enumerate(tqdm(loader, desc="Evaluating", ncols=100)):
        if i >= max_batches:   # stop early to keep eval quick
            break
        ctx, fut = ctx.to(device), fut.to(device)
        Ss=[]
        for _ in range(samples):
            if use_ddim:
                yhat = model.sample_ddim(ctx, eta=0.0, n_steps=10)
            else:
                yhat = model.sample_ddim(ctx, eta=1.0, n_steps=10)
            Ss.append(yhat.unsqueeze(0))
        S = torch.cat(Ss, dim=0)
        mean_pred = S.mean(0)
        maes.append(mae(fut, mean_pred)); rmses.append(rmse(fut, mean_pred))
        allS.append(S.cpu()); allY.append(fut.cpu())
    S = torch.cat(allS, dim=1); Y = torch.cat(allY, dim=0)
    c = crps_empirical(S, Y)
    return float(np.mean(maes)), float(np.mean(rmses)), c, S, Y

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=str, required=True)
    ap.add_argument("--graph_mode", type=str, default="hybrid", choices=["gat","adaptive","hybrid"])
    ap.add_argument("--context_len", type=int, default=24)  # 24 hours (hourly)
    ap.add_argument("--horizon_len", type=int, default=24) # 1 day ahead
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--diffusion_steps", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    set_seed(args.seed)
    os.makedirs("outputs_plus", exist_ok=True)

    df = load_and_prepare(args.csv, resample_to_hour=True)
    mu = df.mean(0); std = df.std(0).replace(0,1.0)
    df_norm = (df - mu)/std
    data = df_norm.values # (T,N)
    ds = SeqDataset(data, context_len=args.context_len, horizon_len=args.horizon_len, stride=1)
    n_total = len(ds)
    n_train = int(0.6 * n_total)
    n_val = int(0.2 * n_total)
    n_test = max(1, n_total - n_train - n_val)  # ensure at least 1 test sample

    train_set, val_set, test_set = random_split(
        ds, [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(args.seed)
    )

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = Diffusion(N=len(VARS), Tc=args.context_len, Tf=args.horizon_len, steps=args.diffusion_steps,
                      graph_mode=args.graph_mode, device=device).to(device)

    model = train(model, train_loader, val_loader, args.epochs, device, use_ema=True)
    m = r = c = 0.0
    S = Y = torch.zeros(1)
    print(f"\n🔍 Total samples: {len(ds)}, Train: {len(train_set)}, Val: {len(val_set)}, Test: {len(test_set)}\n")

    if len(test_loader) > 0:
        print("\n🚀 Running evaluation on test set...\n")
        m, r, c, S, Y = eval_model(model, test_loader, device, samples=5, use_ddim=True, max_batches=50)
        print(f"\n📊 Evaluation Results — MAE: {m:.4f}, RMSE: {r:.4f}, CRPS: {c:.4f}\n")
    else:
        print("\n⚠️ No test data found — skipping evaluation.\n")

    # ---- Save results safely ----
    metrics = {
        "MAE": float(m),
        "RMSE": float(r),
        "CRPS_approx": float(c),
        "graph_mode": args.graph_mode
    }

    os.makedirs("outputs_plus", exist_ok=True)

    # Save metrics JSON
    with open("outputs_plus/metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    print(f"\n✅ Metrics saved: {metrics}\n")


    # Save predictions
    torch.save({"samples": S[:5], "truth": Y[:32]}, "outputs_plus/preds.pt")

    # Save normalization stats
    mu.to_json("outputs_plus/mu.json")
    std.to_json("outputs_plus/std.json")

    print("\n✅ Saved outputs to: outputs_plus/metrics.json, preds.pt, mu/std.json\n")


if __name__=="__main__":
    main()
