# 第 4 关 · 第一部分配套实验：RMSNorm / FFN / 残差
# 用法（项目根目录）：python learn/l4_part1_norm_ffn_residual.py
import os, sys, types, torch
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # 无论从哪运行，都切到项目根目录
sys.path.insert(0, '.')
from transformers import AutoTokenizer
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM

dev = 'cuda' if torch.cuda.is_available() else 'cpu'
tok = AutoTokenizer.from_pretrained('./model')
model = MiniMindForCausalLM(MiniMindConfig())
model.load_state_dict(torch.load('./out/full_sft_768.pth', map_location='cpu'), strict=True)
model.eval().to(dev)
layers = model.model.layers

prompt = tok.apply_chat_template([{"role": "user", "content": "你是谁开发的"}], tokenize=False, add_generation_prompt=True)
ids = tok(prompt, return_tensors='pt').input_ids.to(dev)
rms = lambda t: t.float().pow(2).mean(-1).sqrt().mean().item()

# ---------- 实验 1：RMSNorm 前后，向量的"大小"变成多少 ----------
print("=== 实验 1：RMSNorm 把每个 token 的 768 维向量缩放到 RMS≈1 ===")
cap = {}
for i, l in enumerate(layers):
    l.input_layernorm.register_forward_hook(lambda m, inp, out, i=i: cap.__setitem__(i, (inp[0], m.norm(inp[0].float()), out)))
    l.self_attn.register_forward_hook(lambda m, inp, out, i=i: cap.__setitem__(f'a{i}', out[0].clone()))
    l.mlp.register_forward_hook(lambda m, inp, out, i=i: cap.__setitem__(f'f{i}', out))
with torch.no_grad(): model(ids)
print(f"{'层':<4}{'进入时 RMS':>12}{'norm 后 RMS':>14}{'乘 weight 后 RMS':>18}")
for i in range(len(layers)):
    x, n, o = cap[i]
    print(f"{i:<4}{rms(x):>12.3f}{rms(n):>14.3f}{rms(o):>18.3f}")

print("\nweight 初始是全 1，训练后变成了：")
for i in [0, 3, 7]:
    w = layers[i].input_layernorm.weight.float()
    print(f"  layer{i}: 均值 {w.mean():.3f}  最小 {w.min():.3f}  最大 {w.max():.3f}")

# ---------- 实验 2：残差流——每层只是往里"加一点" ----------
print("\n=== 实验 2：每层往残差流里加的东西有多大 ===")
print(f"{'层':<4}{'残差流 RMS':>12}{'+ 注意力输出 RMS':>18}{'+ FFN 输出 RMS':>16}")
for i in range(len(layers)):
    print(f"{i:<4}{rms(cap[i][0]):>12.3f}{rms(cap[f'a{i}']):>18.3f}{rms(cap[f'f{i}']):>16.3f}")

# FFN 的门：silu(gate) 有多少接近 0（被"关掉"）
g = {}
layers[3].mlp.gate_proj.register_forward_hook(lambda m, inp, out: g.__setitem__('g', out))
with torch.no_grad(): model(ids)
s = torch.nn.functional.silu(g['g'].float())
print(f"\nlayer3 FFN 的 2432 个通道里，|silu(gate)| < 0.05（近似被门关掉）的比例：{(s.abs() < 0.05).float().mean():.1%}")

# ---------- 实验 3：删掉某一层（只保留残差直通），模型还会说话吗 ----------
print("\n=== 实验 3：跳过某一层，只走残差直通 ===")
def gen(n=40):
    with torch.no_grad():
        out = model.generate(ids, max_new_tokens=n, do_sample=False, use_cache=False)
    return tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True).replace('\n', ' ').strip()

print(f"完整模型      : {gen()}")
for k in range(len(layers)):
    orig = layers[k].forward
    layers[k].forward = types.MethodType(lambda self, h, *a, **kw: (h, None), layers[k])
    print(f"跳过 layer{k}  : {gen()}")
    layers[k].forward = orig
