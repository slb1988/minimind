# 第 4 关配套脚本：让一句话流过模型，打印每一步的张量形状
# 用法（项目根目录）：python learn/l4_trace_shapes.py "你是谁"
import os, sys, torch
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # 无论从哪运行，都切到项目根目录
sys.path.insert(0, '.')
from transformers import AutoTokenizer
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM

text = sys.argv[1] if len(sys.argv) > 1 else "你是谁"
tok = AutoTokenizer.from_pretrained('./model')
model = MiniMindForCausalLM(MiniMindConfig())
model.load_state_dict(torch.load('./out/full_sft_768.pth', map_location='cpu'), strict=True)
model.eval()

prompt = tok.apply_chat_template([{"role": "user", "content": text}], tokenize=False, add_generation_prompt=True)
ids = tok(prompt, return_tensors='pt').input_ids
print(f"输入文本 -> {ids.shape[1]} 个 token：{[tok.decode([i]) for i in ids[0]]}\n")

def hook(name):
    def f(_m, _inp, out):
        out = out[0] if isinstance(out, tuple) else out
        print(f"{name:<34} 输出形状 {tuple(out.shape)}")
        if name.startswith("layer0 整个"): print("……layer1 ~ layer7 结构相同，省略……")
    return f

m = model.model
m.embed_tokens.register_forward_hook(hook("embed_tokens (查表)"))
blk = m.layers[0]
blk.input_layernorm.register_forward_hook(hook("layer0.input_layernorm (RMSNorm)"))
blk.self_attn.q_proj.register_forward_hook(hook("layer0.q_proj  (8 个头 x 96)"))
blk.self_attn.k_proj.register_forward_hook(hook("layer0.k_proj  (4 个头 x 96)"))
blk.self_attn.v_proj.register_forward_hook(hook("layer0.v_proj  (4 个头 x 96)"))
blk.self_attn.register_forward_hook(hook("layer0.self_attn (注意力输出)"))
blk.mlp.gate_proj.register_forward_hook(hook("layer0.mlp.gate_proj (升维)"))
blk.mlp.register_forward_hook(hook("layer0.mlp (FFN 输出)"))
blk.register_forward_hook(hook("layer0 整个 Block"))
m.norm.register_forward_hook(hook("最后的 norm"))
model.lm_head.register_forward_hook(hook("lm_head (每个位置对 6400 词打分)"))

with torch.no_grad():
    logits = model(ids).logits

probs = torch.softmax(logits[0, -1], -1)
top = torch.topk(probs, 5)
print("\n模型认为下一个 token 最可能是：")
for p, i in zip(top.values, top.indices):
    print(f"  {tok.decode([int(i)])!r:<10} {p.item():.1%}")

total = sum(p.numel() for p in model.parameters())
emb = m.embed_tokens.weight.numel()
per_layer = sum(p.numel() for p in blk.parameters())
attn = sum(p.numel() for p in blk.self_attn.parameters())
ffn = sum(p.numel() for p in blk.mlp.parameters())
print(f"\n参数量：总计 {total/1e6:.1f}M")
print(f"  词表 embedding（和 lm_head 共享）: {emb/1e6:.2f}M")
print(f"  每层 Block: {per_layer/1e6:.2f}M  = 注意力 {attn/1e6:.2f}M + FFN {ffn/1e6:.2f}M  (x8 层)")
