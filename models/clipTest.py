# test_mobileclip_dims.py
import torch
import open_clip
from PIL import Image
import numpy as np

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # --- load model + preprocess + tokenizer (same as your class) ---
    model, _, preprocess = open_clip.create_model_and_transforms(
        'MobileCLIP2-S0', pretrained='dfndr2b'
    )
    tokenizer = open_clip.get_tokenizer('MobileCLIP2-S0')
    model = model.to(device).eval()

    # --- random image: make a PIL image so preprocess works ---
    # preprocess expects PIL, not a tensor
    rand_np = (np.random.rand(256, 256, 3) * 255).astype(np.uint8)
    rand_pil = Image.fromarray(rand_np)
    img_tensor = preprocess(rand_pil).unsqueeze(0).to(device)   # (1,3,H,W)

    print("image tensor shape:", img_tensor.shape)
    print("image tensor range: min=%.3f max=%.3f" % (img_tensor.min(), img_tensor.max()))

    # --- text tokens ---
    tokens = tokenizer(["a photo of a cat", "a photo of a dog"]).to(device)
    print("token shape:", tokens.shape)

    with torch.no_grad():
        # --- image embedding ---
        img_emb = model.encode_image(img_tensor)
        print("raw image emb shape:", img_emb.shape)
        img_emb = img_emb / img_emb.norm(dim=-1, keepdim=True)

        # --- text embedding ---
        txt_emb = model.encode_text(tokens)
        print("raw text emb shape:", txt_emb.shape)
        txt_emb = txt_emb / txt_emb.norm(dim=-1, keepdim=True)

    # --- dimension checks ---
    D_img = img_emb.shape[-1]
    D_txt = txt_emb.shape[-1]
    print(f"\nimage emb dim = {D_img}")
    print(f"text  emb dim = {D_txt}")
    assert D_img == D_txt, "image and text dims differ — not the same space!"
    print(f"-> same space, dim = {D_img}")
    print(f"-> concat of 3 embeddings = {3 * D_img} (this must be your projector's in_features)")

    # --- sanity: normalized vectors have norm 1 ---
    print("\nimg emb norm:", img_emb.norm(dim=-1))
    print("txt emb norm:", txt_emb.norm(dim=-1))

    # --- sanity: similarity is a valid cosine in [-1,1] ---
    sim = img_emb @ txt_emb.T
    print("\ncosine similarity matrix (1 image x 2 texts):")
    print(sim)

    # --- extra: test the exact height-map path you plan to use ---
    # replicate a 1-channel map to 3 channels, normalize to [0,1], resize to 256
    h = torch.rand(1, 1, 128, 128, device=device)          # random height map
    h = (h - h.amin()) / (h.amax() - h.amin() + 1e-6)      # [0,1]
    h3 = h.repeat(1, 3, 1, 1)                              # (1,3,128,128)
    h3 = torch.nn.functional.interpolate(
        h3, size=(256, 256), mode='bilinear', align_corners=False
    )
    with torch.no_grad():
        h_emb = model.encode_image(h3)
        h_emb = h_emb / h_emb.norm(dim=-1, keepdim=True)
    print("\nheight-map emb shape:", h_emb.shape, "dim =", h_emb.shape[-1])
    assert h_emb.shape[-1] == D_img

    print("\nAll dimension checks passed.")

if __name__ == "__main__":
    main()