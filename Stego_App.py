import streamlit as st
import numpy as np
from PIL import Image
import random, math, io
from skimage.metrics import structural_similarity as ssim

# ---------------- RC4 ----------------
def rc4_process(key, data):
    S = list(range(256))
    j = 0
    out = []
    for i in range(256):
        j = (j + S[i] + key[i % len(key)]) % 256
        S[i], S[j] = S[j], S[i]
    i = j = 0
    for byte in data:
        i = (i + 1) % 256
        j = (j + S[i]) % 256
        S[i], S[j] = S[j], S[i]
        K = S[(S[i] + S[j]) % 256]
        out.append(byte ^ K)
    return out

def rc4_encrypt(key, text):
    data = [ord(c) for c in text]
    return rc4_process(key, data)

def rc4_decrypt(key, data):
    decrypted_bytes = rc4_process(key, data)
    return ''.join(chr(b) for b in decrypted_bytes)

# ---------------- Pixel Shuffling ----------------
def shuffle_pixels(image_array, key):
    flat = image_array.reshape(-1, 3)
    shuffled = flat[key]
    return shuffled.reshape(image_array.shape)

def deshuffle_pixels(image_array, key):
    flat = image_array.reshape(-1, 3)
    deshuffled = np.zeros_like(flat)
    for i, k in enumerate(key):
        deshuffled[k] = flat[i]
    return deshuffled.reshape(image_array.shape)

def fitness(original, shuffled):
    hist, _ = np.histogram(shuffled.flatten(), bins=256, range=(0, 256), density=True)
    entropy = -np.sum(hist * np.log2(hist + 1e-9))
    corr = np.corrcoef(original.flatten(), shuffled.flatten())[0, 1]
    return entropy - abs(corr)

def cheetah_optimizer(image, iterations=10, population=3):
    flat_len = image.shape[0] * image.shape[1]
    candidates = [np.random.permutation(flat_len) for _ in range(population)]
    best_key, best_score = None, -1
    for _ in range(iterations):
        new_candidates = []
        for cand in candidates:
            shuffled = shuffle_pixels(image, cand)
            score = fitness(image, shuffled)
            if score > best_score:
                best_score, best_key = score, cand.copy()
            new_cand = cand.copy()
            i, j = np.random.randint(0, flat_len, 2)
            new_cand[i], new_cand[j] = new_cand[j], new_cand[i]
            new_candidates.append(new_cand)
        candidates = new_candidates
    return best_key

# ---------------- LSB Embedding & Extraction ----------------
def hash_embed(arr, data, secret_key):
    arr = arr.copy()
    flat = arr.reshape(-1, 3)
    bits = ''.join(f'{b:08b}' for b in data)
    idx = 0
    n = len(flat)
    for i in range(n):
        pos = (i * len(secret_key) + sum(secret_key)) % n
        if idx + 8 <= len(bits):
            px = flat[pos]
            r_bits = bits[idx:idx + 3]
            g_bits = bits[idx + 3:idx + 6]
            b_bits = bits[idx + 6:idx + 8]
            idx += 8
            px[0] = (px[0] & 0b11111000) | int(r_bits, 2)
            px[1] = (px[1] & 0b11111000) | int(g_bits, 2)
            px[2] = (px[2] & 0b11111100) | int(b_bits, 2)
        else:
            break
    return arr

def hash_extract(arr, secret_key, max_bytes=10000):
    flat = arr.reshape(-1, 3)
    n = len(flat)
    bits = ""
    for i in range(n):
        pos = (i * len(secret_key) + sum(secret_key)) % n
        px = flat[pos]
        bits += f'{px[0] & 0b111:03b}'
        bits += f'{px[1] & 0b111:03b}'
        bits += f'{px[2] & 0b11:02b}'
    secret_bytes = [int(bits[i:i + 8], 2) for i in range(0, max_bytes * 8, 8)]
    return secret_bytes

# ---------------- Metrics ----------------
def mse(img1, img2):
    return np.mean((img1.astype("float") - img2.astype("float")) ** 2)

def psnr(img1, img2):
    mse_val = mse(img1, img2)
    if mse_val == 0:
        return float('inf')
    return 10 * math.log10((255 ** 2) / mse_val)

def ssim_metric(img1, img2):
    ssim_val = 0
    for i in range(3):
        ssim_val += ssim(img1[..., i], img2[..., i], data_range=255)
    return ssim_val / 3

# ---------------- Streamlit UI ----------------
st.set_page_config(page_title="Image Steganography", layout="wide")
st.title("A Secure Data Hiding System Using Dynamic Pixel Shuffling, Hash-Based LSB Steganography, and RC4 Encryption")

tabs = st.tabs(["🔒 Encryption", "🔓 Decryption"])

# -------- Encryption Tab --------
with tabs[0]:
    st.header("Encryption & Embedding")

    col1, col2 = st.columns(2)
    with col1:
        uploaded_image = st.file_uploader("Upload Cover Image", type=["png", "jpg", "jpeg"])
        secret_key = st.text_input("Enter Secret Key")
    with col2:
        text_file = st.file_uploader("Upload Secret Text File (.txt)", type=["txt"])

    if st.button("🔐 Encrypt & Embed"):
        if uploaded_image and text_file and secret_key:
            image = Image.open(uploaded_image).convert("RGB")
            cover_arr = np.array(image)
            secret_text = text_file.read().decode("utf-8").strip()
            payload_text = str(len(secret_text)) + secret_text
            key = [ord(c) for c in secret_key]

            st.info("Running Cheetah Optimizer...")
            best_key = cheetah_optimizer(cover_arr)
            shuffled_arr = shuffle_pixels(cover_arr, best_key)

            encrypted = rc4_encrypt(key, payload_text)
            stego_shuffled = hash_embed(shuffled_arr, encrypted, key)
            stego_final = deshuffle_pixels(stego_shuffled, best_key)

            mse_value = mse(cover_arr, stego_final)
            psnr_value = psnr(cover_arr, stego_final)
            ssim_value = ssim_metric(cover_arr, stego_final)

            # ---- Store everything in session_state ----
            st.session_state["cover_arr"] = cover_arr
            st.session_state["stego_final"] = stego_final
            st.session_state["mse_value"] = mse_value
            st.session_state["psnr_value"] = psnr_value
            st.session_state["ssim_value"] = ssim_value

            from io import BytesIO
            stego_buffer = BytesIO()
            Image.fromarray(stego_final).save(stego_buffer, format="PNG")
            stego_buffer.seek(0)

            key_buffer = BytesIO()
            np.save(key_buffer, best_key)
            key_buffer.seek(0)

            st.session_state["stego_buffer"] = stego_buffer
            st.session_state["key_buffer"] = key_buffer

            st.success("✅ Encryption Complete! Scroll below for outputs.")
        else:
            st.error("Please upload image, text file, and enter key.")

    # --------- Display results if available ---------
    if "stego_final" in st.session_state:
        col1, col2 = st.columns([1, 1], gap="large")

        with col1:
            st.image(st.session_state["cover_arr"], caption="Original Image", width=300)

        with col2:
            st.image(st.session_state["stego_final"], caption="Stego Image", width=300)

        st.subheader("📊 Verification Metrics")
        c1, c2, c3 = st.columns(3)
        c1.metric("MSE", f"{st.session_state['mse_value']:.4f}")
        c2.metric("PSNR", f"{st.session_state['psnr_value']:.4f} dB")
        c3.metric("SSIM", f"{st.session_state['ssim_value']:.4f}")

        st.download_button(
            "⬇️ Download Stego Image",
            data=st.session_state["stego_buffer"],
            file_name="stego_output.png",
            mime="image/png",
        )
        st.download_button(
            "⬇️ Download Shuffle Key",
            data=st.session_state["key_buffer"],
            file_name="shuffle_key.npy",
            mime="application/octet-stream",
        )


# -------- Decryption Tab --------
with tabs[1]:
    st.header("Decryption & Extraction")

    col1, col2 = st.columns(2)
    with col1:
        stego_file = st.file_uploader("Upload Stego Image", type=["png", "jpg", "jpeg"])
        shuffle_key_file = st.file_uploader("Upload Shuffle Key (.npy)", type=["npy"])
    with col2:
        dec_key = st.text_input("Enter Secret Key for Decryption")

    if st.button("🔓 Decrypt"):
        if stego_file and shuffle_key_file and dec_key:
            stego_img = Image.open(stego_file).convert("RGB")
            stego_arr = np.array(stego_img)
            best_key = np.load(shuffle_key_file)
            key = [ord(c) for c in dec_key]

            reshuffled = shuffle_pixels(stego_arr, best_key)
            extracted = hash_extract(reshuffled, key)
            decrypted = rc4_decrypt(key, extracted)

            digits = ""
            for ch in decrypted:
                if ch.isdigit():
                    digits += ch
                else:
                    break

            if digits == "":
                st.error("Wrong key or corrupted data!")
            else:
                length = int(digits)
                start_idx = len(digits)
                secret_message = decrypted[start_idx:start_idx + length]
                st.success("✅ Secret Message Recovered Successfully!")
                st.text_area("Recovered Secret Message:", secret_message, height=300)
        else:
            st.error("Please upload stego image, shuffle key, and enter decryption key.")