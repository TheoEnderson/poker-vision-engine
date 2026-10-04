import cv2
import numpy as np

# 1. Carrega a imagem original
img = cv2.imread("foto_palco.jpg")
if img is None:
    print("[ERRO] Arquivo 'foto_palco.jpg' não encontrado na pasta atual!")
    exit(1)

h, w = img.shape[:2]

# 2. Recorte cirúrgico (foco na silhueta, ombros e no slide projetado)
# Elimina 100% do chão, cadeiras e fios
crop_y1 = int(h * 0.00)
crop_y2 = int(h * 0.58)
crop_x1 = int(w * 0.22)
crop_x2 = int(w * 0.82)

crop = img[crop_y1:crop_y2, crop_x1:crop_x2]

# Garante que seja um quadrado perfeito
ch, cw = crop.shape[:2]
min_dim = min(ch, cw)
start_x = (cw - min_dim) // 2
start_y = (ch - min_dim) // 2
square = crop[start_y:start_y + min_dim, start_x:start_x + min_dim]

# Redimensiona para resolução ideal de avatar do GitHub (500x500 px)
avatar_base = cv2.resize(square, (500, 500), interpolation=cv2.INTER_LANCZOS4)

# -------------------------------------------------------------
# VERSÃO 1: DARK TECH (Contraste alto, sombras fechadas, foco no slide)
# -------------------------------------------------------------
# Aumenta contraste e saturação para combinar com o tema dark
dark_tech = cv2.convertScaleAbs(avatar_base, alpha=1.35, beta=-30)

# Vinheta suave nas bordas para focar a visão no centro
X = np.linspace(-1, 1, 500)
Y = np.linspace(-1, 1, 500)
X, Y = np.meshgrid(X, Y)
vignette = 1 - 0.5 * (X**2 + Y**2)
vignette = np.clip(vignette, 0, 1)

for c in range(3):
    dark_tech[:, :, c] = (dark_tech[:, :, c] * vignette).astype(np.uint8)

cv2.imwrite("avatar_dark_tech.png", dark_tech)
print("[OK] Gerado: avatar_dark_tech.png")

# -------------------------------------------------------------
# VERSÃO 2: PRETO E BRANCO CINEMATOGRÁFICO
# -------------------------------------------------------------
gray = cv2.cvtColor(avatar_base, cv2.COLOR_BGR2GRAY)
# Equalização e boost de contraste P&B
bw_contraste = cv2.convertScaleAbs(gray, alpha=1.4, beta=-25)
bw_final = (bw_contraste * vignette).astype(np.uint8)

cv2.imwrite("avatar_bw_dramatico.png", bw_final)
print("[OK] Gerado: avatar_bw_dramatico.png")

print("\nImagens prontas! Basta abrir e escolher qual subir no GitHub.")
