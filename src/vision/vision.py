"""
src/vision/vision.py
Módulo de captura de tela de alta performance para Linux (Wayland e X11) e Windows.
"""

import os
import subprocess
from pathlib import Path
from typing import Optional, Union
import cv2
import mss
import numpy as np

from src.config import PROJECT_ROOT, SAMPLES_DIR


def grab_screen() -> np.ndarray:
    """
    Captura a tela do monitor primário com suporte completo a ambientes Linux (Wayland e X11/Xorg) e Windows.
    
    - No Wayland: Executa o utilitário 'grim' para capturar em memória compartilhada (/dev/shm/poker_frame.png).
    - No X11 / Xorg / Windows: Utiliza a biblioteca 'mss' capturando os pixels diretamente na memória RAM sem I/O de disco.
    
    Retorna:
        numpy.ndarray da imagem capturada em formato BGR do OpenCV.
    """
    session_type = os.environ.get("XDG_SESSION_TYPE", "").lower()
    wayland_display = os.environ.get("WAYLAND_DISPLAY", "")

    is_wayland = (session_type == "wayland") or bool(wayland_display)

    if is_wayland:
        shm_path = "/dev/shm/poker_frame.png"
        try:
            res = subprocess.run(["grim", shm_path], capture_output=True, timeout=2.0)
            if res.returncode == 0 and os.path.exists(shm_path):
                frame = cv2.imread(shm_path)
                if frame is not None:
                    return frame
        except Exception:
            # Fallback para mss caso o comando grim falhe
            pass

    # Modo X11 / Xorg ou fallback de alto desempenho via mss
    mss_cls = getattr(mss, "MSS", mss.mss)
    with mss_cls() as sct:
        # Monitor 1 é a tela primária (Monitor 0 é a área virtual unificada)
        monitor = sct.monitors[1] if len(sct.monitors) > 1 else sct.monitors[0]
        sct_img = sct.grab(monitor)
        frame = np.array(sct_img)
        # Converte de BGRA (padrão mss) para BGR (padrão OpenCV)
        return cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)


def select_and_save_board_roi(sample_name: str = "Captura de tela de 2026-09-06 14-44-59.png"):
    """Utilitário interativo de calibração para seleção de ROI do board."""
    image_path = SAMPLES_DIR / sample_name
    if not image_path.exists():
        image_path = PROJECT_ROOT / sample_name

    print(f"Carregando imagem: {image_path}")
    image = cv2.imread(str(image_path))

    if image is None:
        print(f"\n[ERRO] Não foi possível carregar a imagem '{sample_name}'.")
        return

    print("\n[INFO] Imagem carregada com sucesso!")
    print(" -> Desenhe o retângulo com o mouse em volta das cartas comunitárias.")
    print(" -> Pressione ENTER ou ESPAÇO para confirmar a seleção.")
    print(" -> Pressione 'c' para cancelar a seleção.")

    roi_window = "Selecione as Cartas Comunitarias (ENTER/ESPACO confirma, c cancela)"
    cv2.namedWindow(roi_window, cv2.WINDOW_NORMAL)
    
    x, y, w, h = cv2.selectROI(roi_window, image, fromCenter=False, showCrosshair=True)
    cv2.destroyWindow(roi_window)

    if w > 0 and h > 0:
        coords = {
            'top': int(y),
            'left': int(x),
            'width': int(w),
            'height': int(h)
        }

        output_path = PROJECT_ROOT / "board_template.png"
        cropped_board = image[y : y + h, x : x + w]
        cv2.imwrite(str(output_path), cropped_board)

        print("\n" + "=" * 50)
        print("Coordenadas exatas do recorte:")
        print(coords)
        print(f"Recorte salvo com sucesso como '{output_path}'.")
        print("=" * 50 + "\n")
    else:
        print("\n[AVISO] Seleção cancelada ou área inválida (largura ou altura zerada).\n")

    cv2.destroyAllWindows()


if __name__ == "__main__":
    print("\nTestando captura de tela com grab_screen()...")
    frame = grab_screen()
    print(f"Captura realizada com sucesso! Formato: {frame.shape}")
