"""
src/vision/cards.py
Utilitário de segmentação de slots do board em cartas individuais.
"""

import os
from pathlib import Path
from typing import Union
import cv2

from src.config import CARDS_DIR, PROJECT_ROOT


def process_board_cards(
    image_filename: Union[str, Path] = "board_template.png",
    output_dir: Union[str, Path] = CARDS_DIR,
    num_slots: int = 5
):
    target_dir = Path(output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    img_path = Path(image_filename)
    if not img_path.is_absolute():
        img_path = PROJECT_ROOT / image_filename

    print(f"Carregando imagem: {img_path}")
    image = cv2.imread(str(img_path))

    if image is None:
        print(f"[ERRO] Não foi possível carregar '{img_path}'. Verifique se o arquivo existe.")
        return

    height, width = image.shape[:2]
    print(f"Dimensões detectadas: {width}x{height} (largura x altura)")
    print(f"Diretório de saída: {target_dir}")

    debug_image = image.copy()
    slot_details = []

    print("\n" + "=" * 60)
    print("Processando e salvando os slots individuais:")
    print("-" * 60)

    for i in range(num_slots):
        x_start = int(round(i * width / float(num_slots)))
        x_end = int(round((i + 1) * width / float(num_slots)))
        slot_width = x_end - x_start

        slot_crop = image[0:height, x_start:x_end]
        slot_filename = f"slot_{i + 1}.png"
        slot_path = target_dir / slot_filename
        cv2.imwrite(str(slot_path), slot_crop)

        slot_details.append({
            "slot": i + 1,
            "x_start": x_start,
            "x_end": x_end,
            "width": slot_width,
            "height": height,
            "file": slot_filename
        })

        cv2.rectangle(debug_image, (x_start, 0), (x_end - 1, height - 1), (0, 255, 0), 2)
        print(f"Slot {i + 1}: x_start={x_start:3d}, x_end={x_end:3d} (largura: {slot_width}px, altura: {height}px) -> {slot_filename}")

    debug_path = target_dir / "debug_slots.png"
    cv2.imwrite(str(debug_path), debug_image)

    print("-" * 60)
    print(f"Imagem de depuração gerada em: {debug_path}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    process_board_cards()
