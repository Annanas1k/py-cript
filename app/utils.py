# utils.py

from typing import List

def bin_to_hex(binary_str: str) -> str:
    """Convertește un șir binar în șir hexazecimal. Padding automat."""
    if not binary_str: return ""
    return f"{int(binary_str, 2):0{len(binary_str) // 4}X}"


def hex_to_bin(hex_str: str, pad_to=None) -> str:
    """Convertește un șir hexazecimal în șir binar."""
    if not hex_str: return ""
    length = pad_to if pad_to else len(hex_str) * 4
    return bin(int(hex_str, 16))[2:].zfill(length)


def xor_bits(bin1: str, bin2: str) -> str:
    """Aplică XOR pe doi șiruri binare de aceeași lungime."""
    # Atenție: Acum este folosit și în DES. Păstrează doar implementarea simplă:
    return ''.join(['1' if a != b else '0' for a, b in zip(bin1, bin2)])


def permute(data: str, table: list) -> str:
    """Aplică o permutare dată de o listă de indici (table) pe șirul de date (data)."""
    return "".join(data[i - 1] for i in table)


def split_into_blocks(data_hex: str, block_size_hex=16, padding=True) -> List[str]:
    """Împarte datele hex în blocuri de 16 caractere (64 biți). Aplică zero-padding."""
    if padding:
        if len(data_hex) % block_size_hex != 0:
            data_hex += '0' * (block_size_hex - (len(data_hex) % block_size_hex))
    return [data_hex[i:i + block_size_hex] for i in range(0, len(data_hex), block_size_hex)]


def count_bit_differences(bin1: str, bin2: str) -> int:
    """Numără biții diferiți (distanța Hamming)."""
    return xor_bits(bin1, bin2).count('1')