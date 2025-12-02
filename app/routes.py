from flask import Blueprint, render_template, request

bp = Blueprint('main', __name__)

@bp.route('/')
def home():
    return render_template('home.html')
def caesar_encrypt(text, key):
    result = ""
    for c in text:
        if c.isalpha():
            shift = key % 26
            if c.isupper():
                result += chr((ord(c) - 65 + shift) % 26 + 65)
            else:
                result += chr((ord(c) - 97 + shift) % 26 + 97)
        else:
            result += c
    return result

def caesar_decrypt(text, key):
    return caesar_encrypt(text, -key)

@bp.route("/caesar", methods=["GET", "POST"])
def caesar():
    encrypted = decrypted = ""
    plaintext = ciphertext = ""
    key = key2 = ""

    if request.method == "POST":
        action = request.form.get("action")

        if action == "encrypt":
            plaintext = request.form.get("plaintext", "")
            key_str = request.form.get("key", "0")
            try:
                key = int(key_str)
                encrypted = caesar_encrypt(plaintext, key)
            except ValueError:
                encrypted = "Cheia trebuie sa fie un numar intreg!"

        elif action == "decrypt":
            ciphertext = request.form.get("ciphertext", "")
            key_str = request.form.get("key2", "0")
            try:
                key2 = int(key_str)
                decrypted = caesar_decrypt(ciphertext, key2)
            except ValueError:
                decrypted = "Cheia trebuie sa fie un numar intreg!"

    return render_template("caesar.html",
                           encrypted=encrypted, decrypted=decrypted,
                           plaintext=plaintext, ciphertext=ciphertext,
                           key=key, key2=key2)


# ====== Algoritmul Playfair ======

def generate_key_matrix(key: str):
    key = key.upper().replace("J", "I")
    matrix = []
    used = set()

    for c in key:
        if c.isalpha() and c not in used:
            used.add(c)
            matrix.append(c)

    for c in "ABCDEFGHIKLMNOPQRSTUVWXYZ":  # fără J
        if c not in used:
            used.add(c)
            matrix.append(c)

    return [matrix[i*5:(i+1)*5] for i in range(5)]


def find_position(matrix, letter):
    for i, row in enumerate(matrix):
        for j, val in enumerate(row):
            if val == letter:
                return i, j
    return None


def prepare_text(text):
    text = text.upper().replace("J", "I")
    text = "".join([c for c in text if c.isalpha()])
    prepared = []
    i = 0
    while i < len(text):
        a = text[i]
        b = text[i+1] if i+1 < len(text) else "X"
        if a == b:
            prepared.append(a + "X")
            i += 1
        else:
            prepared.append(a + b)
            i += 2
    return prepared


def playfair_encrypt(plaintext, key):
    matrix = generate_key_matrix(key)
    digraphs = prepare_text(plaintext)
    encrypted = ""

    for pair in digraphs:
        a, b = pair
        row1, col1 = find_position(matrix, a)
        row2, col2 = find_position(matrix, b)

        if row1 == row2:  # aceeași linie
            encrypted += matrix[row1][(col1 + 1) % 5]
            encrypted += matrix[row2][(col2 + 1) % 5]
        elif col1 == col2:  # aceeași coloană
            encrypted += matrix[(row1 + 1) % 5][col1]
            encrypted += matrix[(row2 + 1) % 5][col2]
        else:  # dreptunghi
            encrypted += matrix[row1][col2]
            encrypted += matrix[row2][col1]

    return encrypted


def playfair_decrypt(ciphertext, key):
    matrix = generate_key_matrix(key)
    digraphs = [ciphertext[i:i+2] for i in range(0, len(ciphertext), 2)]
    decrypted = ""

    for pair in digraphs:
        a, b = pair
        row1, col1 = find_position(matrix, a)
        row2, col2 = find_position(matrix, b)

        if row1 == row2:  # aceeași linie
            decrypted += matrix[row1][(col1 - 1) % 5]
            decrypted += matrix[row2][(col2 - 1) % 5]
        elif col1 == col2:  # aceeași coloană
            decrypted += matrix[(row1 - 1) % 5][col1]
            decrypted += matrix[(row2 - 1) % 5][col2]
        else:  # dreptunghi
            decrypted += matrix[row1][col2]
            decrypted += matrix[row2][col1]

    return decrypted


@bp.route("/playfair", methods=["GET", "POST"])
def playfair():
    encrypted = None
    decrypted = None
    plaintext = ""
    ciphertext = ""
    key = ""
    matrix = None

    if request.method == "POST":
        action = request.form.get("action")

        if action == "encrypt":
            plaintext = request.form.get("plaintext", "")
            key = request.form.get("key", "")
            encrypted = playfair_encrypt(plaintext, key)
            matrix = generate_key_matrix(key)

        elif action == "decrypt":
            ciphertext = request.form.get("ciphertext", "")
            key = request.form.get("key2", "")
            decrypted = playfair_decrypt(ciphertext, key)
            matrix = generate_key_matrix(key)

    return render_template("playfair.html",
                           plaintext=plaintext,
                           ciphertext=ciphertext,
                           key=key,
                           encrypted=encrypted,
                           decrypted=decrypted,
                           matrix=matrix)




# ====== Algoritmul Vigenere ======

def vigenere_encrypt(plaintext, key):
    plaintext = plaintext.upper()
    key = key.upper()
    ciphertext = ""
    key_index = 0

    for char in plaintext:
        if char.isalpha():
            shift = ord(key[key_index % len(key)]) - ord('A')
            encrypted_char = chr((ord(char) - ord('A') + shift) % 26 + ord('A'))
            ciphertext += encrypted_char
            key_index += 1
        else:
            ciphertext += char
    return ciphertext


def vigenere_decrypt(ciphertext, key):
    ciphertext = ciphertext.upper()
    key = key.upper()
    plaintext = ""
    key_index = 0

    for char in ciphertext:
        if char.isalpha():
            shift = ord(key[key_index % len(key)]) - ord('A')
            decrypted_char = chr((ord(char) - ord('A') - shift) % 26 + ord('A'))
            plaintext += decrypted_char
            key_index += 1
        else:
            plaintext += char
    return plaintext


@bp.route("/vigenere", methods=["GET", "POST"])
def vigenere():
    encrypted = None
    decrypted = None
    plaintext = ""
    ciphertext = ""
    key = ""

    if request.method == "POST":
        action = request.form.get("action")

        if action == "encrypt":
            plaintext = request.form.get("plaintext", "")
            key = request.form.get("key", "")
            encrypted = vigenere_encrypt(plaintext, key)

        elif action == "decrypt":
            ciphertext = request.form.get("ciphertext", "")
            key = request.form.get("key2", "")
            decrypted = vigenere_decrypt(ciphertext, key)

    return render_template("vigenere.html",
                           plaintext=plaintext,
                           ciphertext=ciphertext,
                           key=key,
                           encrypted=encrypted,
                           decrypted=decrypted)





# ====== Algoritmul ADFG(V)X======
import string

SYMBOLS = list(string.ascii_uppercase) + list("0123456789")
CODES = ['A', 'D', 'F', 'G', 'V', 'X']


# Construiește pătratul Polybius
def create_polybius_square(keyword):
    keyword = "".join(dict.fromkeys(keyword.upper()))  # elimină duplicatele
    remaining = [c for c in SYMBOLS if c not in keyword]
    square_list = list(keyword) + remaining

    polybius = {}
    inv_polybius = {}
    for i, row_code in enumerate(CODES):
        for j, col_code in enumerate(CODES):
            if i * 6 + j >= len(square_list):
                continue
            char = square_list[i * 6 + j]
            code = row_code + col_code
            polybius[char] = code
            inv_polybius[code] = char
    return polybius, inv_polybius


# Criptare cu pași
def adfgvx_encrypt(plaintext, poly_key, transpo_key):
    steps = []

    plaintext = plaintext.upper().replace(" ", "")
    steps.append(f"Text curățat: {plaintext}")

    polybius, _ = create_polybius_square(poly_key)

    coded_text = "".join([polybius[c] for c in plaintext if c in polybius])
    steps.append(f"Cod Polybius (ADFGVX): {coded_text}")

    cols = len(transpo_key)
    rows = (len(coded_text) + cols - 1) // cols
    table = [list(coded_text[i * cols:(i + 1) * cols]) for i in range(rows)]
    while len(table[-1]) < cols:
        table[-1].append("")

    steps.append("Tabel transpoziție înainte de sortare:")
    for r in table:
        steps.append(" ".join(r))

    sorted_indices = sorted(range(cols), key=lambda i: transpo_key[i])
    encrypted = ""
    for i in sorted_indices:
        for row in table:
            if row[i]:
                encrypted += row[i]

    steps.append(f"Text criptat final: {encrypted}")
    return encrypted, steps


# Decriptare cu pași
def adfgvx_decrypt(ciphertext, poly_key, transpo_key):
    steps = []

    ciphertext = ciphertext.upper().replace(" ", "")
    steps.append(f"Text criptat primit: {ciphertext}")

    polybius, inv_polybius = create_polybius_square(poly_key)

    cols = len(transpo_key)
    rows = (len(ciphertext) + cols - 1) // cols
    full_cols = len(ciphertext) % cols or cols
    col_lengths = [rows if i < full_cols else rows - 1 for i in range(cols)]
    sorted_indices = sorted(range(cols), key=lambda i: transpo_key[i])

    table = [["" for _ in range(cols)] for _ in range(rows)]
    idx = 0
    for si, col_idx in enumerate(sorted_indices):
        for r in range(col_lengths[si]):
            table[r][col_idx] = ciphertext[idx]
            idx += 1

    steps.append("Tabel transpoziție reconstruit:")
    for r in table:
        steps.append(" ".join(r))

    coded_text = "".join(["".join(row) for row in table])

    plaintext = ""
    for i in range(0, len(coded_text), 2):
        pair = coded_text[i:i + 2]
        if pair in inv_polybius:
            plaintext += inv_polybius[pair]

    steps.append(f"Text decriptat final: {plaintext}")
    return plaintext, steps


# Flask route
@bp.route("/adfgvx", methods=["GET", "POST"])
def adfgvx():
    plaintext = ""
    ciphertext = ""
    poly_key = ""
    transpo_key = ""
    encrypted = ""
    decrypted = ""
    steps_encrypt = []
    steps_decrypt = []

    if request.method == "POST":
        action = request.form.get("action")
        poly_key = request.form.get("poly_key", "").upper()
        transpo_key = request.form.get("transpo_key", "").upper()

        if action == "encrypt":
            plaintext = request.form.get("plaintext", "")
            encrypted, steps_encrypt = adfgvx_encrypt(plaintext, poly_key, transpo_key)
        elif action == "decrypt":
            ciphertext = request.form.get("ciphertext", "")
            decrypted, steps_decrypt = adfgvx_decrypt(ciphertext, poly_key, transpo_key)

    return render_template(
        "adfgvx.html",
        plaintext=plaintext,
        ciphertext=ciphertext,
        poly_key=poly_key,
        transpo_key=transpo_key,
        encrypted=encrypted,
        decrypted=decrypted,
        steps_encrypt=steps_encrypt,
        steps_decrypt=steps_decrypt
    )





# ===== RC4 =====
def rc4_crypt(text, key):
    S = list(range(256))
    j = 0
    out = []

    # KSA - Key Scheduling Algorithm
    for i in range(256):
        j = (j + S[i] + ord(key[i % len(key)])) % 256
        S[i], S[j] = S[j], S[i]

    # PRGA - Pseudo-Random Generation Algorithm
    i = j = 0
    for char in text:
        i = (i + 1) % 256
        j = (j + S[i]) % 256
        S[i], S[j] = S[j], S[i]
        K = S[(S[i] + S[j]) % 256]
        out.append(chr(ord(char) ^ K))

    return ''.join(out)


@bp.route('/rc4', methods=['GET', 'POST'])
def rc4():
    encrypted = decrypted = ''
    plaintext = ciphertext = key = ''
    if request.method == 'POST':
        action = request.form['action']
        key = request.form.get('key') or request.form.get('key2')
        if action == 'encrypt':
            plaintext = request.form['plaintext']
            encrypted = rc4_crypt(plaintext, key)
        elif action == 'decrypt':
            ciphertext = request.form['ciphertext']
            decrypted = rc4_crypt(ciphertext, key)
    return render_template('rc4.html', plaintext=plaintext, ciphertext=ciphertext,
                           key=key, encrypted=encrypted, decrypted=decrypted)



def lfsr(register, taps):
    xor = 0
    for t in taps:
        xor ^= (register >> t) & 1
    register = (register >> 1) | (xor << 18)
    return register

def a5_1_keystream(key, length):
    R1, R2, R3 = key[:19], key[19:41], key[41:]
    R1 = int(R1, 2)
    R2 = int(R2, 2)
    R3 = int(R3, 2)
    stream = []
    for _ in range(length):
        bit = ((R1 >> 18) ^ (R2 >> 21) ^ (R3 >> 22)) & 1
        stream.append(str(bit))
        R1 = lfsr(R1, [13, 16, 17, 18])
        R2 = lfsr(R2, [20, 21])
        R3 = lfsr(R3, [7, 20, 21, 22])
    return ''.join(stream)

def xor_bits(a, b):
    return ''.join('1' if x != y else '0' for x, y in zip(a, b))

@bp.route('/a5_1', methods=['GET', 'POST'])
def a5_1():
    encrypted = decrypted = ''
    plaintext = ciphertext = key = ''
    if request.method == 'POST':
        action = request.form['action']
        key = request.form.get('key') or request.form.get('key2')
        if len(key) < 64:
            key = key.ljust(64, '0')  # completează dacă e prea scurt
        if action == 'encrypt':
            plaintext = ''.join(format(ord(c), '08b') for c in request.form['plaintext'])
            keystream = a5_1_keystream(key, len(plaintext))
            encrypted = xor_bits(plaintext, keystream)
        elif action == 'decrypt':
            ciphertext = request.form['ciphertext']
            keystream = a5_1_keystream(key, len(ciphertext))
            decrypted_bits = xor_bits(ciphertext, keystream)
            decrypted = ''.join(chr(int(decrypted_bits[i:i+8], 2)) for i in range(0, len(decrypted_bits), 8))
    return render_template('a5_1.html', plaintext=plaintext, ciphertext=ciphertext,
                           key=key, encrypted=encrypted, decrypted=decrypted)

























import time
from typing import List, Tuple
from .des_tables import PC1, PC2, IP, IP_INV, SHIFTS, E_TABLE, P_TABLE, S_BOXES


# ***************************************************************
# (RESTUL CODULUI CAESAR, PLAYFAIR, VIGENERE, ADFGVX, RC4, A5/1)
# ***************************************************************

# ==============================================================================
# ====== Implementare Cifru Bloc DES (Sarcina 1, 2, 3, 4, 5) ======
# ==============================================================================

# --- Utilități DES ---

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


def s_box_substitution(input_48bit: str) -> str:
    """Aplică cele 8 S-Boxes (48 biți -> 32 biți)."""
    output_32bit = ""
    for i in range(8):
        block = input_48bit[i * 6: (i + 1) * 6]
        row = int(block[0] + block[5], 2)
        col = int(block[1:5], 2)
        val = S_BOXES[i][row][col]
        output_32bit += format(val, '04b')
    return output_32bit


# --- Sarcina 1: Generare Subchei ---

def generate_subkeys(key_hex: str) -> List[str]:
    """Generează cele 16 subchei de 48 de biți."""
    key_64bit = hex_to_bin(key_hex, pad_to=64)
    permuted_choice_1 = permute(key_64bit, PC1)
    C = permuted_choice_1[:28]
    D = permuted_choice_1[28:]

    subkeys = []
    for shift in SHIFTS:
        C = C[shift:] + C[:shift]
        D = D[shift:] + D[:shift]
        CD_combined = C + D
        subkeys.append(permute(CD_combined, PC2))

    return subkeys


# --- Sarcina 1: Funcția Feistel (F) ---

def feistel_function(R: str, K: str) -> str:
    """Funcția F(R, K) din runda Feistel (32 de biți)."""
    R_expanded = permute(R, E_TABLE)
    R_xor_K = xor_bits(R_expanded, K)
    s_box_output = s_box_substitution(R_xor_K)
    result = permute(s_box_output, P_TABLE)
    return result


# --- Sarcina 1: Operație pe Bloc Unic (DES Core) ---

def des_block_operation(data_hex: str, subkeys: List[str], encrypt: bool) -> str:
    """Criptează/Decriptează un singur bloc (64 de biți) DES."""
    data_64bit = hex_to_bin(data_hex, pad_to=64)
    data_permuted = permute(data_64bit, IP)

    L, R = data_permuted[:32], data_permuted[32:]

    key_order = subkeys if encrypt else subkeys[::-1]

    for K in key_order:
        L_next = R
        F_result = feistel_function(R, K)
        R_next = xor_bits(L, F_result)
        L, R = L_next, R_next

    final_64bit = R + L
    cipher_or_plain = permute(final_64bit, IP_INV)

    return bin_to_hex(cipher_or_plain)


# --- Sarcina 3: Moduri de Operare ECB & CBC ---

def des_ecb_operation(data_hex: str, key_hex: str, encrypt: bool) -> str:
    """Criptare/Decriptare DES în modul ECB."""
    subkeys = generate_subkeys(key_hex)
    blocks = split_into_blocks(data_hex)
    output = ""
    for block in blocks:
        output += des_block_operation(block, subkeys, encrypt)
    return output


def des_cbc_operation(data_hex: str, key_hex: str, iv_hex: str, encrypt: bool) -> str:
    """Criptare/Decriptare DES în modul CBC."""
    subkeys = generate_subkeys(key_hex)
    blocks = split_into_blocks(data_hex)
    output = ""
    current_iv = hex_to_bin(iv_hex, pad_to=64)

    if encrypt:
        for block in blocks:
            block_bin = hex_to_bin(block, pad_to=64)
            xor_block_bin = xor_bits(block_bin, current_iv)
            cipher_block_hex = des_block_operation(bin_to_hex(xor_block_bin), subkeys, encrypt)
            output += cipher_block_hex
            current_iv = hex_to_bin(cipher_block_hex, pad_to=64)
    else:
        for block in blocks:
            cipher_block_hex = block
            decrypted_block_hex = des_block_operation(cipher_block_hex, subkeys, encrypt)
            decrypted_block_bin = hex_to_bin(decrypted_block_hex, pad_to=64)

            plaintext_block_bin = xor_bits(decrypted_block_bin, current_iv)
            output += bin_to_hex(plaintext_block_bin)

            current_iv = hex_to_bin(cipher_block_hex, pad_to=64)

    return output


# --- Sarcina 4: Triple DES (3DES) EDE ---

def triple_des_ede_operation(data_hex: str, key1_hex: str, key2_hex: str, key3_hex: str, encrypt: bool) -> str:
    """Triple DES EDE: E(K3, D(K2, E(K1, P))) sau D(K1, E(K2, D(K3, C)))"""

    blocks = split_into_blocks(data_hex)
    output = ""

    for block in blocks:
        if encrypt:
            # Encrypt -> Decrypt -> Encrypt
            c1 = des_block_operation(block, generate_subkeys(key1_hex), encrypt=True)
            c2 = des_block_operation(c1, generate_subkeys(key2_hex), encrypt=False)
            c3 = des_block_operation(c2, generate_subkeys(key3_hex), encrypt=True)
            output += c3
        else:
            # Decrypt -> Encrypt -> Decrypt (folosește cheile în ordine inversă K3, K2, K1)
            c1 = des_block_operation(block, generate_subkeys(key3_hex), encrypt=False)
            c2 = des_block_operation(c1, generate_subkeys(key2_hex), encrypt=True)
            c3 = des_block_operation(c2, generate_subkeys(key1_hex), encrypt=False)
            output += c3

    return output


# --- Sarcina 5: Atac Forță Brută ---

def brute_force_attack(target_plaintext_hex: str, target_ciphertext_hex: str, fixed_key_template: str) -> Tuple[
    str, float, int]:
    """Atacă DES prin forță brută pe 16 biți."""
    unknown_bits = fixed_key_template.count('x')
    if unknown_bits != 16:
        return "Eroare: Șablonul trebuie să conțină exact 16 'x'.", 0.0, 0

    total_keys = 2 ** unknown_bits
    start_time = time.perf_counter()

    for i in range(total_keys):
        i_bin = format(i, f'0{unknown_bits}b')
        candidate_key_bin = list(fixed_key_template)

        # Înlocuiește 'x' cu biții din i_bin
        bit_index = 0
        for j in range(len(candidate_key_bin)):
            if candidate_key_bin[j] == 'x':
                candidate_key_bin[j] = i_bin[bit_index]
                bit_index += 1

        candidate_key_hex = bin_to_hex("".join(candidate_key_bin))

        encrypted_candidate = des_ecb_operation(target_plaintext_hex, candidate_key_hex, encrypt=True)

        if encrypted_candidate == target_ciphertext_hex:
            end_time = time.perf_counter()
            return candidate_key_hex, end_time - start_time, i + 1

    end_time = time.perf_counter()
    return "Cheie negăsită", end_time - start_time, total_keys


# ==============================================================================
# ====== RUTE FLASK (ADĂUGATE LA bp = Blueprint('main', __name__)) ======
# ==============================================================================

# RUTA DES (Sarcina 1, 2, 3)
@bp.route("/des", methods=["GET", "POST"])
def des_cipher():
    encrypted = decrypted = ""
    plaintext = ciphertext = key = iv = ""
    mode = "ecb"
    avalanche_logs = []
    avalanche_key_diff = avalanche_plain_diff = None

    if request.method == "POST":
        action = request.form.get("action")

        if action in ["encrypt", "decrypt"]:
            mode = request.form.get("mode", "ecb")
            key = request.form.get("key", "").upper().zfill(16)[:16]
            iv = request.form.get("iv", "").upper().zfill(16)[:16]

            if action == "encrypt":
                plaintext = request.form.get("plaintext", "").upper()
                if mode == "ecb":
                    encrypted = des_ecb_operation(plaintext, key, encrypt=True)
                elif mode == "cbc":
                    encrypted = des_cbc_operation(plaintext, key, iv, encrypt=True)
            elif action == "decrypt":
                ciphertext = request.form.get("ciphertext", "").upper()
                if mode == "ecb":
                    decrypted = des_ecb_operation(ciphertext, key, encrypt=False)
                elif mode == "cbc":
                    decrypted = des_cbc_operation(ciphertext, key, iv, encrypt=False)

        # Sarcina 2: Analiza efectului de avalanșă
        elif action == "avalanche_test":
            base_plaintext_hex = request.form.get("avalanche_plaintext", "1122334455667788").upper()
            base_key_hex = request.form.get("avalanche_key", "AABBCCDDEEFF0011").upper()

            # Text Clar
            C1 = des_ecb_operation(base_plaintext_hex, base_key_hex, encrypt=True)
            C1_bin = hex_to_bin(C1)
            P2_bin = list(hex_to_bin(base_plaintext_hex))
            P2_bin[10] = '1' if P2_bin[10] == '0' else '0'
            P2_hex = bin_to_hex("".join(P2_bin))
            C2 = des_ecb_operation(P2_hex, base_key_hex, encrypt=True)
            avalanche_plain_diff = count_bit_differences(C1_bin, hex_to_bin(C2))
            avalanche_logs.append(
                f"Text Clar (1 bit change): {avalanche_plain_diff} / 64 bits ({avalanche_plain_diff / 64 * 100:.2f}%)")

            # Cheie
            K2_bin = list(hex_to_bin(base_key_hex))
            K2_bin[20] = '1' if K2_bin[20] == '0' else '0'
            K2_hex = bin_to_hex("".join(K2_bin))
            C3 = des_ecb_operation(base_plaintext_hex, K2_hex, encrypt=True)
            avalanche_key_diff = count_bit_differences(C1_bin, hex_to_bin(C3))
            avalanche_logs.append(
                f"Cheie (1 bit change): {avalanche_key_diff} / 64 bits ({avalanche_key_diff / 64 * 100:.2f}%)")

    return render_template("des.html",
                           encrypted=encrypted, decrypted=decrypted,
                           plaintext=plaintext, ciphertext=ciphertext,
                           key=key, iv=iv, mode=mode,
                           avalanche_logs=avalanche_logs)


# RUTA 3DES (Sarcina 4)
@bp.route("/3des", methods=["GET", "POST"])
def triple_des():
    encrypted = decrypted = ""
    plaintext = ciphertext = ""
    key1 = key2 = key3 = ""

    if request.method == "POST":
        action = request.form.get("action")
        key1 = request.form.get("key1", "").upper().zfill(16)[:16]
        key2 = request.form.get("key2", "").upper().zfill(16)[:16]
        key3 = request.form.get("key3", "").upper().zfill(16)[:16]

        if action == "encrypt":
            plaintext = request.form.get("plaintext", "").upper()
            encrypted = triple_des_ede_operation(plaintext, key1, key2, key3, encrypt=True)
        elif action == "decrypt":
            ciphertext = request.form.get("ciphertext", "").upper()
            decrypted = triple_des_ede_operation(ciphertext, key1, key2, key3, encrypt=False)

    return render_template("3des.html",
                           encrypted=encrypted, decrypted=decrypted,
                           plaintext=plaintext, ciphertext=ciphertext,
                           key1=key1, key2=key2, key3=key3)


# RUTA BRUTE FORCE (Sarcina 5)
@bp.route("/bruteforce", methods=["GET", "POST"])
def brute_force_des():
    target_p = target_c = key_template = ""
    found_key = time_taken = attempts = None

    if request.method == "POST":
        target_p = request.form.get("plaintext", "").upper().zfill(16)[:16]
        target_c = request.form.get("ciphertext", "").upper().zfill(16)[:16]
        key_template = request.form.get("key_template", "").upper().zfill(64)

        # Rulare atac
        found_key, time_taken, attempts = brute_force_attack(target_p, target_c, key_template)

    return render_template("bruteforce.html",
                           target_p=target_p, target_c=target_c, key_template=key_template,
                           found_key=found_key, time_taken=time_taken, attempts=attempts)