from flask import Blueprint, render_template, request

bp = Blueprint('adfg', __name__)


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
