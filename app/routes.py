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




