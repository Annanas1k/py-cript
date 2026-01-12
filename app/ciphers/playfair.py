from flask import Blueprint, render_template, request

bp = Blueprint('playfair', __name__)


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