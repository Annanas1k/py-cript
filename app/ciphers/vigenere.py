from flask import Blueprint, render_template, request

bp = Blueprint('vigenere', __name__)


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