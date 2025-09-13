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