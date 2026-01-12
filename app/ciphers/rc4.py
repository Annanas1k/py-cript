from flask import Blueprint, render_template, request

bp = Blueprint('rc4', __name__)

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
