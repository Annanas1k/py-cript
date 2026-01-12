from flask import Blueprint, render_template, request

bp = Blueprint('a5_1', __name__)


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

