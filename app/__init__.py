from flask import Flask
from app.main_bp import bp as main_bp
from .ciphers.caesar import bp as caesar_bp
from .ciphers.playfair import  bp as playfair_bp
from .ciphers.vigenere import bp as vigenere_bp
from .ciphers.rc4 import bp as rc4_bp
from .ciphers.a5_1 import bp as a5_1_bp
from .ciphers.adfgvx import bp as adfgvx_bp
from .ciphers.des import bp as des_bp




def create_app():
    app = Flask(__name__)

    #------------------------------
    app.register_blueprint(main_bp, url_prefix='/')
    app.register_blueprint(caesar_bp, url_prefix='/caesar')
    app.register_blueprint(playfair_bp, url_prefix='/playfair')
    app.register_blueprint(vigenere_bp, url_prefix='/vigenere')
    app.register_blueprint(adfgvx_bp, url_prefix='/adfgvx')
    app.register_blueprint(rc4_bp, url_prefix='/rc4')
    app.register_blueprint(a5_1_bp, url_prefix='/a5_1')
    app.register_blueprint(des_bp, url_prefix='/des')




    return app