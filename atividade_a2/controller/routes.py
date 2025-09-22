

from flask import Blueprint, render_template, request
from model.atributos import GeradorAtributos, Atributos
from model.racas import Humano, Elfo, Anao
from model.classes import Guerreiro, Clerigo, Ladrao
from model.personagem import Personagem

bp = Blueprint('main', __name__)

@bp.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        nome = request.form['nome']
        estilo_geracao = request.form['geracao_atributos']
        raca_escolhida = request.form['raca']
        classe_escolhida = request.form['classe']
        
        if estilo_geracao == 'classico':
            atributos = GeradorAtributos.classico()

            if raca_escolhida == 'Humano':
                raca = Humano()
            elif raca_escolhida == 'Elfo':
                raca = Elfo()
            else:
                raca = Anao()

            if classe_escolhida == 'Guerreiro':
                classe = Guerreiro()
            elif classe_escolhida == 'Clerigo':
                classe = Clerigo()
            else:
                classe = Ladrao()

            personagem = Personagem(nome, atributos, raca, classe)
            return render_template('ficha.html', personagem=personagem)
        
        elif estilo_geracao in ['heroico', 'aventureiro']:
            if estilo_geracao == 'heroico':
                valores_atributos = GeradorAtributos.heroico()
            else:
                valores_atributos = GeradorAtributos.aventureiro()
            
            return render_template('distribuir_atributos.html', nome=nome, valores=valores_atributos, raca=raca_escolhida, classe=classe_escolhida)
    
    return render_template('index.html')


@bp.route('/distribuir_atributos', methods=['POST'])
def distribuir_atributos():
    nome = request.form['nome']
    raca_escolhida = request.form['raca']
    classe_escolhida = request.form['classe']


    forca = int(request.form['forca'])
    destreza = int(request.form['destreza'])
    constituicao = int(request.form['constituicao'])
    inteligencia = int(request.form['inteligencia'])
    sabedoria = int(request.form['sabedoria'])
    carisma = int(request.form['carisma'])

    atributos = Atributos(forca, destreza, constituicao, inteligencia, sabedoria, carisma)

    if raca_escolhida == 'Humano':
        raca = Humano()
    elif raca_escolhida == 'Elfo':
        raca = Elfo()
    else:
        raca = Anao()

    if classe_escolhida == 'Guerreiro':
        classe = Guerreiro()
    elif classe_escolhida == 'Clerigo':
        classe = Clerigo()
    else:
        classe = Ladrao()

    personagem = Personagem(nome, atributos, raca, classe)
    return render_template('ficha.html', personagem=personagem)