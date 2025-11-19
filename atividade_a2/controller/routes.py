from flask import Blueprint, render_template, request, redirect, url_for, session
import json
import os 

from model.personagem import Personagem
from model.atributos import GeradorAtributos, Atributos
from model.racas import Humano, Elfo, Anao
from model.classes import Guerreiro, Clerigo, Ladrao

bp = Blueprint('main', __name__)

RACA_CLASSES = {
    "Humano": Humano,
    "Elfo": Elfo,
    "Anao": Anao
}

CLASSE_CLASSES = {
    "Guerreiro": Guerreiro,
    "Clerigo": Clerigo,
    "Ladrao": Ladrao
}


@bp.route('/', methods=['GET', 'POST'])
def index():

    if request.method == 'POST':
        nome = request.form['nome']
        raca = request.form['raca']
        classe = request.form['classe']
        geracao_atributos = request.form['geracao_atributos']
        
        if geracao_atributos == 'classico':
            atributos_obj = GeradorAtributos.classico()
            raca_obj = RACA_CLASSES[raca]()
            classe_obj = CLASSE_CLASSES[classe]()
            
            personagem = Personagem(nome, atributos_obj, raca_obj, classe_obj)
            
            salvar_personagem_json(personagem)
            
            return render_template('ficha.html', personagem=personagem)
            
        elif geracao_atributos in ['aventureiro', 'heroico']:
            if geracao_atributos == 'aventureiro':
                valores = [GeradorAtributos.rolar_3d6() for _ in range(6)]
            elif geracao_atributos == 'heroico':
                valores = [GeradorAtributos.rolar_4d6_descarta_menor() for _ in range(6)]
            
            session['personagem_temp'] = {'nome': nome, 'raca': raca, 'classe': classe}
            session['valores_disponiveis'] = valores
            return redirect(url_for('main.distribuir_atributos'))

    return render_template('index.html')

@bp.route('/distribuir_atributos', methods=['GET', 'POST'])
def distribuir_atributos():

    if request.method == 'GET':
        if 'valores_disponiveis' not in session or 'personagem_temp' not in session:
            return redirect(url_for('main.index'))
        
        return render_template('distribuir_atributos.html', 
                               valores=session['valores_disponiveis'],
                               nome=session['personagem_temp']['nome'],
                               raca=session['personagem_temp']['raca'],
                               classe=session['personagem_temp']['classe'])
    
    elif request.method == 'POST':
        nome = request.form['nome']
        raca_str = request.form['raca']
        classe_str = request.form['classe']
        
        forca = int(request.form['forca'])
        destreza = int(request.form['destreza'])
        constituicao = int(request.form['constituicao'])
        inteligencia = int(request.form['inteligencia'])
        sabedoria = int(request.form['sabedoria'])
        carisma = int(request.form['carisma'])

        atributos = Atributos(forca, destreza, constituicao, inteligencia, sabedoria, carisma)
        raca_obj = RACA_CLASSES[raca_str]()
        classe_obj = CLASSE_CLASSES[classe_str]()

        personagem = Personagem(nome, atributos, raca_obj, classe_obj)
        
        salvar_personagem_json(personagem)
        
        session.pop('personagem_temp', None)
        session.pop('valores_disponiveis', None)

        return render_template('ficha.html', personagem=personagem)



def salvar_personagem_json(personagem: Personagem):

    try:
        personagem_data = personagem.to_dict()

        nome_arquivo = f"{personagem.nome.replace(' ', '_').lower()}_ficha.json"
        
        diretorio_raiz = os.path.dirname(os.path.abspath(os.path.join(__file__, '..', '..')))
        diretorio_destino = os.path.join(diretorio_raiz, 'personagens_salvos')
        caminho_completo = os.path.join(diretorio_destino, nome_arquivo)

        os.makedirs(diretorio_destino, exist_ok=True)

        with open(caminho_completo, 'w', encoding='utf-8') as f:
            json.dump(personagem_data, f, ensure_ascii=False, indent=4)
        
        print(f"SUCESSO: Personagem '{personagem.nome}' salvo em: {caminho_completo}")

    except AttributeError:
        print("ERRO FATAL: A classe Personagem não possui o método to_dict(). Implemente-o em models/personagem.py.")
    except Exception as e:
        print(f"Erro ao salvar o personagem em JSON: {e}")
