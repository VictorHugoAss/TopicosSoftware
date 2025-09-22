class Personagem:
    def __init__(self, nome, atributos, raca, classe):
        self.nome = nome
        self.atributos = atributos
        self.raca = raca
        self.classe = classe

    def ficha(self):
        return f"""
--- FICHA DE PERSONAGEM ---
Nome: {self.nome}
Raça: {self.raca.__class__.__name__}
Classe: {self.classe.__class__.__name__}
Atributos: {self.atributos}
Movimento: {self.raca.movimento}
Infravisão: {self.raca.infravisao}
Alinhamento: {self.raca.alinhamento}
Habilidades Raciais: {", ".join(self.raca.habilidades())}
Habilidades da Classe: {", ".join(self.classe.habilidades())}
"""
