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
    
    def to_dict(self):
        """Converte o objeto Personagem em um dicionário serializável em JSON."""
        return {
            "nome": self.nome,
            "raca": {
                "tipo": self.raca.__class__.__name__,
                "movimento": self.raca.movimento,
                "infravisao": self.raca.infravisao,
                "alinhamento": self.raca.alinhamento,
                "habilidades": self.raca.habilidades()
            },
            "classe": {
                "tipo": self.classe.__class__.__name__,
                "habilidades": self.classe.habilidades()
            },
            "atributos": {
                "forca": self.atributos.forca,
                "destreza": self.atributos.destreza,
                "constituicao": self.atributos.constituicao,
                "inteligencia": self.atributos.inteligencia,
                "sabedoria": self.atributos.sabedoria,
                "carisma": self.atributos.carisma
            }
        }
