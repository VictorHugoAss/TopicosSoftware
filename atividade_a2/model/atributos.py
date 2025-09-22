import random

class Atributos:
    def __init__(self, forca, destreza, constituicao, inteligencia, sabedoria, carisma):
        self.forca = forca
        self.destreza = destreza
        self.constituicao = constituicao
        self.inteligencia = inteligencia
        self.sabedoria = sabedoria
        self.carisma = carisma

    def __str__(self):
        return (f"FOR:{self.forca} DES:{self.destreza} CON:{self.constituicao} "
                f"INT:{self.inteligencia} SAB:{self.sabedoria} CAR:{self.carisma}")

class GeradorAtributos:
    @staticmethod
    def rolar_3d6():
        return sum(random.randint(1, 6) for _ in range(3))

    @staticmethod
    def rolar_4d6_descarta_menor():
        rolagem = [random.randint(1, 6) for _ in range(4)]
        rolagem.remove(min(rolagem))
        return sum(rolagem)

    @staticmethod
    def classico():
        return Atributos(
            GeradorAtributos.rolar_3d6(),  # Força
            GeradorAtributos.rolar_3d6(),  # Destreza
            GeradorAtributos.rolar_3d6(),  # Constituição
            GeradorAtributos.rolar_3d6(),  # Inteligência
            GeradorAtributos.rolar_3d6(),  # Sabedoria
            GeradorAtributos.rolar_3d6()   # Carisma
        )

    @staticmethod
    def aventureiro():
        valores = [GeradorAtributos.rolar_3d6() for _ in range(6)]
        return valores  

    @staticmethod
    def heroico():
        valores = [GeradorAtributos.rolar_4d6_descarta_menor() for _ in range(6)]
        return valores  
