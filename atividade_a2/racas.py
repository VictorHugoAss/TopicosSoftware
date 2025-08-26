from abc import ABC, abstractmethod

class Raca(ABC):
    def __init__(self, movimento, infravisao, alinhamento):
        self.movimento = movimento
        self.infravisao = infravisao
        self.alinhamento = alinhamento

    @abstractmethod
    def habilidades(self):
        pass

class Humano(Raca):
    def __init__(self):
        super().__init__(movimento=9, infravisao=0, alinhamento="Qualquer")

    def habilidades(self):
        return ["Aprendizado (+10% XP)", "Adaptabilidade (+1 JP à escolha)"]

class Elfo(Raca):
    def __init__(self):
        super().__init__(movimento=9, infravisao=18, alinhamento="Ordeiro")

    def habilidades(self):
        return ["Percepção Natural (detecta portas secretas)", "Graciosos (+1 JPD)", "Arma Racial (Bônus em arcos)", "Imunidades (sono e paralisia de Ghoul)"]

class Anao(Raca):
    def __init__(self):
        super().__init__(movimento=6, infravisao=18, alinhamento="Neutro")

    def habilidades(self):
        return ["Mineradores (detecta anomalias em pedras/armadilhas)"]
