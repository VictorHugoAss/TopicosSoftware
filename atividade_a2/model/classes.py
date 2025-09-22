from abc import ABC, abstractmethod

class Classe(ABC):
    @abstractmethod
    def habilidades(self):
        pass

class Guerreiro(Classe):
    def habilidades(self):
        return ["Uso de todas as armas e armaduras", "Aparar (ataques)", "Maestria em Arma (bônus de dano)", 
                "Ataque Extra (a partir do 6º nível)"]

class Clerigo(Classe):
    def habilidades(self):
        return ["Magias divinas", "Afastar mortos-vivos", "Restrição: Apenas armas impactantes"]

class Ladrao(Classe):
    def habilidades(self):
        return ["Ouvir Ruídos", "Talentos de Ladrão (Armadilha, Arrombar, Escalar, Furtividade, Punga)",
                "Restrição: Apenas armas pequenas ou médias", "Restrição: Apenas armaduras leves"]
