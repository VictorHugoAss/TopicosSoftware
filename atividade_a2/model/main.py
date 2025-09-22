from personagem_creator.model.atributos import GeradorAtributos, Atributos
from personagem_creator.model.racas import Humano, Elfo, Anao
from personagem_creator.model.classes import Guerreiro, Clerigo, Ladrao
from personagem_creator.model.personagem import Personagem

def escolher_opcao(pergunta, opcoes):
    print(pergunta)
    for i, opcao in enumerate(opcoes, 1):
        print(f"{i} - {opcao}")
    while True:
        try:
            escolha = int(input("Escolha: "))
            if 1 <= escolha <= len(opcoes):
                return escolha
            else:
                print("Opção inválida, tente novamente.")
        except ValueError:
            print("Digite um número válido!")

if __name__ == "__main__":
    print("=== CRIAÇÃO DE PERSONAGEM ===")
    nome = input("Digite o nome do personagem: ")

    escolha_attr = escolher_opcao(
        "Escolha o estilo de geração de atributos:",
        ["Clássico (3d6 ordem fixa)", "Aventureiro (3d6 escolha ordem)", "Heróico (4d6 escolha ordem)"]
    )

    if escolha_attr == 1:
        atributos = GeradorAtributos.classico()
    else:
        if escolha_attr == 2:
            valores = GeradorAtributos.aventureiro()
        else:
            valores = GeradorAtributos.heroico()

        print(f"\nValores rolados: {valores}")
        print("Distribua os valores nos atributos:")

        ordem = ["Força", "Destreza", "Constituição", "Inteligência", "Sabedoria", "Carisma"]
        alocados = []
        for atributo in ordem:
            while True:
                try:
                    print(f"Disponíveis: {valores}")
                    escolha = int(input(f"Escolha valor para {atributo}: "))
                    if escolha in valores:
                        alocados.append(escolha)
                        valores.remove(escolha)
                        break
                    else:
                        print("Valor inválido!")
                except ValueError:
                    print("Digite um número válido.")
        atributos = Atributos(*alocados)


    escolha_raca = escolher_opcao("Escolha a raça:", ["Humano", "Elfo", "Anão"])
    raca = [Humano(), Elfo(), Anao()][escolha_raca-1]


    escolha_classe = escolher_opcao("Escolha a classe:", ["Guerreiro", "Clérigo", "Ladrão"])
    classe = [Guerreiro(), Clerigo(), Ladrao()][escolha_classe-1]


    personagem = Personagem(nome, atributos, raca, classe)
    print(personagem.ficha())
