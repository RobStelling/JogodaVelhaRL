# velha.py - Implementação de engine para jogar o jogo da velha utilizando Reinforcement Learning - Q-Learning.
#
import numpy as np
import pickle

from collections import Counter
from copy import deepcopy
from pathlib import Path
from random import sample


# Valores para casa vazia, jogador X e jogador O
# X e O são potências de 2, ou seja, usam bits diferentes
# e a operação X & O == 0
VAZIA = 0
X = 1
O = 4

# Resultados, valores são referência à implementação
# dos leds do Arduino : VELHA -> V, XGANHOU -> X, OGANHOU -> O
DEUVELHA = int('0b010101101', 2)
XGANHOU = int('0b101010101', 2)
OGANHOU = int('0b111101111', 2)

# Tamanho do tabuleiro, internamente ele é representado como um vetor de 9 posições e
# não uma matriz 3x3
LINHAS = 3
COLUNAS = 3
NUM_CASAS = LINHAS * COLUNAS

# Hiperparâmetros da política
# TAXA_EXPLORACAO é a frequência com que o política tenta alternativas não previstas
# TAXA_APRENDIZADO é o peso entre o valor do estado atual e o valor da recompensa
# GAMMA é o desconto dado à recompensa
# LIMITE_EXPLORACAO deve ser usado apenas pela política em simulações e jogo, mas não em treino
TAXA_EXPLORACAO = 0.3
TAXA_APRENDIZADO = 0.1
GAMMA = 0.8
LIMITE_EXPLORACAO = 0.0

# Recompensas a propagar
# Velha para quem inicia tem uma recompensa melhor que Velha para quem joga depois
# O jogador com X sempre inicia o jogo
VITORIA = 1.0
DERROTA = 0.0
VELHAX = 0.1
VELHAO = 0.5

# Prefixo dos nomes dos arquivos de política ao serem salvos
# p_: Política
# .pjv: Política do Jogo da Velha
PASTA_POLITICAS = "politicas"
PREFIXO_POLITICA = "p_"
EXTENSAO_POLITICA = "pjv"

# Esta implementação do jogo da velha com Reinforcement Learning (Q-learning) é feita com 3 classes.
#
# jogodaVelha:
# Nessa classe temos a representação do tabuleiro, métodos para treinamento de políticas (entre duas políticas),
# métodos para partidas (entre pessoas e políticas ou mesmo entre políticas) e simulações (entre políticas)
#
#  Maquina:
# Classe que representa uma política que será treinada
# 
# Humano:
# Classe que representa um jogador humano
#
# Humano e Maquina poderiam ser classe e subclasse porém creio que a implementação é mais simples e intuitiva
# com duas classes separadas
#
# O 'coração' do Q-learning acontece nos métodos recompensa da classe jogoDaVelha e no método
# propagaRecompensa da classe Máquina. Os dados sáo preservados no campo valores_estado da classe Máquina.
# O espaço de estados é finito mas não é necessário preencher a matriz de estados com 0 em todas as
# posições ao iniciar o treinamento. Um estado que não exista em valores_estado é assumido como 0.
# O método recompensa (jogoDaVelha) indica os valores que devem ser propagados por todos os movimentos a partir de uma partida de
# treinamento que terminou
# O método propagaRecompensa (Maquina) efetivamente propaga a premiação por todos os estados da partida que acabou de ser jogada
#
# Ao fim de cada partida de treinamento os valores de recompensa são propagados por todas as posições que ocorreram no jogo, dando
# prêmios correspondentes ao jogadores de X e O. As políticas para X e O são treinadas e preservadas em separado mas é possível,
# ao fim do treinamento, juntar as políticas e salvá-las como se fosse uma só (método combinaESalvaPolitica), gerando assim uma
# política que joga como X e como O.
# Como curiosidade, se usarmos uma política treinada apenas como X para jogar como O teremos um comportamento randômico da política,
# porque como os estados do ponto de vista do jogador O não existem na política para X, todos os valores de Q para essses estados
# serão 0.

def geraHashTabuleiro(posicao):
    return str(posicao)

class jogoDaVelha:
    """Classe para treinamento de políticas com reinforcement learning do jogo da velha
    A classe também pode ser usada para jogos entre humanos, entre humano e políticas e entre políticas
    A sua inicialização pede uma classe para cada um dos jogadores/políticas envolvidos
    Assume que o primeiro jogador inicia e representa o seus movimentos com X
    Os movimentos do adversário são representados com O
    """
    # O Tabuleiro é um vetor de 9 posições (3x3),
    # onde cada casa pode ser 0 (casa vazia), 1 (casa com X) ou 4 (casa com O)
    # Observe que 1 e 4 são potências de 2, para permitir
    # operações bit a bit e também somas como consultas (não utilizadas nessa versão até o momento).
    # Por exemplo:
    # - A operação E bit a bit (&) entre 3 casas só tem resultado X ou O se todas as casas forem X ou todas forem O
    # X sempre começa
    def __init__(self, jogadorX, jogadorO):
        self.tabuleiro = np.zeros(NUM_CASAS, dtype=int)
        self.jogador = {X: jogadorX, O: jogadorO}
        self.terminou = False
        # X sempre começa
        self.vez = X

    def reinicia(self):
        """Reinicializa as condições do jogo, mantendo os mesmos jogadores
        Usado normalmente durante o treinamento da política
        """
        self.tabuleiro = np.zeros(NUM_CASAS, dtype=int)
        self.terminou = False
        self.vez = X

    def resultado(self):
        """Verifica o resultado do jogo
        Verifica todas as linhas, colunas e as duas diagonais em busca de 3 marcas consecutivas
        Retorna quem ganhou ou velha, se o jogo tiver acabado, senão retorna None
        Atualiza a flag jogoDaVelha.terminou se o jogo tiver terminado
        """
        # Verifica todas as linhas
        for i in range(0, NUM_CASAS, LINHAS):
            completou = np.bitwise_and.reduce(self.tabuleiro[i:i+LINHAS])
            if completou == X:
                self.terminou = True
                return XGANHOU
            elif completou == O:
                self.terminou = True
                return OGANHOU
        # Verifica todas as colunas
        for i in range(0, COLUNAS):
            completou = self.tabuleiro[i] & self.tabuleiro[i+COLUNAS] & self.tabuleiro[i+2*COLUNAS]
            if completou == X:
                self.terminou = True
                return XGANHOU
            elif completou == O:
                self.terminou = True
                return OGANHOU
        # Verifica as duas diagonais
        completou = (self.tabuleiro[0] & self.tabuleiro[4] & self.tabuleiro[8]) | \
                    (self.tabuleiro[2] & self.tabuleiro[4] & self.tabuleiro[6])
        if completou == X:
            self.terminou = True
            return XGANHOU
        elif completou == O:
            self.terminou = True
            return OGANHOU
        # Verifica se o jogo acabou com velha (não há mais posições livres)
        if self.numcasasLivres() == 0:
            self.terminou = True
            return DEUVELHA
        
        # Senão, o jogo ainda não terminou
        return None

    def numcasasLivres(self):
        """Retorna o número de casas livres do jogo atual"""
        return sum(self.tabuleiro == VAZIA)

    def treinamento(self, rodadas=1000, verifica=1000):
        """Executa o loop de treinamento
        Recebe como parâmetros opcionais o número de rodadas e de quantas em quantas rodadas o treinamento
        deve ser verificado
        Enquanto o treinamento é realizado as políticas para X e O são atualizadas com recompensas pré-determinadas
        """
        for rodada in range(rodadas):
            if rodada % verifica == 0:
                print(f"Rodadas: {rodada}")
            while True:
                alternativas = self.casasLivres()
                vez = self.vez
                jogada = self.jogador[vez].escolheJogada(alternativas, self.tabuleiro, vez)
                self.jogada(jogada)
                hash_tabuleiro = geraHashTabuleiro(self.tabuleiro)
                self.jogador[vez].acrescentaEstado(hash_tabuleiro)

                # Se o jogo terminou (X venceu, O venceu ou velha)
                # propaga as recompensas pelos estados,
                # reinicia jogo e jogadores e volta ao loop de treinamento
                if self.resultado() is not None:
                    self.recompensa()
                    self.reinicia()
                    self.jogador[X].reinicia()
                    self.jogador[O].reinicia()
                    break

        print(f"Treinamento finalizado: {rodadas} rodadas")

    def casasLivres(self):
        """Retorna uma lista de casas livres"""
        return [casa for casa, vazia in enumerate(self.tabuleiro == VAZIA) if vazia]

    def jogada(self, casa):
        """Faz uma jogada no jogo atual
        Ou seja, coloca um X ou O na casa que foi escolhida para jogar e
        troca o jogador da vez
        """
        self.tabuleiro[casa] = self.vez
        self.vez = X if self.vez == O else O

    def recompensa(self):
        """Passa as recompensas para as políticas de acordo com o resultado do jogo
        As recompensas são propagadas nas políticas de cada jogador a partir dos seus lances nessa instância do jogo
        """
        resultado = self.resultado()
        if resultado == XGANHOU:
            self.jogador[X].propagaRecompensa(VITORIA)
            self.jogador[O].propagaRecompensa(DERROTA)
        elif resultado == OGANHOU:
            self.jogador[X].propagaRecompensa(DERROTA)
            self.jogador[O].propagaRecompensa(VITORIA)
        else: # Deu velha
            self.jogador[X].propagaRecompensa(VELHAX)
            self.jogador[O].propagaRecompensa(VELHAO)

    def partida(self, saida=True):
        """Jogo entre dois jogadores
        Podem ser duas políticas, uma política e um humano ou dois humanos
        Se saida == True então mostrará o tabuleiro com os lances efetuados
        Se pelo menos um dos jogadores for humano é recomendável que a flag saida seja True
        Reinicia as condições do jogo ao fim da partida
        """
        while not self.terminou:
            alternativas = self.casasLivres()
            vez = self.vez
            if self.jogador[vez].tipo == "Computador":
                jogada = self.jogador[vez].escolheJogada(alternativas, self.tabuleiro, vez)
            else:
                jogada = self.jogador[vez].escolheJogada(alternativas)
            # O método self.jogada altera o jogador da vez (self.vez),
            # por isso o valor self.vez é guardado na variável vez
            self.jogada(jogada)
            if (saida):
                self.mostraTabuleiro()

            resultado = self.resultado()
            if resultado is not None:
                # Então o jogo acabou
                if saida:
                    if resultado == DEUVELHA:
                        print("Deu velha!")
                    else:
                        print(f"{self.jogador[vez].nome} venceu!")
    
        self.reinicia()
        return resultado

    def simulacao(self, partidas=100):
        """Simulação do jogo entre políticas
        Retorna o total de resultados e os tabuleiros finais
        """
        totalizacao = Counter()
        tabuleiros = Counter()
        for _ in range(partidas):
            while not self.terminou:
                alternativas = self.casasLivres()
                vez = self.vez
                jogada = self.jogador[vez].escolheJogada(alternativas, self.tabuleiro, vez)
                self.jogada(jogada)
                resultado = self.resultado()
                if resultado is not None:
                    # Então o jogo acabou
                    totalizacao['Velha' if resultado == DEUVELHA else self.jogador[vez].nome]+=1
                    tabuleiros[geraHashTabuleiro(self.tabuleiro)]+=1
        
            self.reinicia()
        return totalizacao, tabuleiros
    
    def mostraTabuleiro(self):
        """Mostra a posição atual do tabuleiro de forma simples"""
        simbolo = {X: 'X', O: 'O', 0: ' '}
        for i in range(0, LINHAS):
            print('-------------')
            saida = "| "
            for j in range(0, COLUNAS):
                valor = self.tabuleiro[i*COLUNAS + j]
                saida += simbolo[valor] + " | "
            print(saida)
        print('-------------', flush=True)


def valorEstado(estados, posicao):
    """Retorna o valor de um estado, se ele não existir, retorna 0
    Isso equivale a inicializar todos os estados com 0
    """
    valor = estados.get(posicao)
    # Se não tem hash do próximo tabuleiro, então assume que o valor é 0
    if valor is None:
        return 0
    return valor

class Maquina():
    """Classe para representar uma política de jogo da velha
    Utilizado tanto no treinamento com reinforcement learning da política quanto em partidas contra outros adversários
    Para o treinamento espera como entrada uma taxa de exploração, uma taxa de aprendizado e um fator de desconto gamma
    """
    def __init__(self, nome,
                 taxa_exploracao=TAXA_EXPLORACAO,
                 taxa_aprendizado=TAXA_APRENDIZADO,
                 gamma=GAMMA,
                 limite_exploracao=LIMITE_EXPLORACAO,
                 depuracao=False):
        """Intancia o objeto Maquina
        Nome: usado para salvar/recuperar as políticas e também para representar o jogador
        Tipo: indica se é uma  política ou um humano
        Estados lista os estados do jogo atual
        Taxa_aprendizado: peso utilizado na propagação das recompensas
        Taxa_exploracao: percentual de exploracao de alternativas fora da política atual
        Gamma: desconto da recompensa a ser propagada
        Valores_estado: dicionário de valores dos estados da política, pares (posição, valor)
        """
        self.nome = nome
        self.tipo = "Computador"
        self.estados = []
        self.taxa_aprendizado = taxa_aprendizado
        self.taxa_exploracao = taxa_exploracao
        self.gamma = gamma
        self.limite_exploracao = limite_exploracao
        self.depuracao = depuracao
        self.valores_estado = {}

    def reinicia(self):
        """Reinicia a política para a próxima partida
        Apenas descarta os estados do jogo atual"""
        self.estados = []

    def escolheJogada(self, casasLivres, tabuleiro, jogador):
        """Retorna a jogada a fazer, em função da política até o momento
        Pode retornar uma jogada randômica, entre as jogadas disponíveis, de
        acordo com a taxa de exploração
        Durante uma partida a taxa de exploração deve ser 0
        """
        def geraAlternativas(valores_estado, tabuleiro, casasLivres, jogador):
            valoresHash = []
            for c in casasLivres:
                tabuleiro[c] = jogador
                hash_tabuleiro = geraHashTabuleiro(tabuleiro)
                valoresHash.append({'movimento': c, 'valor': valorEstado(valores_estado, hash_tabuleiro)})
                tabuleiro[c] = VAZIA
            return sorted(valoresHash, key=lambda x: x['valor'], reverse=True)

        if np.random.uniform(0, 1) <= self.taxa_exploracao:
            # Executa ação randômica de acordo com a taxa de exploração
            # se a taxa de exploração for 0.0 então todas as ações virão da
            # política
            jogada = np.random.choice(casasLivres)
        else:
            valores_hash = geraAlternativas(self.valores_estado, tabuleiro, casasLivres, jogador)
            max = valores_hash[0]['valor']
            if self.depuracao:
                print(max, self.limite_exploracao, valores_hash)
            # Seleciona alternativas de lances, se valor >= max-limite_exploração
            alternativas = [opcao['movimento'] for opcao in valores_hash if (max-opcao['valor']) <= self.limite_exploracao]
            # Escolhe uma opção aleatoriamente
            jogada = sample(alternativas, 1)
        return jogada
    
    def acrescentaEstado(self, estado):
        """Acresdenta um estado na lista, usado durante o treinamento"""
        self.estados.append(estado)

    def propagaRecompensa(self, premio):
        """Propaga a recompensa pelos estados do jogo atual
        É o principal processo do reinforcement learning
        Perceba que os estados são percorridos de tras para frente, ou seja,
        dos ultimos movimentos para os primeros, e que o valor da recompensa é reduzido pelo fator de desconto gamma"""
        for estado in reversed(self.estados):
            if self.valores_estado.get(estado) is None:
                self.valores_estado[estado] = 0
            self.valores_estado[estado] += self.taxa_aprendizado * (self.gamma * premio - self.valores_estado[estado])
            premio = self.valores_estado[estado]

    def salvaPolitica(self, prefixo=PREFIXO_POLITICA):
        """Salva uma política para uso futuro"""
        pasta = Path(f'./{PASTA_POLITICAS}')
        if not pasta.exists():
            pasta.mkdir()
        if pasta.is_dir():
            nome_arquivo = pasta / f'{prefixo}{str(self.nome)}.{EXTENSAO_POLITICA}' 
            with open(nome_arquivo, 'wb' ) as arquivo:
                pickle.dump(self.valores_estado, arquivo)
        else:
            raise ValueError(f"Não consigo criar arquivos em {pasta}")

    def carregaPolitica(self, politica):
        """Carrega uma política para jogar ou continuar um treinamento"""
        pasta = Path(f'./{PASTA_POLITICAS}')
        nome_arquivo = pasta / f'{politica}.{EXTENSAO_POLITICA}'
        if nome_arquivo.exists():
            with open(nome_arquivo, 'rb') as arquivo:
                self.valores_estado = pickle.load(arquivo)
        else:
            raise ValueError(f"Política {politica} não existe!")

    def combinaESalvaPolitica(self, politica2, nome, prefixo=PREFIXO_POLITICA):
        """Combina duas políticas em uma e salva valores_estado
        O objetivo é combinar duas políticas, uma para X e uma para O em uma só política, já que as
        tuplas (hashTabuleiro, valor) são mutualmente excludentes nas políticas para X e O
        """
        politica = deepcopy(self)
        politica.nome = nome
        politica.valores_estado = {**self.valores_estado, **politica2.valores_estado}
        pasta = Path(f'./{PASTA_POLITICAS}')
        if not pasta.exists():
            pasta.mkdir()
        if pasta.is_dir():
            nome_arquivo = pasta / f'{prefixo}{nome}.{EXTENSAO_POLITICA}'
            with open(nome_arquivo, 'wb') as arquivo:
                pickle.dump(politica.valores_estado, arquivo)
        else:
            raise ValueError(f"Não consigo criar arquivos em {pasta}")

class Humano:
    """Classe que representa as ações de um jogador humano"""
    def __init__(self, nome):
        self.nome = nome
        self.tipo = "Humano"
    
    def escolheJogada(self, casasLivres):
        """Pergunta qual o lance a fazer
        Recomendável que a partida seja chamada com saida == True
        se esta envolver pelo menos um jogador humano
        """
        while True:
            print("Qual a casa? ", end='')
            acao = int(input())
            print(f"Você jogou {acao}")
            if acao in casasLivres:
                return acao
    

if __name__ == "__main__":
    # Exemplo de treinamento e jogo
    # Define duas políticas que serão treinadas
    politicaX = Maquina("X")
    politicaO = Maquina("O")

    treinamento = jogoDaVelha(politicaX, politicaO)
    print("Treinando...")
    treinamento.treinamento(10000)
    # Salva as políticas geradas
    politicaX.salvaPolitica()
    politicaO.salvaPolitica()

    # Carrega uma politica salva e joga contra um humano
    politicaX = Maquina("Computador", taxa_exploracao=0.0)
    politicaX.carregaPolitica("p_X")
    humano = Humano("Walter")
    jogo = jogoDaVelha(politicaX, humano)
    jogo.partida()