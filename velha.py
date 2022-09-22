import numpy as np
import pickle
from random import sample
from collections import Counter

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

# Hiperparâmetros do modelo
# TAXA_EXPLORACAO é a frequência com que o modelo tenta alternativas não previstas
# TAXA_APRENDIZADO é o peso entre o valor do estado atual e o valor da recompensa
# GAMMA é o desconto dado à recompensa
# LIMITE_EXPLORACAO deve ser usado apenas pelo modelo em simulações e jogo, mas não em treino
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
PREFIXO_POLITICA = "p_"
EXTENSAO_POLITICA = "pjv"


def geraHashTabuleiro(posicao):
    return str(posicao)

class jogoDaVelha:
    """Classe para treinamento de modelo com reinforcement learning de modelo do jogo da velha
    A classe também pode ser usada para jogos entre humanos, entre humano e modelos e entre modelos
    A sua inicialização pede uma classe para cada um dos jogadores/modelos envolvidos
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
        Usado normalmente durante o treinamento do modelo
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
        Podem ser dois modelos, um modelo e um humano ou dois humanos
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
    """Classe para representar um modelo de jogo da velha
    Utilizado tanto no treinamento com reinforcement learning do modelo quanto em partidas contra outros adversários
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
        Tipo: indica se é um modelo ou um humano
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
        """Reinicia o modelo para a próxima partida
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
            jogada = np.random.choice(casasLivres)
        else:
            valores_hash = geraAlternativas(self.valores_estado, tabuleiro, casasLivres, jogador)
            max = valores_hash[0]['valor']
            if self.depuracao:
                print(max, self.limite_exploracao, valores_hash)
            alternativas = [opcao['movimento'] for opcao in valores_hash if (max-opcao['valor']) <= self.limite_exploracao]
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
        with open(f'{prefixo}{str(self.nome)}.{EXTENSAO_POLITICA}', 'wb' ) as arquivo:
            pickle.dump(self.valores_estado, arquivo)

    def carregaPolitica(self, politica):
        """Carrega uma política para jogar ou continuar um treinamento"""
        with open(f'{politica}.{EXTENSAO_POLITICA}', 'rb') as arquivo:
            self.valores_estado = pickle.load(arquivo)

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
    # Define dois modelos que serão treinados
    modeloX = Maquina("X")
    modeloO = Maquina("O")

    treinamento = jogoDaVelha(modeloX, modeloO)
    print("Treinando...")
    treinamento.treinamento(10000)
    # Salva as políticas geradas
    modeloX.salvaPolitica()
    modeloO.salvaPolitica()

    # Carrega um modelo salva e joga contra  um humano
    modeloX = Maquina("Computador", taxa_exploracao=0.0)
    modeloX.carregaPolitica("p_X")
    humano = Humano("Walter")
    jogo = jogoDaVelha(modeloX, humano)
    jogo.partida()