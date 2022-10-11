# velha.py - Implementação de engine para jogar o jogo da velha utilizando Reinforcement Learning - Q-Learning.
# https://github.com/RobStelling/JogodaVelhaRL
#
# Reinforcement Learning (RL) é uma área de aprendizado de máquina em que agentes inteligentes tomam ações em um ambiente
# de forma a maximizar a noção de recompensa comulativa. Um mecanismo similar ocorre quando uma trilha é explorada
# e aberta por animais na floresta, ou quando formigas exploram um ambiente e encontram alimento.
# O Reinforcement Learning é um dos tres paradigmas mais importantes de aprendizado de máquina, em conjunto com
# aprendizado supervisionado e aprendizado não supervisionado.
#
# Nesse exemplo de RL vamos utilizar uma técnica conhecida como Q-learning.
# A ideia principal do Q-learning é preencher uma tabela de tamanho SxA, com S estados e A ações com pontuações para cada par - ou tupla - (s, a)
# Para cada estado "s", a melhor ação "a" a tomar é aquela com a pontuação mais alta. Assim, com a tabela totalmente preenchida, e preenchida
# corretamente, é possível escolher a melhor ação entre as várias ações possíveis para cada estado.
#
# No jogo da velha o estado é o tabuleiro atual, e as ações são as possíveis jogadas de um jogador no tabuleiro atual. Durante o
# treinamento a tabela Q é inicializada com todos os valores em 0, já que no início não sabemos qual ação tomar em cada estado.
# Para preencher a tabela Q é utilizada uma etapa de treinamento, onde os agentes (ou jogadores simulados) jogam várias vezes
# entre si, distribuindo recompensas de acordo com o resultado do jogo. No nosso caso as recompensas são distribuídas apenas
# no fim da partida, mas há implementações onde as recompensas são dadas a cada movimento. As recompensas do nosso jogo da velha
# são VITORIA, DERROTA, VELHAX e VELHAO.
# Enquanto o treinamento avança, são atribuídos valores às ações tomadas durante cada partida de treinamento. No fim de cada partida
# as recompensas são atribuídas à posição final, de acordo com o resultado da partida, e então "propagadas" para os lances anteriores,
# de forma que uma partida que tenha sido perdida propague recompensas negativas e uma partida vencida propague recompensas positivas.
# As recompensas no caso de VELHA são 0, mas poderiam ser outros valores. Escolhemos também separar as recompensas de VELHA para o
# jogador X, que sempre inicia nessa implementação, e jogador O, mas os valores dessas recompensas foram definidos como 0.
#
# A recompensa é calculada atualizando os valores Q(s, a), usando a equação de Bellman:
#
# Qnovo(s, a) = (1 - alfa) * q(s, a) + alfa * (Rt+1 + gama * max(q(s', a')))
#                                                             a'
# A equação acima também pode ser reescrita como:
#
# Qnovo(s, a) = q(s, a) + alfa * (Rt+1 + gama * max(q(s', a')) - q(s, a))
#                                                a'
#
# Essa segunda forma facilita converter o seu cálculo em uma atualização do valor de Q, tal que Q(s, a) += alfa * (...)
#
# O valor alfa, chamado de taxa de aprendizado (learning rate em inglês), ajusta o peso de cada novo jogo na atualização
# dos valores de q(s, a). Alfa é um valor entre 0 e 1.
# Com valores baixos de alfa (próximos de 0) temos um peso maior do valor do estado e ações atuais, já com valores altos
# de alfa (próximos de 1), temos um peso alto da recompensa, com pouca influência do valor Q(s, a) do estado e ação atuais.
# Em geral alfa tem um valor mais próximo de 0. Na nossa implementação, alfa (TAXA_APRENDIZADO) é 0.2, mas pode ser modificado.
# O fator gama, que também é entre 0 e 1, indica quanto o valor da melhor ação nos estados seguintes é propagado no estado e
# ação atuais durante o treinamento. No nosso exemplo, gama (GAMA) tem valor 0.9, mas também pode ser modificado.
#

import numpy as np
import pickle

from collections import Counter
from copy import deepcopy
from pathlib import Path
from random import sample


# Valores para casa vazia, jogador X e jogador O
# X e O são potências de 2, ou seja, usam bits diferentes
# de forma que e a operação bitwise X & O seja igual a 0.
# Por exemplo, se X = 1 e O = 2 então, em binário:
# X = 001 e O = 010 então X & O == 0
VAZIA = 0
X = 1
O = 2

# Resultados, valores são referência à implementação
# dos leds do Arduino : VELHA -> V, XGANHOU -> X, OGANHOU -> O
# Ver: https://github.com/RobStelling/JogodaVelhaIRS
DEUVELHA = int('0b010101101', 2)
XGANHOU = int('0b101010101', 2)
OGANHOU = int('0b111101111', 2)

# Tamanho do tabuleiro, internamente ele é representado como um vetor de 9 posições e
# não uma matriz 3x3
LINHAS = 3
COLUNAS = 3
NUM_CASAS = LINHAS * COLUNAS

# Hiperparâmetros da política
# TAXA_EXPLORACAO: Frequência com que o política tenta alternativas não previstas
# TAXA_APRENDIZADO: (ALFA) é o peso entre o valor do estado atual e o valor da recompensa
# GAMA: Desconto dado à recompensa
# LIMITE_EXPLORACAO: Limite máximo entre a melhor ação e ações alternativas para um determinado estado
TAXA_EXPLORACAO = 0.3
TAXA_APRENDIZADO = 0.2
GAMA = 0.9
LIMITE_EXPLORACAO = 0.0

# Valor inicial de um estado e ação - q(s, a) = 0.0
INICIAL = 0.0
# Recompensas a propagar
# O jogador com X sempre inicia o jogo
VITORIA = 2.0
DERROTA = -1.0
VELHAX = 0.1
VELHAO = 1.0
LANCE = -0.05

# Atributos dos nomes dos arquivos de política ao serem salvos
# Ex: {PASTA_POLITICAS}/{PREFIXO_POLITICA}{nome_política}.{EXTENSAO_POLITICA}
PASTA_POLITICAS = "politicas"
PREFIXO_POLITICA = "p_"
EXTENSAO_POLITICA = "pjv"

# Esta implementação do jogo da velha com Reinforcement Learning (Q-learning) é feita utilizando 3 classes.
#
# jogodaVelha:
# Nessa classe temos a representação do tabuleiro, métodos para treinamento de políticas (entre duas políticas),
# métodos para partidas (entre pessoas e políticas ou mesmo entre políticas) e simulações (entre políticas)
#
# Maquina:
# Classe que representa uma política que será treinada
# 
# Humano:
# Classe que representa um jogador humano
#
# Humano e Maquina poderiam ser classe e subclasse porém creio que a implementação é mais simples e intuitiva
# com duas classes separadas
#
# O 'coração' do Q-learning acontece nos métodos recompensa da classe jogoDaVelha e no método
# propaga_recompensa da classe Máquina. A tabela Q é preservados no campo q da classe Máquina.
# O espaço de estados é finito mas não é necessário preencher a matriz q com 0 em todas as
# posições ao iniciar o treinamento. Um estado que não exista em q é assumido como 0.0 no momento
# da sua criação.
# O método recompensa (classe jogoDaVelha) indica os valores que devem ser propagados por todos os movimentos a partir de uma partida de
# treinamento que terminou.
# O método propaga_recompensa (classe Maquina) efetivamente propaga a premiação por todos os estados da partida que acabou de ser jogada
#
# Ao fim de cada partida de treinamento os valores de recompensa são propagados por todas as posições que ocorreram no jogo, dando
# prêmios correspondentes ao jogadores de X e O. As políticas para X e O são treinadas e preservadas em separado mas é possível,
# ao fim do treinamento, juntar as políticas e salvá-las como se fosse uma só (método combina_e_salva_politica), gerando assim uma
# política que joga como X e como O.
# Como curiosidade, se usarmos uma política treinada apenas como X para jogar como O teremos um comportamento randômico da política,
# porque como os estados do ponto de vista do jogador O não existem na política para X, todos os valores de Q para esses estados
# serão 0.

def gera_hash_tabuleiro(posicao):
    """Gera o hash de uma posição, para representar o estado de uma jogada"""
    return str(posicao)

def _num_casas_livres(tabuleiro):
    return sum(tabuleiro == VAZIA)

def _resultado_jogo(tabuleiro):
    """Verifica o resultado do jogo
    Verifica todas as linhas, colunas e as duas diagonais em busca de 3 marcas consecutivas
    Retorna quem ganhou ou velha, se o jogo tiver acabado, senão retorna None
    """
    # Verifica todas as linhas
    for i in range(0, NUM_CASAS, LINHAS):
        completou = np.bitwise_and.reduce(tabuleiro[i:i+LINHAS])
        if completou == X:
            return XGANHOU
        elif completou == O:
            return OGANHOU
    # Verifica todas as colunas
    for i in range(0, COLUNAS):
        completou = tabuleiro[i] & tabuleiro[i+COLUNAS] & tabuleiro[i+2*COLUNAS]
        if completou == X:
            return XGANHOU
        elif completou == O:
            return OGANHOU
    # Verifica as duas diagonais
    completou = (tabuleiro[0] & tabuleiro[4] & tabuleiro[8]) | \
                (tabuleiro[2] & tabuleiro[4] & tabuleiro[6])
    if completou == X:
        return XGANHOU
    elif completou == O:
        return OGANHOU
    # Verifica se o jogo acabou com velha (não há mais posições livres)
    if _num_casas_livres(tabuleiro) == 0:
        return DEUVELHA
    # Senão o jogo ainda não acabou
    return None

def existe_politica(politica):
    return (Path(PASTA_POLITICAS) / f"{PREFIXO_POLITICA}{politica}.{EXTENSAO_POLITICA}").exists()

def mostra_tabuleiro(tabuleiro):
    """Mostra a posição atual do tabuleiro de forma simples"""
    simbolo = {X: 'X', O: 'O', 0: ' '}
    for i in range(0, LINHAS):
        print('-------------')
        linha = "| "
        for j in range(0, COLUNAS):
            valor = tabuleiro[i*COLUNAS + j]
            linha += simbolo[valor] + " | "
        print(linha)
    print('-------------', flush=True)


class jogoDaVelha:
    """Classe para treinamento de políticas com reinforcement learning do jogo da velha
    A classe também pode ser usada para jogos entre humanos, entre humano e políticas e entre políticas
    A sua inicialização pede uma classe para cada um dos jogadores/políticas envolvidos
    Assume que o primeiro jogador inicia e representa o seus movimentos com X
    Os movimentos do adversário são representados com O
    """
    # O Tabuleiro é um vetor de 9 posições (3x3),
    # onde cada casa pode ser 0 (casa vazia), 1 (casa com X) ou 2 (casa com O)
    # Observe que 1 e 2 são potências de 2 (2^0 e 2^1), para permitir
    # operações bit a bit entre as casas do tabuleiro
    # Por exemplo:
    # - A operação E bit a bit (&) entre 3 casas só tem resultado X se TODAS as casas forem X. E só tem
    # resultado O se TODAS as casas forem O
    # Lembre-se: X sempre começa!!
    def __init__(self, jogador_X, jogador_O):
        """Inicialização da classe jogo da velha
        Recebe como parâmetros os jogadores X e O, das classes Máquina ou Humano
        """
        self.tabuleiro = np.zeros(NUM_CASAS, dtype=int)
        self.jogador = {X: jogador_X, O: jogador_O}
        self.terminou = False
        # X sempre começa
        self.vez = X

    def reinicia(self):
        """Reinicializa as condições do jogo, mantendo os mesmos jogadores
        Usado normalmente durante o treinamento da política
        """
        self.tabuleiro = np.zeros(NUM_CASAS, dtype=int)
        self.jogador[X].reinicia()
        self.jogador[O].reinicia()
        self.terminou = False
        self.vez = X

    def resultado(self):
        """Verifica o resultado do jogo
        Verifica todas as linhas, colunas e as duas diagonais em busca de 3 marcas consecutivas
        Retorna quem ganhou ou velha, se o jogo tiver acabado, senão retorna None
        Atualiza a flag jogoDaVelha.terminou se o jogo tiver terminado
        """
        estado = _resultado_jogo(self.tabuleiro)
        if estado is not None:
            self.terminou = True
        return estado

    def treinamento(self, rodadas=1000, verifica=100):
        """Executa o loop de treinamento
        Recebe como parâmetros opcionais o número de rodadas e de quantas em quantas rodadas o treinamento
        deve ser verificado
        Enquanto o treinamento é realizado as políticas para X e O são atualizadas com recompensas pré-determinadas
        """
        for rodada in range(rodadas):
            if rodada % verifica == 0:
                print(f"Rodadas: {rodada}")
            while True:
                alternativas = self.casas_livres()
                vez = self.vez
                jogada = self.jogador[vez].escolhe_jogada(alternativas, self.tabuleiro)
                self.jogada(jogada)
                # Se o jogo terminou (X venceu, O venceu ou velha)
                # propaga as recompensas pelos estados,
                # reinicia jogo e jogadores e volta ao loop de treinamento
                resultado = self.resultado()
                if resultado is not None:
                    self.recompensa(resultado, NUM_CASAS - len(alternativas) + 1)
                    self.reinicia()
                    break

        print(f"Treinamento finalizado: {rodadas} rodadas")

    def casas_livres(self):
        """Retorna uma lista de casas livres"""
        return [casa for casa, vazia in enumerate(self.tabuleiro == VAZIA) if vazia]

    def jogada(self, casa):
        """Faz uma jogada no jogo atual
        Ou seja, coloca um X ou O na casa que foi escolhida para jogar e
        troca o jogador da vez
        """
        troca = {X: O, O: X}
        self.tabuleiro[casa] = self.vez
        self.vez = troca[self.vez]

    def recompensa(self, resultado, total_jogadas):
        """Passa as recompensas para as políticas de acordo com o resultado do jogo
        As recompensas são propagadas nas políticas de cada jogador a partir dos seus lances nessa instância do jogo
        resultado: resultado do jogo (XGANHOU, OGANHOU, VELHA
        num_lances: número de lances da partida, importante para o desconto de número de lances, já que queremos que os
                    agentes descubram as vitórias mais curtas sempre que possível
        O jogador que fez o último movimento pode ser inferido pelo resultado, se XGANHOU, então foi X,
        se OGANHOU, então foi O, e se deu velha, foi X, já que X sempre começa. O número de movimentos é sempre ímpar depois
        de X jogar e par depois de O jogar.
        """
        if resultado == XGANHOU:
            self.jogador[X].propaga_recompensa(VITORIA + total_jogadas * LANCE)
            self.jogador[O].propaga_recompensa(DERROTA + (NUM_CASAS - total_jogadas) * LANCE)
        elif resultado == OGANHOU:
            self.jogador[X].propaga_recompensa(DERROTA + total_jogadas * LANCE)
            self.jogador[O].propaga_recompensa(VITORIA + (NUM_CASAS - total_jogadas) * LANCE)
        else: # Deu velha, X jogou 5 vezes, O jogou 4 vezes
            self.jogador[X].propaga_recompensa(VELHAX + NUM_CASAS * LANCE)
            self.jogador[O].propaga_recompensa(VELHAO + NUM_CASAS * LANCE)

    def partida(self, saida=True):
        """Jogo entre dois jogadores
        Podem ser duas políticas, uma política e um humano ou dois humanos
        Se saida == True então mostrará o tabuleiro com os lances efetuados
        Se pelo menos um dos jogadores for humano é recomendável que a flag saida seja True
        Reinicia as condições do jogo ao fim da partida
        """
        while not self.terminou:
            alternativas = self.casas_livres()
            vez = self.vez
            if self.jogador[vez].tipo == "Computador":
                jogada = self.jogador[vez].escolhe_jogada(alternativas, self.tabuleiro)
            else:
                jogada = self.jogador[vez].escolhe_jogada(alternativas)
            # O método self.jogada altera o jogador da vez (self.vez),
            # por isso o valor self.vez é guardado na variável vez
            self.jogada(jogada)
            if (saida):
                print(f'{self.jogador[vez].nome} jogou {jogada}')
                self.mostra_tabuleiro()

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
                alternativas = self.casas_livres()
                vez = self.vez
                jogada = self.jogador[vez].escolhe_jogada(alternativas, self.tabuleiro)
                self.jogada(jogada)
                resultado = self.resultado()
                if resultado is not None:
                    # Então o jogo acabou
                    totalizacao['Velha' if resultado == DEUVELHA else self.jogador[vez].nome]+=1
                    tabuleiros[gera_hash_tabuleiro(self.tabuleiro)]+=1
        
            self.reinicia()
        return totalizacao, tabuleiros
    
    def mostra_tabuleiro(self):
        """Mostra a posição atual do tabuleiro de forma simples"""
        mostra_tabuleiro(self.tabuleiro)

class Maquina():
    """Classe para representar uma política de jogo da velha
    Utilizado tanto no treinamento com reinforcement learning da política quanto em partidas contra outros adversários
    Para o treinamento espera como entrada uma taxa de exploração, uma taxa de aprendizado e um fator de desconto GAMA
    """
    def __init__(self, nome,
                 taxa_exploracao=TAXA_EXPLORACAO,
                 taxa_aprendizado=TAXA_APRENDIZADO,
                 gama=GAMA,
                 limite_exploracao=LIMITE_EXPLORACAO,
                 depuracao=False):
        """Intancia o objeto Maquina
        Nome: usado para salvar/recuperar as políticas e também para representar o jogador
        Tipo: indica se é uma  política ou um humano
        Estados: lista os estados do jogo atual
        q: Valores q
        Taxa_aprendizado: peso utilizado na propagação das recompensas
        Taxa_exploracao: percentual de exploracao de alternativas fora da política atual
        Gama: desconto da recompensa a ser propagada
        """
        self.nome = nome
        self.tipo = "Computador"
        self.estados = []
        self.q = {}
        self.taxa_aprendizado = taxa_aprendizado
        self.taxa_exploracao = taxa_exploracao
        self.gama = gama
        self.limite_exploracao = limite_exploracao
        self.depuracao = depuracao

    def reinicia(self):
        """Reinicia a política para a próxima partida
        Apenas descarta os estados do jogo atual"""
        self.estados = []

    def escolhe_jogada(self, casas_livres, tabuleiro):
        """Retorna a jogada a fazer, em função da política até o momento
        Pode retornar uma jogada randômica, entre as jogadas disponíveis, de
        acordo com a taxa de exploração
        Durante uma partida a taxa de exploração deve ser 0
        """
        copia_tabuleiro = tabuleiro.copy()
        hash_tabuleiro = gera_hash_tabuleiro(copia_tabuleiro)
        # Se o jogador ainda não "viu" a posição atual então insere em q
        # e inicializa q[hash_tabuleiro][jogada] = 0.0, para todas a jogadas
        # possíveis no tabuleiro atual
        if not hash_tabuleiro in self.q:
            self.q[hash_tabuleiro] = {casa: INICIAL for casa in casas_livres}
        
        if np.random.uniform(0, 1) < self.taxa_exploracao:
            # Executa ação randômica de acordo com a taxa de exploração
            # se a taxa de exploração for 0.0 então todas as ações virão da
            # política
            jogada = np.random.choice(casas_livres)
        else:
            # Escolhe uma das alternativas que maximizam q
            # jogada_max é uma das jogadas com q maior (pode haver mais de uma com o mesmo valor máximo)
            # valor_max é o valor dessa jogada
            # alternativas são todas as jogadas com esse valor máximo
            jogada_max = max(self.q[hash_tabuleiro], key=self.q[hash_tabuleiro].get)
            valor_max = self.q[hash_tabuleiro][jogada_max]
            alternativas = [casa for casa in self.q[hash_tabuleiro] if (valor_max-self.q[hash_tabuleiro][casa]) <= self.limite_exploracao]
            # Escolhe randomicamente uma das jogadas, se houver mais de uma alternativa
            jogada = sample(alternativas, 1)[0]
            if self.depuracao:
                print(hash_tabuleiro, jogada_max, valor_max, self.q[hash_tabuleiro])            
        self.acrescenta_estado(copia_tabuleiro, jogada)
        return jogada
    
    def acrescenta_estado(self, tabuleiro, jogada):
        """Acrescenta um estado na lista, usado durante o treinamento
        para representar os lances jogados durante a partida
        """
        hash_tabuleiro = gera_hash_tabuleiro(tabuleiro)
        self.estados.append({'posicao': hash_tabuleiro, 'jogada': jogada})

    def maxq(self, estado):
        """Retorna o valor mais alto de q entre as alternativas de ações em um dado estado"""
        max_index = max(self.q[estado], key=self.q[estado].get)
        return self.q[estado][max_index]

    def propaga_recompensa(self, recompensa):
        """Propaga a recompensa pelos estados do jogo atual
        É o principal processo do reinforcement learning
        Perceba que os estados são percorridos de trás para frente, ou seja,
        dos ultimos movimentos para os primeiros, e que o valor da recompensa
        é reduzido pelo fator de desconto gama
        """
        estados = self.estados.copy()
        # Há duas formas (com o mesmo resultado) para o cálculo no novo Q(s, a)
        # estamos usando a forma:
        # Novo Q(s, a) = Q(s, a) + alfa * [R(s, a) + gama * maxQ'(s', a') - Q(s, a)]
        for i in range(len(estados)-1):
            s = estados[i]['posicao']
            a = estados[i]['jogada']
            s_linha = estados[i+1]['posicao']
            self.q[s][a] += self.taxa_aprendizado * (recompensa + self.gama * self.maxq(s_linha) - self.q[s][a])
        s = estados[i+1]['posicao']
        a = estados[i+1]['jogada']
        self.q[s][a] += self.taxa_aprendizado * (recompensa + self.gama * recompensa - self.q[s][a])

    def salva_politica(self, prefixo=PREFIXO_POLITICA):
        """Salva uma política para uso futuro"""
        pasta = Path(f'./{PASTA_POLITICAS}')
        if not pasta.exists():
            pasta.mkdir()
        if pasta.is_dir():
            nome_arquivo = pasta / f'{prefixo}{str(self.nome)}.{EXTENSAO_POLITICA}' 
            with open(nome_arquivo, 'wb' ) as arquivo:
                pickle.dump(self.q, arquivo)
        else:
            raise ValueError(f"Não consigo criar arquivos em {pasta}")

    def carrega_politica(self, politica, prefixo=PREFIXO_POLITICA):
        """Carrega uma política para jogar ou continuar um treinamento"""
        pasta = Path(f'./{PASTA_POLITICAS}')
        nome_arquivo = pasta / f'{prefixo}{politica}.{EXTENSAO_POLITICA}'
        if nome_arquivo.exists():
            with open(nome_arquivo, 'rb') as arquivo:
                self.q = pickle.load(arquivo)
        else:
            raise ValueError(f"Política {politica} não existe!")

    def combina_e_salva_politica(self, politica2, nome, prefixo=PREFIXO_POLITICA):
        """Combina duas políticas em uma e salva tabela q
        O objetivo é combinar duas políticas, uma para X e uma para O em uma só política, já que as
        tuplas (hashTabuleiro, valor) são mutualmente excludentes nas políticas para X e O
        """
        politica = deepcopy(self)
        politica.nome = nome
        politica.q = {**self.q, **politica2.q}
        pasta = Path(f'./{PASTA_POLITICAS}')
        if not pasta.exists():
            pasta.mkdir()
        if pasta.is_dir():
            nome_arquivo = pasta / f'{prefixo}{nome}.{EXTENSAO_POLITICA}'
            with open(nome_arquivo, 'wb') as arquivo:
                pickle.dump(politica.q, arquivo)
        else:
            raise ValueError(f"Não consigo criar arquivos em {pasta}")

class Humano:
    """Classe que representa as ações de um jogador humano"""
    def __init__(self, nome):
        self.nome = nome
        self.tipo = "Humano"
    
    def escolhe_jogada(self, casas_livres):
        """Pergunta qual o lance a fazer
        Recomendável que a partida seja chamada com saida == True
        se esta envolver pelo menos um jogador humano
        """
        while True:
            print(f"Qual a sua jogada, {self.nome}? ", end='', flush=True)
            acao = int(input())
            #print(f"{self.nome} jogou {acao}")
            if acao in casas_livres:
                return acao

    def reinicia(self):
        pass
    

if __name__ == "__main__":
    # Exemplo de treinamento e jogo
    # Define duas políticas que serão treinadas
    politicaX = Maquina("X")
    politicaO = Maquina("O")

    treinamento = jogoDaVelha(politicaX, politicaO)
    print("Treinando...")
    treinamento.treinamento(10000)
    # Salva as políticas geradas
    politicaX.salva_politica()
    politicaO.salva_politica()

    # Carrega uma politica salva e joga contra um humano
    politicaX = Maquina("Computador", taxa_exploracao=0.0)
    politicaX.carrega_politica("X")
    humano = Humano("Walter")
    jogo = jogoDaVelha(politicaX, humano)
    jogo.partida(saida=True)