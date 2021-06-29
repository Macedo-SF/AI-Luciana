#A suposição de Markov é a suposição de que o estado atual depende de um número finito de estados anteriores.
#Ela serve para limitar a quantidade de estados anteriores, facilitando a tarefa de predizer o próximo estado.
#O modelo pode usar apenas o estado anterior/o último evento para predizer o proximo estado/evento.

#O modelo de transição é a distribuição das probabilidades dos eventos de acordo com o evento atual:
#   X Y (+1)
# X a b
# Y c d   //c=1-d, a=1-b

#A cadeia de Markov é um número de observações obtido através de um modelo de transição e um estado atual:
# X -> X -> Y -> X -> X -> X -> Y -> Y

from pomegranate import *
from collections import Counter

# Define starting probabilities
start = DiscreteDistribution({
    "sol": 0.5,
    "chuva": 0.5
})

# Define transition model
transitions = ConditionalProbabilityTable([
    ["sol", "sol", 0.9],
    ["sol", "chuva", 0.1],
    ["chuva", "sol", 0.6],
    ["chuva", "chuva", 0.4]
], [start])

# Create Markov chain
model = MarkovChain([start, transitions])

# Sample 50 states from chain
wModel=(model.sample(100))
print(wModel)
print(Counter(wModel))

# Fun little test
sCoin = DiscreteDistribution({
    "Cara": 0.5,
    "Coroa": 0.5
})

tCoin = ConditionalProbabilityTable([
    ["Cara", "Cara", 0.5],
    ["Cara", "Coroa", 0.5],
    ["Coroa", "Cara", 0.5],
    ["Coroa", "Coroa", 0.5]
], [sCoin])

mCoin = MarkovChain([sCoin, tCoin])

print(Counter(mCoin.sample(100000)))