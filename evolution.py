import random
import numpy as np
from sklearn.datasets import make_moons
import pennylane as qp
import torch

def mutate(child,numberOfLayers):
    #changes random neural network either from classic to quantum or otherwise
    child[random.randint(0,numberOfLayers-1)]^=1

def connect(genome1,genome2,numberOfLayers):
    #connects two genomes from random point
    crossPoint=random.randint(0,numberOfLayers-1)
    child=genome1.copy()
    for i in range(crossPoint,numberOfLayers):
        child[i]=genome2[i]
    return child

def makeGenome(numberOfLayers):
    #makes random genome
    pole=[random.choice([0, 1]) for _ in range(numberOfLayers)]
    return pole

def buildModel(genome,qnode,weight_shapes):
    layers=[]
    layers.append(torch.nn.BatchNorm1d(4))  #normalizes inputs across classical and quantum layers--Claude recommended this--
    for n in genome:
        if n==0:
            layers.append(torch.nn.Linear(4, 4))
            layers.append(torch.nn.ReLU())  
        else:
            layers.append(qp.qnn.TorchLayer(qnode, weight_shapes))
    layers.insert(0,torch.nn.Linear(2,4))
    layers.append(torch.nn.Linear(4,2))
    model=torch.nn.Sequential(*layers)
    return model

def fitnessFunction(genome,qnode,weight_shapes,batches,data_loader,X,y):
    model=buildModel(genome,qnode,weight_shapes)
    #this whole section was taken https://pennylane.ai/demos/tutorial_qnn_module_torch 
    opt = torch.optim.Adam(model.parameters(), lr=0.01) #more stable than SGD for hybrid models--Claude recommended to change this--
    loss = torch.nn.CrossEntropyLoss() 

    epochs=35

    for epoch in range(epochs):

        running_loss=0

        for xs,ys in data_loader:
            opt.zero_grad()

            loss_evaluated = loss(model(xs), ys)
            loss_evaluated.backward()

            opt.step()

            running_loss+=loss_evaluated

        avg_loss=running_loss/batches

    y_pred=model(X)
    predictions=torch.argmax(y_pred,axis=1).detach().numpy()

    correct=[1 if p==p_true else 0 for p,p_true in zip(predictions,y)]
    accuracy=sum(correct)/len(correct)
    return accuracy


torch.manual_seed(42)
np.random.seed(42)

results={}

X, y = make_moons(n_samples=200, noise=0.1)

n_qubits = 4
dev = qp.device("default.qubit", wires=n_qubits)

@qp.qnode(dev)
def qnode(inputs, weights):
    qp.AngleEmbedding(inputs, wires=range(n_qubits))
    qp.BasicEntanglerLayers(weights, wires=range(n_qubits))
    return [qp.expval(qp.PauliZ(wires=i)) for i in range(n_qubits)]

n_layers=6
weight_shapes={"weights":(n_layers,n_qubits)}

numberOfLayers=6    #number of neural network layers

pole=[0 for _ in range(numberOfLayers)] #without any quantum layers

genome=makeGenome(numberOfLayers)

X = torch.tensor(X, requires_grad=True).float()

batch_size = 5
batches = 200 // batch_size

data_loader = torch.utils.data.DataLoader(
    list(zip(X,torch.tensor(y).long())), batch_size=5, shuffle=True, drop_last=True
)

results[tuple(pole)]=fitnessFunction(pole,qnode,weight_shapes,batches,data_loader,X,y) #training normal neural network

attempt=0
#Creating two individuals
for i in range(2):
    #Rating them
    results[tuple(genome)]=fitnessFunction(genome,qnode,weight_shapes,batches,data_loader,X,y)
    while tuple(genome) in results:
        genome=makeGenome(numberOfLayers)
#Crossover
for j in range(2):
    #Sorting them and taking 2 best
    best=sorted(results,key=results.get,reverse=True)[:2]
    child1=connect(list(best[0]),list(best[1]),numberOfLayers)
    while tuple(child1) in results and attempt!=10:
        mutate(child1,numberOfLayers)  
        attempt=attempt+1
    attempt=0
    results[tuple(child1)]=fitnessFunction(child1,qnode,weight_shapes,batches,data_loader,X,y)


#print
sorted_results = sorted(results.items(), key=lambda x: x[1])
for genome, accuracy in sorted_results:
    if accuracy == 0.5:
        label = "⚠ UNTRAINED"
    elif accuracy == max(results.values()):
        label = "★ BEST"
    elif accuracy == min(v for v in results.values() if v != 0.5):
        label = "✗ WORST"
    else:
        label = ""
    print(f"{list(genome)} → {accuracy:.3f} {label}")