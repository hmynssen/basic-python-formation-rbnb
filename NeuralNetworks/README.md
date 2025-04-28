### 5\. Introdução a PyTorch para Redes Neurais

  * **Instalação:** A instalação do PyTorch depende da sua configuração (sistema operacional, CUDA para uso de GPU, etc.). Siga as instruções detalhadas no [site oficial do PyTorch](https://pytorch.org/). Geralmente, envolve copiar um comando específico para o seu ambiente. Por exemplo:
    ```bash
    # Exemplo para Linux, CUDA 11.8
    pip install torch torchvision torchaudio --index-url [https://download.pytorch.org/whl/cu118](https://download.pytorch.org/whl/cu118)
    ```
  * **Conceitos Básicos:** PyTorch é um framework de aprendizado de máquina de código aberto que fornece estruturas de dados tensoriais e rotinas otimizadas para construir e treinar redes neurais.
      * **Tensores:** São semelhantes aos arrays NumPy, mas podem ser executados em GPUs para aceleração.
      * **Módulos (`nn.Module`):** Blocos de construção de redes neurais. Contêm camadas e outras operações.
      * **Funções de Perda:** Medem o quão bem o modelo está performando.
      * **Otimizadores:** Algoritmos para ajustar os pesos da rede durante o treinamento.
      * **Backpropagation:** Algoritmo para calcular os gradientes da função de perda em relação aos pesos da rede.
  * **Exemplo Simples de Rede Neural:**
    ```python
    import torch
    import torch.nn as nn
    import torch.optim as optim

    # Definindo a rede neural
    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.fc1 = nn.Linear(10, 5) # Camada totalmente conectada: 10 entradas, 5 saídas
            self.fc2 = nn.Linear(5, 1)  # Camada totalmente conectada: 5 entradas, 1 saída
            self.sigmoid = nn.Sigmoid()

        def forward(self, x):
            x = self.fc1(x)
            x = self.sigmoid(x)
            x = self.fc2(x)
            return x

    # Criando uma instância da rede
    net = Net()

    # Criando dados de exemplo
    inputs = torch.randn(1, 10) # Um exemplo com 10 features
    targets = torch.randn(1, 1)

    # Definindo a função de perda e o otimizador
    criterion = nn.MSELoss()
    optimizer = optim.SGD(net.parameters(), lr=0.01)

    # Treinamento (um único passo)
    optimizer.zero_grad() # Zerar os gradientes
    outputs = net(inputs)
    loss = criterion(outputs, targets)
    loss.backward() # Calcular os gradientes
    optimizer.step() # Atualizar os pesos

    print(f"Perda após um passo: {loss.item()}")
    ```
