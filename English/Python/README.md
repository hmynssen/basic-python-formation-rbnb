Este bloco tem como objetivo fornecer uma base sólida em Python, Git e nas principais bibliotecas para computação científica, preparando o terreno para tópicos mais avançados.

### 1\. O Básico da Linguagem Python

  * **Tipagem:** Python é uma linguagem de tipagem dinâmica e forte. Isso significa que você não precisa declarar o tipo de uma variável explicitamente, e o interpretador verifica os tipos em tempo de execução, impedindo operações inadequadas entre tipos diferentes.
  * **Variáveis:** Usadas para armazenar dados. Os nomes de variáveis devem ser descritivos e seguir algumas regras (começar com letra ou sublinhado, não conter espaços ou caracteres especiais, etc.).
    ```python
    nome = "João"
    idade = 30
    altura = 1.75
    eh_estudante = True
    ```
  * **Tipos de Dados Fundamentais:**
      * Inteiros (`int`): `10`, `-5`, `0`
      * Ponto flutuante (`float`): `3.14`, `-0.001`, `2.0`
      * Strings (`str`): `"Olá"`, `'Python'`, `"123"`
      * Booleanos (`bool`): `True`, `False`
      * Listas (`list`): `[1, 2, 3]`, `['a', 'b', 'c']`, `[1, 'hello', 3.14]` (mutáveis, ordenadas, permitem duplicatas)
      * Tuplas (`tuple`): `(1, 2, 3)`, `('a', 'b', 'c')` (imutáveis, ordenadas, permitem duplicatas)
      * Dicionários (`dict`): `{'nome': 'Maria', 'idade': 25}` (coleções de pares chave-valor, não ordenadas a partir do Python 3.7)
      * Conjuntos (`set`): `{1, 2, 3}`, `{'a', 'b', 'c'}` (coleções não ordenadas de elementos únicos)
  * **Operadores:**
      * **Aritméticos:** `+` (adição), `-` (subtração), `*` (multiplicação), `/` (divisão), `//` (divisão inteira), `%` (módulo - resto da divisão), `**` (potenciação).
      * **Comparação:** `==` (igual a), `!=` (diferente de), `>` (maior que), `<` (menor que), `>=` (maior ou igual a), `<=` (menor ou igual a).
      * **Lógicos:** `and` (e), `or` (ou), `not` (não).
      * **Atribuição:** `=`, `+=`, `-=`, `*=`, `/=`, etc.
  * **Estruturas de Controle de Fluxo:**
      * **Condicionais (`if`, `elif`, `else`):** Permitem executar diferentes blocos de código dependendo de uma condição.
        ```python
        idade = 18
        if idade >= 18:
            print("Você é maior de idade.")
        elif idade > 16:
            print("Você é quase maior de idade.")
        else:
            print("Você é menor de idade.")
        ```
      * **Loops (`for` e `while`):** Permitem repetir um bloco de código várias vezes.
          * `for`: Itera sobre uma sequência (lista, tupla, string, range, etc.).
            ```python
            for i in range(5):  # Itera de 0 a 4
                print(i)

            nomes = ["Alice", "Bob", "Charlie"]
            for nome in nomes:
                print(nome)
            ```
          * `while`: Executa um bloco de código enquanto uma condição for verdadeira.
            ```python
            contador = 0
            while contador < 5:
                print(contador)
                contador += 1
            ```
  * **Funções:** Blocos de código reutilizáveis que realizam uma tarefa específica.
    ```python
    def saudacao(nome):
        """Esta função cumprimenta a pessoa passada como parâmetro."""
        print(f"Olá, {nome}!")

    saudacao("Ana")  # Saída: Olá, Ana!

    def soma(a, b):
        """Esta função retorna a soma de dois números."""
        return a + b

    resultado = soma(5, 3)
    print(resultado)  # Saída: 8
    ```

### 2\. Primeiros Passos com NumPy e Matplotlib

  * **Instalação:** Use o `pip` para instalar as bibliotecas. Abra o terminal/prompt de comando e execute:
    ```bash
    pip install numpy matplotlib
    ```
  * **NumPy (Numerical Python):** Biblioteca fundamental para computação numérica em Python. Fornece suporte para arrays multidimensionais e diversas funções matemáticas.
    ```python
    import numpy as np

    # Criando arrays NumPy
    a = np.array([1, 2, 3, 4, 5])
    b = np.array([[1, 2], [3, 4]])

    print(a)
    print(b)

    # Operações com arrays
    print(a + 1)
    print(b * 2)
    print(np.sin(a))

    # Funções úteis
    print(np.mean(a))
    print(np.dot([1, 2], [3, 4])) # Produto escalar
    ```
  * **Matplotlib:** Biblioteca para criação de gráficos e visualizações de dados em Python. O módulo `pyplot` é o mais comumente usado.
    ```python
    import matplotlib.pyplot as plt
    import numpy as np

    # Dados para o gráfico
    x = np.linspace(0, 2 * np.pi, 100)
    y = np.sin(x)

    # Criando o gráfico
    plt.plot(x, y)
    plt.xlabel("Ângulo (radianos)")
    plt.ylabel("Seno")
    plt.title("Gráfico da função Seno")
    plt.grid(True)
    plt.show()

    # Outro exemplo: gráfico de dispersão
    x2 = np.random.rand(50)
    y2 = np.random.rand(50)
    plt.scatter(x2, y2)
    plt.xlabel("Variável X")
    plt.ylabel("Variável Y")
    plt.title("Gráfico de Dispersão")
    plt.show()
    ```

### 3\. Introdução a Pandas e SciPy

  * **Instalação:**
    ```bash
    pip install pandas scipy
    ```
  * **Pandas:** Biblioteca para análise e manipulação de dados. A estrutura principal é o DataFrame, uma tabela com rótulos de linhas e colunas.
    ```python
    import pandas as pd

    # Criando um DataFrame
    data = {'Nome': ['Alice', 'Bob', 'Charlie'],
            'Idade': [25, 30, 28],
            'Pontuação': [85, 92, 78]}
    df = pd.DataFrame(data)

    print(df)

    # Acessando dados
    print(df['Nome'])
    print(df.loc[0]) # Primeira linha

    # Estatísticas descritivas
    print(df.describe())
    ```
  * **SciPy (Scientific Python):** Biblioteca que fornece diversas funções para matemática, ciência e engenharia, incluindo otimização, álgebra linear, integração, interpolação, transformada de Fourier, processamento de sinais, estatística e muito mais.
    ```python
    import scipy.optimize as opt
    from scipy.integrate import solve_ivp

    # Otimização: encontrar o mínimo de uma função
    def f(x):
        return x**2 - 5*x + 6

    resultado_otim = opt.minimize_scalar(f)
    print(resultado_otim)

    # Resolução de uma equação diferencial ordinária (exemplo simples)
    def dy_dt(t, y):
        return -0.5 * y

    solucao = solve_ivp(dy_dt, [0, 10], [2]) # Intervalo de tempo [0, 10], condição inicial y(0) = 2
    print(solucao.y[0])
    ```
