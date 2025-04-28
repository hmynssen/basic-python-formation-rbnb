## Python para Análise de Dados Biológicos

Esta seção visa introduzir o uso de Python para tarefas comuns em biologia, como manipulação e visualização de dados experimentais.

### 1\. Análise de Dados Biológicos com Pandas e Matplotlib/Seaborn

Após ter uma base sólida nos fundamentos de Python (tipos de dados, loops, condicionais, funções), podemos começar a usar bibliotecas poderosas para análise de dados biológicos.

  * **Leitura de Dados em Formato .csv com Pandas:**

      * Muitos dados biológicos são armazenados em arquivos `.csv` (Comma Separated Values), onde os valores são separados por vírgulas e as linhas representam observações ou amostras.
      * O Pandas facilita a leitura desses arquivos para um objeto chamado DataFrame, que se assemelha a uma planilha.

    ```python
    import pandas as pd

    # Suponha que temos um arquivo chamado 'dados_biologicos.csv'
    # com colunas como 'ID', 'Genótipo', 'Expressão_Gênica', 'Condição'
    try:
        data = pd.read_csv('dados_biologicos.csv')
        print("Dados lidos com sucesso:")
        print(data.head()) # Exibe as primeiras 5 linhas do DataFrame
    except FileNotFoundError:
        print("Arquivo 'dados_biologicos.csv' não encontrado.")
    ```

      * **Exemplo de um arquivo `dados_biologicos.csv`:**
        ```csv
        ID,Genótipo,Expressão_Gênica,Condição
        A1,WT,2.5,Controle
        A2,MUT,8.1,Tratamento
        B1,WT,3.0,Controle
        B2,MUT,9.5,Tratamento
        C1,WT,2.8,Controle
        ```
      * **Imagem Ilustrativa:**
          * [Link para uma imagem ilustrativa de um DataFrame do Pandas sendo impresso no console](https://www.google.com/search?q=https://pandas.pydata.org/docs/getting_started/intro_tutorials/images/01_table_dataframe.png) (Fonte: Documentação oficial do Pandas)

  * **Filtragem de Dados:**

      * Podemos selecionar subconjuntos dos dados com base em condições específicas.

    ```python
    # Selecionando apenas as observações com 'Genótipo' igual a 'MUT'
    mutantes = data[data['Genótipo'] == 'MUT']
    print("\nObservações com genótipo MUT:")
    print(mutantes.head())

    # Selecionando observações com 'Expressão_Gênica' maior que 5
    alta_expressao = data[data['Expressão_Gênica'] > 5]
    print("\nObservações com alta expressão gênica:")
    print(alta_expressao.head())

    # Combinando condições com os operadores lógicos '&' (e) e '|' (ou)
    tratamento_alta_expressao = data[(data['Condição'] == 'Tratamento') & (data['Expressão_Gênica'] > 7)]
    print("\nObservações em tratamento com alta expressão gênica:")
    print(tratamento_alta_expressao)
    ```

  * **Visualização de Dados com Matplotlib e Seaborn:**

      * **Boxplot:** Útil para comparar a distribuição de uma variável entre diferentes grupos.

        ```python
        import matplotlib.pyplot as plt
        import seaborn as sns

        sns.boxplot(x='Genótipo', y='Expressão_Gênica', data=data)
        plt.title('Comparação da Expressão Gênica entre Genótipos')
        plt.xlabel('Genótipo')
        plt.ylabel('Nível de Expressão Gênica')
        plt.show()
        ```

          * **Imagem Ilustrativa:**
              * [Link para uma imagem de um boxplot comparando grupos](https://www.google.com/search?q=https://seaborn.pydata.org/_images/seaborn-boxplot-1.png) (Fonte: Documentação oficial do Seaborn)

      * **Histograma:** Mostra a distribuição de frequência de uma única variável contínua.

        ```python
        plt.hist(data['Expressão_Gênica'], bins=10, edgecolor='black')
        plt.title('Distribuição da Expressão Gênica')
        plt.xlabel('Nível de Expressão Gênica')
        plt.ylabel('Frequência')
        plt.show()
        ```

          * **Imagem Ilustrativa:**
              * [Link para uma imagem de um histograma](https://www.google.com/search?q=https://matplotlib.org/_images/sphx_glr_hist_001.png) (Fonte: Documentação oficial do Matplotlib)

      * **Gráfico de Dispersão (Scatter Plot):** Útil para visualizar a relação entre duas variáveis.

        ```python
        plt.scatter(data['ID'], data['Expressão_Gênica'])
        plt.title('Expressão Gênica por ID da Amostra')
        plt.xlabel('ID da Amostra')
        plt.ylabel('Nível de Expressão Gênica')
        plt.show()
        ```

          * **Imagem Ilustrativa:**
              * [Link para uma imagem de um gráfico de dispersão](https://www.google.com/search?q=https://matplotlib.org/_images/sphx_glr_scatter_001.png) (Fonte: Documentação oficial do Matplotlib)

  * **Enfeitando os Gráficos:**

      * **Legendas:**
        ```python
        plt.scatter(data[data['Condição'] == 'Controle']['ID'], data[data['Condição'] == 'Controle']['Expressão_Gênica'], label='Controle', color='blue')
        plt.scatter(data[data['Condição'] == 'Tratamento']['ID'], data[data['Condição'] == 'Tratamento']['Expressão_Gênica'], label='Tratamento', color='red')
        plt.xlabel('ID da Amostra')
        plt.ylabel('Nível de Expressão Gênica')
        plt.title('Expressão Gênica por Condição')
        plt.legend()
        plt.show()
        ```
      * **Textos:**
        ```python
        plt.hist(data['Expressão_Gênica'], bins=5, edgecolor='black')
        plt.xlabel('Nível de Expressão Gênica')
        plt.ylabel('Frequência')
        plt.title('Distribuição da Expressão Gênica')
        plt.text(6, 3, 'Pico de Expressão', fontsize=10, color='green') # (x, y, texto)
        plt.show()
        ```
      * **Alterando Cores e Temas:**
        ```python
        # Alterando cores diretamente nos comandos de plotagem (como no exemplo da legenda)

        # Usando temas do Seaborn
        sns.set_theme(style="whitegrid")
        sns.boxplot(x='Genótipo', y='Expressão_Gênica', data=data)
        plt.title('Expressão Gênica entre Genótipos (Tema Whitegrid)')
        plt.xlabel('Genótipo')
        plt.ylabel('Nível de Expressão Gênica')
        plt.show()

        plt.style.use('seaborn-v0_8-darkgrid') # Outro tema do Matplotlib
        plt.hist(data['Expressão_Gênica'], bins=5, edgecolor='black')
        plt.xlabel('Nível de Expressão Gênica')
        plt.ylabel('Frequência')
        plt.title('Distribuição da Expressão Gênica (Tema Darkgrid)')
        plt.show()
        ```

**Próximos Passos para Biólogos:**

  * **Explorar diferentes tipos de gráficos:** Gráficos de barras, gráficos de linha para séries temporais (se aplicável), gráficos de pizza (com cautela).
  * **Aprender a salvar os gráficos:** Usando `plt.savefig('nome_do_arquivo.png')`.
  * **Analisar dados reais:** Buscar conjuntos de dados biológicos públicos ou usar seus próprios dados experimentais para praticar.
  * **Integrar com outras bibliotecas:** Introduzir o SciPy para análise estatística básica (testes t, ANOVA, etc.) após se sentir confortável com Pandas e Matplotlib/Seaborn.
