# Exemplos de uso

Esta pasta apresenta três experimentos completos da biblioteca. Cada programa constrói dados pequenos em memória, prepara as partições, ajusta um estimador e calcula resultados. Não é necessário baixar datasets ou instalar dependências.

## Reprodutibilidade

Cada programa declara sua própria constante `randomState` perto do início do arquivo. A seed fixa a partição ou a ordem dos folds, permitindo repetir a execução com a mesma versão da biblioteca e os mesmos dados. Os exemplos não compartilham uma constante entre arquivos porque cada fluxo representa um experimento independente e pode ser alterado sem mudar os demais.

A seed do Go garante repetibilidade dentro desta implementação. Ela não promete gerar os mesmos índices que NumPy ou scikit-learn para o mesmo número. Para reproduzir resultados de outro sistema, compare ou persista os índices usados, além dos dados e parâmetros.

## Fluxos disponíveis

### Leitura e gravação CSV

[tabular_csv/main.go](tabular_csv/main.go) carrega uma matriz numérica CSV de uma string local, executa um pipeline de regressão e grava as previsões em CSV e JSON. As APIs recebem `io.Reader` e `io.Writer`, então o chamador mantém controle dos arquivos e demais recursos. JSON usa uma matriz de arrays numéricos, por exemplo `[[1,2],[3,4]]`.

### Tabela nomeada

[tabular_table/main.go](tabular_table/main.go) carrega CSV com cabeçalho, seleciona características e alvo pelo nome, treina regressão e grava previsões em CSV e JSON nomeado. Recebe um arquivo CSV local com colunas area, rooms e price e grava uma previsão por linha no arquivo de saída.

### Regressão

[regression/main.go](regression/main.go) divide X e y em treino e teste usando `TrainTestSplit`. O pipeline ajusta `StandardScaler` e `LinearRegression` somente no treino. Em seguida, prevê no teste e calcula MSE e R². Esse fluxo demonstra a separação entre ajuste e avaliação e evita que o scaler aprenda estatísticas do conjunto de teste.

### Classificação binária

[classification/main.go](classification/main.go) divide as linhas com `TrainTestSplit` e aplica os índices retornados aos rótulos inteiros. O split atual recebe alvos `float64`, então o exemplo converte os rótulos apenas para obter os mesmos índices e preserva `[]int` no ajuste e nas métricas. O pipeline ajusta `StandardScaler` e `LogisticRegression`, depois calcula previsões, probabilidades, acurácia, precisão, recall e matriz de confusão no teste. A precisão e o recall usam explicitamente a classe `1` como positiva.

### Validação cruzada

[cross_validation/main.go](cross_validation/main.go) calcula R² para regressão e acurácia para classificação. Cada chamada usa quatro folds embaralhados. As fábricas criam pipelines novos em cada fold, e o scaler de cada pipeline aprende somente com as linhas de treino daquele fold. Os resultados são apresentados por fold, sem agregação, para que a variação entre partições fique visível.

## Executar

Execute estes comandos na raiz do repositório. Cada um imprime as métricas ou pontuações produzidas pelo fluxo.

```sh
go run ./examples/regression
go run ./examples/classification
go run ./examples/cross_validation
go run ./examples/tabular_csv
go run ./examples/tabular_table entrada.csv previsoes.csv
```

## Manutenção

Ao implementar uma feature, revise este índice e os exemplos existentes. Atualize o fluxo relacionado para mostrar a nova API ou comportamento. Se a feature introduzir um fluxo distinto, acrescente um exemplo executável com explicação aqui. Os exemplos devem compilar e permanecer compatíveis com as APIs públicas atuais. Registre no mesmo change a atualização deste índice e da documentação da feature.
