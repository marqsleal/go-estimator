# Exemplos de uso

Esta pasta reúne fluxos completos e executáveis da biblioteca. Cada exemplo explica a sequência de preparação dos dados, ajuste e avaliação e pode ser executado a partir da raiz do módulo.

```sh
go run ./examples/regression
go run ./examples/classification
go run ./examples/cross_validation
```

## Fluxos disponíveis

- [Regressão](regression/main.go): divide os dados, ajusta `StandardScaler` e `LinearRegression` em um pipeline, prevê no conjunto de teste e calcula MSE e R².
- [Classificação binária](classification/main.go): ajusta scaler e `LogisticRegression`, prevê rótulos e probabilidades e calcula acurácia, precisão, recall e matriz de confusão.
- [Validação cruzada](cross_validation/main.go): avalia pipelines de regressão e classificação com folds independentes e semente fixa.

## Manutenção

Ao implementar uma feature, revise este índice e os exemplos existentes. Atualize o fluxo relacionado para mostrar a nova API ou comportamento. Se a feature introduzir um fluxo distinto, acrescente um exemplo executável com explicação aqui. Os exemplos devem compilar e permanecer compatíveis com as APIs públicas atuais. Registre no mesmo change a atualização deste índice e da documentação da feature.
