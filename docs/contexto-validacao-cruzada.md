# Entrega · validação cruzada

Base analisada: commit `053b430`. A biblioteca já oferece regressão linear, classificação logística binária, scaler, métricas, pipelines e persistência JSON de regressão. A etapa 6 do roteiro já tem API e exemplos executáveis.

## Ponto de partida

[KFold](../model_selection/kfold.go) produz índices. [CrossValScore e CrossValClassificationScore](../model_selection/cross_validation.go) chamam a fábrica em cada fold, ajustam o estimador no treino e avaliam as previsões no teste. A fábrica fornecida pelo consumidor deve devolver uma instância nova. Os testes cobrem partições, repetibilidade, isolamento do transformador em regressão e classificação, e comparação por fold com scikit-learn 1.9.1.

## Implementação e critérios de aceite

1. **Isolamento na classificação.** Concluído. As médias observadas por fold são 16, 11 e 6.
2. **Comparação por fold.** Concluída com fixture gerada no `venv` do repositório usando scikit-learn 1.9.1. R² e acurácia são comparados em ordem, com rejeição de valores não finitos e tolerância de 1e-10.
3. **Fechamento.** Concluído. `go test ./...`, o exemplo de validação cruzada e `git diff --check` passaram. A etapa 6 está concluída.

## Depois desta entrega

Após fechar a etapa 6, a primeira fatia proposta para a etapa 7 é ler e gravar CSV com a biblioteca padrão. O aceite dessa fatia é salvar uma matriz e recarregá-la preservando dimensões, valores e ordem das colunas. Antes de codificar, definir a representação de colunas e alvos que JSON e Parquet também poderão usar.

Outra melhoria de API com uso concreto é aceitar alvos inteiros na divisão treino/teste. Hoje o [exemplo de classificação](../examples/classification/main.go) converte os rótulos para obter a partição e usa os índices para recuperar os inteiros. A extensão deve preservar a chamada atual com `[]float64` e o alinhamento de X e y. O requisito de ler e gravar matrizes em CSV, Parquet e JSON está registrado na [spec de dados tabulares](specs/11-carregamento-dados.md), como expansão separada da validação cruzada. Integrações SQL ficam fora desse requisito por dependerem de bibliotecas e drivers externos. Estratificação, persistência de classificadores, busca de hiperparâmetros, novas famílias e dados esparsos são extensões a priorizar separadamente. Para publicação, seguem pendentes a escolha do caminho público do módulo e da licença e a confirmação da versão mínima de Go.
