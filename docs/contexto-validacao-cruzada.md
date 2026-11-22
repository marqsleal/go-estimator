# Próxima entrega · concluir a validação cruzada

Base analisada: commit `092480e`. A biblioteca já oferece regressão linear, classificação logística binária, scaler, métricas, pipelines e persistência JSON de regressão. A etapa 6 do roteiro já tem API e exemplos executáveis.

## Ponto de partida

[KFold](../model_selection/kfold.go) produz índices. [CrossValScore e CrossValClassificationScore](../model_selection/cross_validation.go) chamam a fábrica em cada fold, ajustam o estimador no treino e avaliam as previsões no teste. A fábrica fornecida pelo consumidor deve devolver uma instância nova. Os [testes atuais](../model_selection/cross_validation_test.go) verificam a divisão, a repetibilidade e o número de chamadas da fábrica. O teste de regressão usa uma reta perfeita e o de classificação verifica a faixa da acurácia. Ambos podem passar se o scaler aprender estatísticas do conjunto inteiro.

## Implementação e critérios de aceite

1. **Verificar o isolamento.** Usar uma coluna com valores distintos, por exemplo `[0, 2, 10, 12, 20, 22]`, e três folds em ordem. Capturar os scalers criados pelas fábricas de etapas nos dois tipos de pipeline. Após a avaliação, suas médias devem ser, em ordem, 16, 11 e 6. Um ajuste global produziria 11 nos três. Usar rótulos alternados na classificação para manter ambas as classes em cada treino.
2. **Conferir o resultado por fold.** Gerar externamente uma fixture com scikit-learn 1.9.1, pipelines com scaler, regressão com algum ruído e classificação com margem suficiente para previsões estáveis. Passar explicitamente as mesmas partições sem embaralhamento nos dois experimentos. Registrar dados, índices, parâmetros, versões, receita de geração, scores e tolerância para R². O teste Go deve ler o JSON, verificar índices e ordem, rejeitar NaN inesperado e comparar R² e acurácia por fold. Escolher dados que produzam pontuações diferentes entre folds para detectar troca de ordem.
3. **Fechar a entrega.** Registrar a cobertura e os limites na [matriz de compatibilidade](compatibilidade.md). Revisar o [exemplo](../examples/cross_validation/main.go) e seu [guia](../examples/README.md), preservando a semente local. Executar o exemplo e as verificações da [spec de validação](specs/02-testes-validacao.md). Marcar a etapa 6 concluída após os critérios acima passarem.

## Depois desta entrega

Uma próxima melhoria de API com uso concreto é aceitar alvos inteiros na divisão treino/teste. Hoje o [exemplo de classificação](../examples/classification/main.go) converte os rótulos para obter a partição e usa os índices para recuperar os inteiros. A extensão deve preservar a chamada atual com `[]float64` e o alinhamento de X e y. Estratificação, persistência de classificadores, busca de hiperparâmetros, novas famílias e dados esparsos são extensões a priorizar separadamente. Para publicação, seguem pendentes a escolha do caminho público do módulo e da licença e a confirmação da versão mínima de Go.
