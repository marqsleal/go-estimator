# linear_model

## Objetivo e dependências

Implementar estimadores familiares ao scikit-learn usando fundação, álgebra linear interna e contratos de dados. O primeiro é `LinearRegression`; `LogisticRegression` binária é uma etapa posterior.

## LinearRegression

- `Fit`, `Predict` e, quando houver métrica suportada, `Score` com semântica R² da referência. Publicar coeficientes, intercepto e número de características ajustadas com acesso seguro.
- Suportar `fit_intercept` com default da referência e dados densos, finitos, alvo único e sem pesos. Registrar parâmetros e comportamentos Python ainda não suportados, incluindo saídas múltiplas, pesos, restrições de positividade e casos de posto deficiente, quando aplicável.
- Resolver mínimos quadrados por QR; não inverter explicitamente XᵀX. Prever entradas inconsistentes, dados insuficientes e posto deficiente de acordo com o contrato publicado, sem afirmar paridade onde não houver.
- Testar reta exata, intercepto habilitado/desabilitado, múltiplas características, ruído, variáveis de escala distinta, ajuste repetido, erro antes de ajuste e dimensões inválidas. Comparar coeficientes, intercepto, predições e R² com Python, usando tolerâncias adequadas.

## LogisticRegression binária — posterior

- Antes de implementar, escolher e documentar um único solver da referência, sua função objetivo, regularização, defaults, critério de convergência, tratamento do intercepto e ordem das classes. Não afirmar compatibilidade de parâmetros ou valores padrão que o solver escolhido não suporta.
- Implementar `Fit`, `Predict`, `PredictProba` e `Score`; definir retorno para não convergência. Usar cálculo estável de sigmoid e log-loss. Multiclasse e pesos ficam fora até serem especificados.
- Testar classes balanceadas/desbalanceadas, valores extremos, ordem das classes, probabilidades, regularização, convergência e erros; comparar com Python considerando tolerância e soluções próximas.

## Desempenho

Medir treinamento e predição separadamente por dimensão e por escala, além de alocações. Não prometer equivalência de tempo com implementações aceleradas por BLAS.
