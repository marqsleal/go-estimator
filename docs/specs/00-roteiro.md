# Roteiro de implementação

Cada etapa entrega uma fatia utilizável e só passa à próxima após cumprir os critérios de [validação](02-testes-validacao.md). Prazos e desempenho só serão estimados depois de medir o primeiro recorte.

1. **Concluído — fundação:** matriz densa, validações, contratos, testes e exemplo. A versão mínima inicial é Go 1.22; caminho público do módulo e licença seguem pendentes antes da publicação.
2. **Concluído — primeiro fluxo de regressão:** StandardScaler, LinearRegression, divisão com `random_state`, MAE/MSE/R², pipeline, exemplo completo, fixture scikit-learn 1.9.1 e [matriz de compatibilidade](../compatibilidade.md).
3. **Concluído — persistência:** JSON próprio v1 e ciclo de salvar, carregar e prever conforme a [spec](09-persistencia.md).
4. **Concluído — primeira medição:** benchmarks Go em `benchmarks/`, baseline reproduzível e comparação externa com scikit-learn. O QR puro em Go é o gargalo de treino para a carga maior; qualquer otimização deve preservar o contrato numérico.
5. **Concluído — classificação binária:** LogisticRegression com Newton e L2, métricas de classificação, pipeline com rótulos inteiros e fixture scikit-learn 1.9.1. Persistência de classificação exige uma extensão futura do formato.
6. **Implementada, em validação — validação de modelos:** KFold e avaliação por fold usam fábricas de estimadores. Restam a verificação direta do isolamento do scaler e a comparação por fold com a referência. Ver [contexto e critérios de aceite](../contexto-validacao-cruzada.md).
7. **Expansão:** priorizar leitura e gravação de dados tabulares, outras famílias, suporte esparso e seleção de hiperparâmetros com specs próprias e evidência de necessidade. O requisito para CSV, Parquet e JSON está em [dados tabulares](11-carregamento-dados.md). Integrações SQL ficam fora deste escopo por dependerem de bibliotecas e drivers externos.

## Marco da primeira versão utilizável

Um consumidor consegue importar a biblioteca, preparar dados densos, dividir treino/teste, ajustar um pipeline scaler + regressão, prever e calcular métricas. A documentação informa versão de referência e diferenças suportadas; testes passam sem Python ou rede; exemplos compilam; o benchmark registra ambiente e não atribui uma meta de velocidade inexistente.
