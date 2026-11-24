# Matriz de compatibilidade inicial

Referência: scikit-learn 1.9.1, Python 3.11.17, NumPy 2.4.6 e SciPy 1.17.1. Os resultados são equivalentes dentro de tolerâncias numéricas; igualdade bit a bit não é prometida.

| Recurso Go | Equivalente Python | Cobertura inicial | Diferenças e limites |
| --- | --- | --- | --- |
| `matrix.Dense` | `numpy.ndarray` 2D | `float64`, row-major, cópia na construção, indexação e shape explícitos | Sem tipos heterogêneos, esparsidade ou NaN em estimadores. `At` causa panic para índice inválido. |
| `tabular` | Leitura de dados tabulares | CSV, JSON e Parquet para tabelas numéricas; `Table` nomeada, seleção ordenada de características, alvos `float64` e inteiros | Leitura integral em memória. Parquet aceita apenas colunas DOUBLE planas e requeridas. Sem nulos, tipos heterogêneos ou operações incrementais. JSON tabular usa `{columns,data}`. |
| `preprocessing.StandardScaler` | `StandardScaler` | `Fit`, `Transform`, `FitTransform`, `with_mean`, `with_std`, variância populacional, coluna constante | Sem `copy=false`, pesos, NaN, sparse ou detecção numérica de quase-constantes. |
| `linear_model.LinearRegression` | `LinearRegression` | Um alvo denso, `fit_intercept`, atributos ajustados, `Predict`, `Score` | Sem pesos, múltiplas saídas, positividade ou `n_jobs`. Solver QR de Householder com pivoteamento e limiar relativo `1e-12`; posto deficiente retorna erro em vez de solução de norma mínima por SVD. |
| `linear_model.LogisticRegression` | `LogisticRegression` | Classificação binária com rótulos `int`, L2, `C`, intercepto, probabilidades ordenadas por classe e acurácia | Solver Newton amortecido próprio em vez de `lbfgs`; critério de convergência próprio. Sem multiclasse, pesos, rótulos não inteiros ou ausência de regularização. Ver [spec](specs/07-linear-model.md). |
| `model_selection.TrainTestSplit` | `train_test_split` | Matriz + vetor, `test_size` fracionário, `shuffle`, índices retornados, semente inteira | `test_size=0.25` por padrão; arredondamento do teste para cima. Semente de 0 a 2³²−1. Sequência Go não coincide com NumPy. Sem `stratify`, `train_size` ou múltiplos arrays. |
| `model_selection.KFold` e `CrossValScore` | `KFold` e `cross_val_score` | Folds com ou sem shuffle, fábricas de estimadores por fold, R² e acurácia por partição; fixture scikit-learn 1.9.1 | Sem estratificação ou agregação automática. Seeds Go e NumPy podem produzir ordens diferentes. |
| `metrics` | métricas de regressão e classificação | MAE, MSE, R². accuracy, precision, recall e confusion matrix para rótulos inteiros | Sem `sample_weight`, múltiplas saídas ou opção `force_finite` em regressão. Precision/recall exigem classe positiva; denominador zero retorna zero. |
| `pipeline.RegressionPipeline` | `Pipeline` | Etapas nomeadas `StandardScaler` + regressor de saída única; ajuste atômico | Sem `get_params`, `set_params`, cache, passthrough ou composição de classificação. |
| `pipeline.ClassificationPipeline` | `Pipeline` | Etapas nomeadas + classificador binário, ajuste atômico | Interface separada da regressão; sem persistência nesta etapa. |
| `modelio` | Sem equivalente direto | JSON próprio v1; carrega regressão isolada ou pipeline scaler + regressão para predição | Não importa pickle/joblib. O formato é específico do go-learn. |

O fixture [scikit-learn 1.9.1](../testdata/scikit-learn-1.9.1-regression.json) cobre split sem shuffle, estatísticas do scaler, coeficientes, intercepto e predições no fluxo completo. A validação usa `atol=1e-12` nesse caso pequeno. Cobertura de outras escalas e posto numérico é feita por testes Go específicos.

Os benchmarks em [benchmarks](../benchmarks/README.md) comparam o custo em um host e registram limitações; eles não ampliam a promessa de compatibilidade numérica.

A [fixture de classificação](../testdata/scikit-learn-1.9.1-classification.json) cobre dados não separáveis, classes -4 e 9, intercepto habilitado/desabilitado e scaler opcional. Os testes com tolerância de otimização Go `1e-10` comparam coeficientes, intercepto e probabilidades com `atol=1e-7`, além dos rótulos previstos.
