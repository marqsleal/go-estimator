# Contrato proposto para o primeiro fluxo

Esta especificação detalha o primeiro fluxo de regressão. Os nomes Go abaixo são propostas para testar com um exemplo compilável assim que os pacotes existirem. A referência é o scikit-learn 1.9.1 instalado localmente.

## Superfície pública

| Referência Python | Proposta Go | Comportamento verificável |
| --- | --- | --- |
| Array de forma (n, p) | matrix.NewDense(n, p, values) | Copiar os valores em ordem de linhas. Rejeitar dimensões negativas, overflow e tamanho incompatível. Estimadores exigem ao menos uma linha e uma coluna. |
| StandardScaler() | preprocessing.NewStandardScaler() | copy, with_mean e with_std começam verdadeiros. Opções como WithMean(false) preservam a diferença entre omitir e desabilitar. |
| scaler.mean_, scaler.var_ | scaler.Mean(), scaler.Var() | Devolver cópias. Variância populacional, divisor n. Se ambas as operações estiverem desabilitadas, os atributos equivalentes a None são slices nil após o ajuste. |
| LinearRegression() | linear_model.NewLinearRegression() | Intercepto habilitado, entrada preservada, uma saída. Fit recebe matriz e vetor. Predict devolve um valor por linha. |
| train_test_split(X, y) | model_selection.TrainTestSplit(x, y, opções...) | Retornar estrutura com XTrain, XTest, YTrain e YTest. Por padrão embaralhar e reservar ceil(0,25 × n) linhas para teste. Iniciar com test_size fracionário e shuffle configuráveis. |
| mean_absolute_error, mean_squared_error, r2_score | metrics.MeanAbsoluteError, MeanSquaredError, R2Score | Funções puras recebem dois vetores e retornam (float64, error). Alinhar a ordem yTrue, yPred com Python. |
| Pipeline com scaler e regressor | pipeline.NewRegressionPipeline(etapas, estimador) | Validar nomes e ordem na construção. Cada Fit prepara estado novo e só o publica após a última etapa concluir. |

As opções disponíveis no MVP devem aparecer na documentação de cada construtor. Para regressão, iniciar com fit_intercept configurável. Registrar copy_X=True, positive=False e n_jobs=None como configurações fixas. Para o scaler, adiar copy=false até definir uma operação pública que permita modificar os dados com segurança. Para o split, adiar train_size, stratify e múltiplos arrays arbitrários.

## Um exemplo que atravessa os pacotes

Considere X = [[0], [1], [2], [3], [4], [5]] e y = [1, 3, 5, 7, 9, 11]. Com test_size=0,25 e shuffle=False as quatro primeiras linhas treinam o modelo. A referência externa pode ser consultada com o seguinte trecho.

```python
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

X = [[0], [1], [2], [3], [4], [5]]
y = [1, 3, 5, 7, 9, 11]
xt, xv, yt, yv = train_test_split(X, y, test_size=0.25, shuffle=False)
model = make_pipeline(StandardScaler(), LinearRegression()).fit(xt, yt)
pred = model.predict(xv)
```

Esboço de uso Go em uma função que recebe x e y e retorna error. A etapa nomeada guarda uma fábrica para que dois ajustes possam construir estados independentes.

```go
split, err := model_selection.TrainTestSplit(x, y,
    model_selection.WithTestSize(0.25), model_selection.WithShuffle(false))
if err != nil {
    return err
}
model, err := pipeline.NewRegressionPipeline(
    []pipeline.Step{{Name: "scale", New: func() pipeline.Transformer {
        return preprocessing.NewStandardScaler()
    }}},
    func() pipeline.Regressor { return linear_model.NewLinearRegression() },
)
if err != nil {
    return err
}
if err := model.Fit(split.XTrain, split.YTrain); err != nil {
    return err
}
pred, err := model.Predict(split.XTest)
if err != nil {
    return err
}
mse, err := metrics.MeanSquaredError(split.YTest, pred)
```

A média ajustada deve ser 1,5 e a variância 1,25. A previsão deve ser [9, 11], MSE aproximadamente zero e R² aproximadamente 1. Usar tolerância absoluta de 1e-12 para esse caso pequeno. A média distingue um pipeline correto de outro que inclua as linhas de teste ao ajustar o scaler. A previsão sozinha pode coincidir mesmo com esse erro. No teste de integração, usar um transformador observável para registrar as linhas recebidas por Fit.

## Casos que delimitam a compatibilidade

- **Posto numérico.** Na referência instalada, LinearRegression.fit passa tol (default 1e-6) como cond ao solver denso de mínimos quadrados. QR simples pode obter coeficientes muito diferentes em colunas quase colineares. Começar com sistemas sobredeterminados ou quadrados de posto completo após centralização. Usar QR com pivoteamento e critério relativo documentado sobre a diagonal de R. Esse critério aproxima o posto e pode divergir do corte por valores singulares. Testar matrizes diagonais com razões 1e-4 e 1e-8, fit_intercept=false, além de duas colunas idênticas. Comparar coeficientes no primeiro caso e exigir erro de posto nos dois últimos. Documentar essa divergência em relação à solução de norma mínima do Python.
- **Intercepto.** Ao centralizar X para p características independentes são necessárias pelo menos p+1 amostras. Testar também X deslocado por uma constante grande e comparar resíduos. Nunca usar coeficientes ajustados sobre variáveis padronizadas como se estivessem nas unidades originais.
- **Estado.** Após Fit válido, provocar erro com y curto e confirmar que as previsões anteriores continuam disponíveis. Repetir com erro no estimador final do pipeline para verificar que o scaler anterior também foi preservado.
- **Métricas.** Com yTrue=[2,2], R² padrão deve valer 1 para [2,2] e 0 para [1,2]. Com uma só amostra aceitar NaN para R² e documentar que Python também emite aviso. MSE para [1,3] e [2,1] deve valer 2,5. Entradas de comprimentos diferentes retornam erro.
- **Partições.** Para seis linhas, verificar treino de tamanho quatro e teste de tamanho dois. Uma linha só é inválida com as opções padrão. Com shuffle e semente fixa, comparar cobertura, alinhamento e repetibilidade dentro do Go. Para comparar modelos entre linguagens, usar os mesmos índices persistidos.
- **Fixture.** Guardar X, y, índices, opções, atributos esperados e tolerâncias em testdata/. Representar NaN de forma explícita, por exemplo com a string "NaN" no JSON. A suíte Go lê o dado estático com encoding/json e identifica a operação e a posição de qualquer divergência.

## Sequência de aceite

1. Fazer o exemplo de construção de matriz compilar e testar cópia, forma e overflow.
2. Entregar scaler, split e métricas isoladamente. Validar que a mesma semente, inclusive zero, reproduz os índices e que shuffle desabilitado preserva a ordem. Executar os casos manuais antes de comparar fixtures.
3. Entregar regressão e validar os limites de posto, coeficientes e resíduos. Integrar o pipeline e converter o esboço em Example executável com verificação da saída.
4. Depois de validar a [persistência para inferência](09-persistencia.md), criar benchmarks/ com casos de 128 × 4 e 10.000 × 32 em float64. Medir Fit com instância nova e Predict com modelo já ajustado. Excluir geração e leitura dos dados do cronômetro. Em ambas as linguagens usar as mesmas entradas e conferir previsões antes de medir. Registrar cinco execuções por cenário, mediana, dispersão, versões e threads. Comando Go previsto: `go test ./benchmarks/... -run '^$' -bench . -benchmem -count 5`.
