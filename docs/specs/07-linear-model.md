# linear_model

## Objetivo e dependências

Implementar estimadores familiares ao scikit-learn usando fundação, álgebra linear interna e contratos de dados. O primeiro é `LinearRegression`; `LogisticRegression` binária é uma etapa posterior.

## LinearRegression

- `Fit`, `Predict` e, quando houver métrica suportada, `Score` com semântica R² da referência. Publicar coeficientes, intercepto e número de características ajustadas com acesso seguro.
- Suportar `fit_intercept` com default da referência e dados densos, finitos, alvo único e sem pesos. Registrar parâmetros e comportamentos Python ainda não suportados, incluindo saídas múltiplas, pesos, restrições de positividade e casos de posto deficiente, quando aplicável.
- Resolver mínimos quadrados por QR; não inverter explicitamente XᵀX. Prever entradas inconsistentes, dados insuficientes e posto deficiente de acordo com o contrato publicado, sem afirmar paridade onde não houver.
- Testar reta exata, intercepto habilitado/desabilitado, múltiplas características, ruído, variáveis de escala distinta, ajuste repetido, erro antes de ajuste e dimensões inválidas. Comparar coeficientes, intercepto, predições e R² com Python, usando tolerâncias adequadas.

## LogisticRegression binária

### Decisões do primeiro recorte

- O pacote `linear_model` terá `LogisticRegression`, criado por `NewLogisticRegression(options ...LogisticRegressionOption)`. Opções: `WithC(float64)`, `WithLogisticFitIntercept(bool)`, `WithTolerance(float64)` e `WithMaxIterations(int)`. Defaults: `C=1`, intercepto habilitado, tolerância `1e-6` e no máximo `100` iterações. `C` e tolerância devem ser finitos e positivos; o limite de iterações deve ser positivo. Não há opção de ausência de regularização no primeiro recorte.
- O único solver é Newton com busca linear Armijo, Hessiana em memória densa e fatoração de Cholesky. Inicializa coeficientes e intercepto em zero. Para `n` amostras, minimiza a log-loss binária média mais `||w||²/(2 n C)`; o intercepto não é regularizado. Avaliar log-loss, sigmoid, gradiente e Hessiana com fórmulas estáveis em `float64`. A busca começa com passo 1, verifica `f(θ + αd) <= f(θ) + 1e-4 α ∇f(θ)·d` e reduz `α` pela metade, até 50 tentativas.
- O critério de convergência é `max(abs(gradiente)) <= tolerance`, calculado para a função objetivo acima. Se a fatoração ou busca linear falhar, ou se não convergir em `maxIterations`, `Fit` retorna `ErrConvergence` com o número de iterações no erro e conserva qualquer ajuste válido anterior. Falha de validação também conserva o ajuste anterior. O modelo expõe o número de iterações do último ajuste bem-sucedido.
- O alvo é `[]int` e precisa conter exatamente duas classes distintas. `Classes()` retorna as duas em ordem crescente; a maior é positiva. `PredictProba` retorna `*matrix.Dense` de forma `(n_samples, 2)`, com colunas nessa ordem. `Predict` escolhe a classe positiva quando o logit é estritamente positivo (equivalente a `p > 0.5` antes do arredondamento); no empate escolhe a classe menor, como a decisão de primeira classe no scikit-learn.
- A interface pública inclui `Fit(*matrix.Dense, []int) error`, `Predict(*matrix.Dense) ([]int, error)`, `PredictProba(*matrix.Dense) (*matrix.Dense, error)`, `Score(*matrix.Dense, []int) (float64, error)`, `Coef() []float64`, `Intercept() float64`, `Classes() []int`, `NFeaturesIn() int` e `NIter() int`. Acessores de estado retornam cópias para slices; antes do ajuste, slices são nil e valores numéricos/dimensões são zero, em linha com `LinearRegression`.
- O escopo é entrada densa `float64`, finita, alvo binário inteiro e sem pesos. Multiclasse, rótulos não inteiros, pesos, warm start, solver selecionável e previsão de decisão ficam fora. Classificação não promete mesmo coeficiente ou trajetória de iterações que `lbfgs`; comparação com scikit-learn verifica a solução da mesma função objetivo dentro de tolerância, além de probabilidades e classes previstas.
- O custo por iteração é `O(n q² + q³)` e a memória adicional é `O(q²)`, onde `q` é o número de características mais um quando o intercepto é habilitado. Esse limite deve ser medido antes de ampliar a dimensão de uso recomendada.

Essas decisões usam `C`, o intercepto não penalizado, classes ordenadas e colunas de probabilidade em ordem de classe como convenções familiares do scikit-learn 1.9.1. O scikit-learn usa `lbfgs` por padrão; o solver Newton e o critério de gradiente acima são escolhas próprias do go-learn e devem aparecer como diferenças na matriz de compatibilidade. Consulte a [documentação de LogisticRegression 1.9.1](https://scikit-learn.org/1.9/modules/generated/sklearn.linear_model.LogisticRegression.html) e a [formulação da loss binária](https://scikit-learn.org/1.9/modules/linear_model.html#binary-case).

### Testes e aceite

- Testar gradiente/Hessiana por diferenças finitas em casos pequenos; convergência em dados separáveis e não separáveis com regularização; classes balanceadas e desbalanceadas; intercepto desligado; características constantes e escalas distintas; logits extremos; ajuste repetido; estado anterior preservado em falha; classe/forma/não finitos inválidos; probabilidades finitas que somam 1 por linha e empate no limiar.
- Gerar fixtures estáticas com scikit-learn 1.9.1, `solver="lbfgs"`, `l1_ratio=0`, `C=1`, `fit_intercept` igual e mesmas amostras. Comparar objetivo, probabilidades, predições e coeficientes com tolerâncias apropriadas à condição dos dados; não comparar quantidade de iterações nem trajetória do solver. Usar pelo menos um caso não separável para evitar comparação baseada só em saturação de probabilidade.

## Desempenho

Medir treinamento e predição separadamente por dimensão e por escala, além de alocações. Não prometer equivalência de tempo com implementações aceleradas por BLAS.
