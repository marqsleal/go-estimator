# random_state


No primeiro fluxo, apenas train_test_split embaralha dados. StandardScaler e LinearRegression densos são determinísticos. O pipeline delega qualquer opção de aleatoriedade à etapa correspondente.

| Chamada Go proposta | Semântica |
| --- | --- |
| TrainTestSplit(x, y) | Com shuffle habilitado, iniciar um gerador local com entropia do sistema a cada chamada. |
| TrainTestSplit(x, y, WithRandomState(0)) | Zero é uma semente válida. Reiniciar o gerador a cada chamada. Aceitar sementes inteiras de 0 a 2³²−1 e retornar erro para valores fora dessa faixa. |
| TrainTestSplit(x, y, WithShuffle(false)) | Preservar a ordem. Uma semente válida fornecida nesse modo fica sem efeito. |

Usar uma permutação única para X e y. A garantia cobre entradas com a mesma ordem, mesmas opções e mesma versão da biblioteca. No primeiro recorte, o usuário pode fornecer uma semente inteira. O compartilhamento de uma instância mutável de RandomState, disponível em Python, fica para uma extensão. O modo sem semente usa entropia local em Go e pode produzir partições repetidas por acaso.

**Implementado.** O Go usa `math/rand` com uma instância local e Fisher–Yates. Para oito amostras, `test_size=0.25` e `random_state=0`, os índices do conjunto de teste são `[1, 7]`; esse vetor está fixado em teste. A mesma semente é reprodutível nesta implementação Go, mas não promete paridade com NumPy. Nas comparações numéricas com Python, persistir e reutilizar os próprios índices.
