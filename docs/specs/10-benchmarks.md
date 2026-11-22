# Benchmarks de comparação com scikit-learn

## Objetivo

Medir custo de treino e inferência em cargas equivalentes, identificar gargalos e acompanhar regressões. Os resultados não são uma promessa de paridade de desempenho: scikit-learn pode executar NumPy/SciPy com BLAS nativo otimizado, enquanto go-learn usa somente Go padrão.

Os benchmarks Go serão criados depois dos recursos correspondentes, na pasta `benchmarks/`. A execução comparativa do scikit-learn ocorre em ambiente externo, sem incorporar Python ou scripts Python ao repositório. Resultados e metadados podem ser registrados como dados estáticos.

Antes de iniciar os benchmarks do primeiro fluxo, concluir os testes de `random_state` do split. Nas medições de treino e inferência, Go e Python devem carregar os mesmos índices previamente salvos.

## Protocolo

1. Gerar uma vez dados e índices persistidos comuns, com dimensão, distribuição, semente e tipo `float64` registrados. Garantir mesmos parâmetros, intercepto, transformações, dados e resultados válidos; tratar configurações sem equivalência como não comparáveis.
2. Separar leitura e conversão de dados do tempo de `Fit`, `Predict` e `Transform`; reportar também um cenário completo quando relevante. Padronizar se cópias e alocações fazem parte da medição. Não incluir inicialização do interpretador em operações isoladas; medir isso separadamente se desejado.
3. Definir tamanhos pequenos, médios e grandes que caibam na memória disponível; registrar amostras, características, proporção treino/teste e memória dos dados. Não usar dados sintéticos triviais como única carga.
4. Fixar e registrar limites de threads de BLAS/OpenMP **antes de importar NumPy**, bem como `GOMAXPROCS`. Executar pelo menos comparação com um thread; cenários multithread devem ser reportados separadamente. Registrar biblioteca BLAS e configuração efetiva, CPU, RAM, SO, versões, flags de compilação e condições do host.
5. Fazer aquecimento quando apropriado e medições repetidas em processos independentes. Reportar mediana e dispersão, quantidade de repetições e tempo por operação; evitar comparar um único resultado. Em Go usar `go test -bench ... -benchmem` com múltiplas execuções; no Python usar temporização monotônica com harness reproduzível. Documentar diferenças de GC e aquecimento.
6. Reportar alocações Go em bytes/op e allocs/op. Medir pico de RSS dos processos com o mesmo método e unidade, separadamente dos tempos; memória de processos e alocações do Go são métricas distintas. Registrar limitações da medição.
7. Verificar equivalência numérica antes de cronometrar e após mudanças relevantes. Manter artefatos com configurações e resultados brutos; não estabelecer limiares de CI entre máquinas diferentes. Comparar regressões somente em ambiente controlado, após estabelecer baseline.

## Matriz inicial

| Operação | Carga | Resultado |
| --- | --- | --- |
| StandardScaler | Fit, Transform, FitTransform; tamanhos variados | tempo, memória e equivalência |
| LinearRegression | Fit e Predict; amostras e características variadas | tempo, memória, resíduos e previsões |
| Pipeline | Fit e Predict completos | tempo total e equivalência |
| Métricas | vetores de tamanhos variados | tempo, alocações e valor |

Os benchmarks Go devem funcionar sem Python instalado. As versões do ambiente externo usado para comparação devem ser registradas explicitamente.
