# Baseline inicial

Medição exploratória em 2026-11-22. Os números servem como ponto inicial, não como meta de desempenho.

## Ambiente

- Go 1.26.1, Linux/amd64, AMD Ryzen 7 5700X, benchmark Go com `GOMAXPROCS=4` (sufixo `-4` do runner).
- Python 3.11.17, scikit-learn 1.9.1, NumPy 2.4.6, SciPy OpenBLAS 0.3.31.188.0; `OPENBLAS_NUM_THREADS=1`, `OMP_NUM_THREADS=1` e `MKL_NUM_THREADS=1` definidos antes dos imports.
- Entradas determinísticas `float64`, recriadas pela mesma fórmula em Go e Python. O alvo é uma combinação linear exata das características. As predições foram verificadas contra o alvo antes das medições.
- Para índices começando em zero, `v[i,j] = (((i*73856093) XOR (j*19349663) XOR ((i+1)*(j+11)*83492791)) mod 1009) / 100`; `y[i] = 0.5 + soma(v[i,j] * (j+1) / p)`. A geração dos dados ficou fora do cronômetro.
- Go: cinco execuções de `go test ./benchmarks -run '^$' -bench 'Benchmark(LinearRegression|RegressionPipelineFitPredict)' -benchmem -benchtime=100ms -count 5` e `go test ./benchmarks -run '^$' -bench '^BenchmarkStandardScaler/FitTransform/' -benchmem -benchtime=100ms -count 5`; a tabela mostra a mediana do tempo. Python: um harness temporário externo fez aquecimento, cinco repetições e mediana por chamada, com 1.000 chamadas por repetição no caso pequeno e 10 no grande. Nenhum script Python foi adicionado ao projeto.
- Os casos medem construção do estimador + `Fit`, predição em modelo ajustado, `StandardScaler.fit_transform` e construção + ajuste + predição do pipeline.

## Tempos medianos

| Operação | Dados | Go | scikit-learn | Relação observada |
| --- | ---: | ---: | ---: | --- |
| LinearRegression Fit | 128 × 4 | 36,6 µs | 408 µs | Go 11,2× menor |
| LinearRegression Predict | 128 × 4 | 1,18 µs | 72,2 µs | Go 61,3× menor |
| StandardScaler FitTransform | 128 × 4 | 5,06 µs | 360 µs | Go 71,2× menor |
| Pipeline Fit + Predict | 128 × 4 | 46,1 µs | 1,18 ms | Go 25,5× menor |
| LinearRegression Fit | 10.000 × 32 | 109 ms | 3,80 ms | Go 28,6× maior |
| LinearRegression Predict | 10.000 × 32 | 499 µs | 169 µs | Go 2,96× maior |
| StandardScaler FitTransform | 10.000 × 32 | 2,56 ms | 1,80 ms | Go 1,42× maior |
| Pipeline Fit + Predict | 10.000 × 32 | 105 ms | 7,60 ms | Go 13,9× maior |

No Go, o caso `10.000 × 32` usou cerca de 10,5 MB/op e 42 alocações para `Fit`, 80 KB/op e uma alocação para `Predict`, 5,13 MB/op e sete alocações para o scaler, e 20,8 MB/op e 56 alocações para o pipeline.

## Leitura dos resultados

O custo fixo maior do Python aparece nos casos pequenos. Nos dados maiores, o QR puro em Go é o gargalo principal: o scikit-learn usa rotinas nativas otimizadas. A baseline confirma a limitação já documentada e aponta QR como alvo para otimização depois de perfis, sem trocar dependências ou relaxar estabilidade numérica. A comparação não mede RSS, não inclui divisão treino/teste e não substitui fixtures numéricos por versão.
