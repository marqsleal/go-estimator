# Benchmarks

Os benchmarks Go medem ajuste, predição, métricas, `StandardScaler` (`Fit`, `Transform` e `FitTransform`) e o fluxo com pipeline. A geração dos dados ocorre antes do cronômetro. Os casos iniciais são `128 × 4` e `10.000 × 32`, com entradas determinísticas.

```sh
go test ./benchmarks -run '^$' -bench . -benchmem -count 5
```

Para comparar com scikit-learn, use a mesma matriz, o mesmo alvo e os mesmos índices de treino/teste nas duas linguagens. Registre versão exata de Go e scikit-learn, sistema operacional, arquitetura, CPU, número de threads de BLAS e cinco execuções por cenário. Verifique predições e métricas antes de cronometrar. O Python fica fora deste módulo e não é chamado pelo benchmark Go.

Compare medianas e dispersão, separando `Fit`, `Predict`, `StandardScaler` e pipeline completo. Não interprete os resultados como promessa de desempenho equivalente a scikit-learn, que pode usar BLAS otimizada.

A primeira medição está em [results/initial-baseline.md](results/initial-baseline.md). Ela registra as versões e limitações do experimento.
