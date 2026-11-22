# Leitura e gravação de dados tabulares

## Objetivo

Permitir que consumidores leiam e gravem dados tabulares associados às matrizes da biblioteca em formatos comuns, com uma experiência comparável à leitura e gravação de DataFrames. Os dados carregados devem alimentar os fluxos de preparação, divisão, ajuste e avaliação já existentes.

## Requisitos

- Ler e gravar dados tabulares em arquivos CSV, Parquet e JSON.
- Converter colunas numéricas para `matrix.Dense` preservando a ordem das linhas e das características.
- Permitir selecionar colunas de características e uma ou mais colunas de alvo, mantendo X e y alinhados.
- Tornar explícitos cabeçalhos, delimitadores quando aplicável, nomes e ordem das colunas, tipos, nulos e valores ausentes. Não inferir ou descartar dados silenciosamente.
- Retornar erros que identifiquem fonte, coluna e linha quando a conversão falhar, sem retornar uma matriz parcial como resultado válido.
- Permitir que o chamador controle leitores, writers e ciclo de vida dos recursos.
- Documentar formatos, limitações, comportamento de nulos e tipos, uso de memória e diferenças entre operações completas e leitura ou gravação incremental.
- Manter os fluxos atuais baseados diretamente em `matrix.Dense`; o carregamento é uma porta de entrada adicional.
- Fornecer exemplos executáveis para fontes locais. Os testes devem usar fixtures locais, sem depender de serviços externos.

## Decisões necessárias antes da implementação

- Definir se o módulo principal suporta cada formato diretamente ou por adaptadores opcionais.
- Escolher contratos e pacotes para arquivos locais e eventual leitura ou gravação incremental.
- Definir a representação de tabela intermediária necessária para preservar nomes e tipos antes de selecionar as colunas numéricas.
- Definir como representar alvos de regressão e classificação sem perder os tipos públicos atuais `[]float64` e `[]int`.
- Selecionar bibliotecas para Parquet e avaliar licença, manutenção, tamanho de dependências e compatibilidade com a versão mínima de Go. CSV e JSON devem usar a biblioteca padrão sempre que possível.
- Estabelecer limites para tamanho de arquivo, profundidade de JSON, precisão numérica e tratamento de colunas heterogêneas.

## Critérios de aceite

- CSV, Parquet e JSON têm contratos documentados para leitura e gravação e testes sem serviços externos.
- Conversões mantêm forma, ordem e alinhamento entre características e alvos.
- Tipos incompatíveis, valores ausentes e dados malformados produzem erros explícitos com contexto.
- Pelo menos um exemplo executável demonstra carregar uma fonte local, ajustar um pipeline existente e gravar resultados.
- A matriz de compatibilidade e o catálogo de exemplos registram formatos suportados, dependências opcionais e limitações.
