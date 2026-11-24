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

### Primeira fatia implementada: CSV numérico

O pacote `tabular` expõe `ReadCSV(io.Reader, CSVOptions)` e `WriteCSV(io.Writer, *matrix.Dense, rune)`. A leitura é integral, aceita delimitador configurável e cabeçalho opcional. Os campos de dados devem ser números float64 e todas as linhas devem ter a mesma largura. Erros indicam registro e coluna. A gravação não inclui cabeçalho e usa precisão suficiente para reconstruir os valores float64. O chamador controla os recursos de I/O. Valores ausentes e campos não numéricos geram erro. Esta fatia não suporta nomes de coluna ou alvos tipados.

O próximo passo da spec é consolidar o contrato de tabela com nomes e tipos antes de adicionar JSON e Parquet.

### JSON numérico

O pacote `tabular` também lê e grava matrizes JSON como arrays de arrays numéricos, por exemplo `[[1,2],[3,4]]`. A ordem das linhas e colunas é preservada. Linhas irregulares, valores nulos, strings, dados após o documento e valores não numéricos são rejeitados com erro. O formato não carrega nomes de colunas nem metadados de tipos. A operação é integral e mantém os dados em memória.

### Tabela numérica com nomes de coluna

`tabular.Table` associa nomes únicos e não vazios às colunas numéricas de `matrix.Dense`. `Select` retorna colunas na ordem solicitada e preserva a ordem das linhas. `Split` seleciona características e um alvo `float64`; `SplitInt` exige valores integrais representáveis por `int` para classificação. Ambos mantêm o alinhamento por linha. `ReadCSVTable` e `WriteCSVTable` leem e gravam CSV com cabeçalho. `ReadJSONTable` e `WriteJSONTable` usam um objeto com `columns` e `data`. Valores seguem sendo numéricos, sem tipos heterogêneos ou nulos.

### Parquet numérico

`ReadParquet` e `WriteParquet` suportam tabelas planas com colunas requeridas do tipo Parquet `DOUBLE`, preservando nomes e ordem das colunas e linhas. A implementação usa `github.com/parquet-go/parquet-go`. A leitura e gravação são integrais e bufferizadas. Colunas aninhadas, nulas e tipos físicos diferentes de DOUBLE são rejeitados. A API ainda não oferece processamento incremental ou seleção de colunas durante a leitura.

- Definir se o módulo principal suporta cada formato diretamente ou por adaptadores opcionais.
- Escolher contratos e pacotes para arquivos locais e eventual leitura ou gravação incremental.
- Definir a representação de tabela intermediária necessária para preservar nomes e tipos antes de selecionar as colunas numéricas.
- Definir como representar alvos de regressão e classificação sem perder os tipos públicos atuais `[]float64` e `[]int`.
- Parquet usa `github.com/parquet-go/parquet-go`; revisar periodicamente licença, manutenção, tamanho de dependências e compatibilidade com a versão mínima de Go. CSV e JSON usam a biblioteca padrão.
- Estabelecer limites para tamanho de arquivo, profundidade de JSON, precisão numérica e tratamento de colunas heterogêneas.

## Critérios de aceite

- CSV, Parquet e JSON têm contratos documentados para leitura e gravação e testes sem serviços externos.
- Conversões mantêm forma, ordem e alinhamento entre características e alvos.
- Tipos incompatíveis, valores ausentes e dados malformados produzem erros explícitos com contexto.
- Pelo menos um exemplo executável demonstra carregar uma fonte local, ajustar um pipeline existente e gravar resultados.
- A matriz de compatibilidade e o catálogo de exemplos registram formatos suportados, dependências opcionais e limitações.
