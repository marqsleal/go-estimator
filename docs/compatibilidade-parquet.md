# Compatibilidade Parquet

ReadParquet e WriteParquet usam parquet-go v0.32.1 e suportam tabelas numéricas planas com campos requeridos DOUBLE. Nomes, ordem das colunas e ordem das linhas são preservados. Campos opcionais, repetidos, aninhados e tipos físicos diferentes de DOUBLE produzem erro.

A leitura materializa o arquivo e os valores em memória. A gravação recebe a tabela completa e usa o buffer do escritor Parquet. O chamador controla o ciclo de vida do io.Reader e do io.Writer. As funções não oferecem leitura incremental nem seleção de colunas durante a leitura.

O módulo exige Go 1.26.0. parquet-go é uma dependência direta do pacote tabular. CSV e JSON usam a biblioteca padrão. Tabelas mistas devem selecionar colunas numéricas e construir uma NewTable antes de chamar o escritor.

Os testes de integridade incluem uma tabela com colunas z e a, nessa ordem, para verificar a associação dos valores aos nomes durante a ida e volta.
