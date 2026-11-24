# Contexto de implementação da spec 11

As correções de integridade estão implementadas na árvore de trabalho. Parquet preserva nomes, posições e valores na ordem original e rejeita colunas opcionais e repetidas. Os leitores JSON rejeitam null, strings numéricas, linhas null e conteúdo posterior ao documento. CSV e JSON preservam tabelas nomeadas sem linhas. SplitInt usa o limite superior exclusivo do tipo int.

Select aceita colunas numéricas de NewTypedTable, com erro contextual para células incompatíveis e inteiros que perderiam precisão. SplitTargets retorna matrizes de características e múltiplos alvos. Os escritores continuam oferecendo o contrato de tabela numérica com Data. Para persistir uma seleção de uma tabela mista, construir NewTable com a matriz selecionada.

O exemplo tabular_table recebe arquivos locais de entrada e saída e grava uma previsão por amostra. O módulo mantém Go 1.26.0 e a dependência direta parquet-go. As operações de tabela são integrais. ForEachCSVRow entrega strings por registro e recebe o cabeçalho como argumento em cada chamada.

A suíte de regressão cobre colunas Parquet fora de ordem, JSON inválido, tabelas vazias, seleção tipada e overflow de rótulos. Ainda cabem decisões de expansão sobre persistência de tipos heterogêneos e API incremental tipada. O contrato atual rejeita nulos e valores incompatíveis em conversões numéricas.
