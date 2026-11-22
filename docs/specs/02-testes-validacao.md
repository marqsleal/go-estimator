# Testes e validação de compatibilidade

## Suíte obrigatória Go

Usar `testing`, exemplos executáveis e dados em `testdata/`, sem bibliotecas externas. Cobrir API pública, casos calculados à mão, entradas inválidas, regressões e propriedades matemáticas relevantes. Executar `go test ./...`, `go vet ./...` e `go test -race ./...`; `gofmt` deve estar limpo. Fuzzing direcionado a formatos e validação é complementar e deve ter orçamento limitado em CI.

## Referência Python e fixtures

- Quando necessário, gerar externamente fixtures versionados e legíveis, com entradas, parâmetros explícitos, saídas e metadados: versões de Python, scikit-learn, NumPy e SciPy, data e comando de geração. Fixar versão da referência por lote; quando houver divergência, manter fixtures separados por versão ou atualizar com revisão explícita.
- A suíte Go lê fixtures existentes e não executa Python, acessa rede ou instala dependências. Scripts e ambientes Python não integram o projeto; fixtures são dados estáticos, não código Python. Não versionar `venv/` nem copiar código de terceiros sem avaliar licença.
- Incluir casos típicos, degenerados e limites suportados. Registrar recursos da referência excluídos e diferenças de erros, formatos, aleatoriedade e estabilidade; não converter uma diferença não suportada em sucesso silencioso.
- Comparar números finitos com `|a-b| <= atol + rtol*|b|`, escolhendo tolerâncias por operação e escala e justificando-as na fixture ou teste. Comparar NaN e infinitos explicitamente quando suportados; nunca mascarar resultados não finitos com tolerância. Não comparar coeficientes de solução não única como se fossem únicos.
- Para aleatoriedade, persistir entradas e índices ou gerador compatível; não assumir que sementes iguais geram amostras iguais em linguagens diferentes.

## Critério de conclusão por recurso

API e defaults documentados; testes públicos e de erro passando; comparação com referência para casos suportados; divergências registradas; exemplos executáveis; revisão de estabilidade numérica. Uma matriz de compatibilidade deve acompanhar cada recurso e identificar versão, opções suportadas e pendências.
