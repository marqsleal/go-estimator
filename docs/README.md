# Plano de implementação do go-learn

O go-learn será uma biblioteca Go de aprendizado de máquina clássico, inspirada na API e no comportamento do scikit-learn. Produção, testes obrigatórios e benchmarks Go usarão exclusivamente a biblioteca padrão do Go. Python e seus pacotes são apenas uma referência externa: não serão incorporados, distribuídos, executados ou chamados pela biblioteca. Também não serão dependências do projeto ou da suíte obrigatória.

## Referência e escopo

Referência inicial observada no ambiente local: Python 3.11.17, scikit-learn 1.9.1, NumPy 2.4.6 e SciPy 1.17.1. A versão do scikit-learn deve ser fixada na geração dos fixtures e informada nos resultados. O ambiente `venv/` existente é local e serve somente para consultar a referência: não faz parte do projeto e não deve ser versionado ou distribuído.

Compatibilidade é por funcionalidade e versão de referência, não uma promessa de paridade integral. Cada recurso publicado deve registrar assinatura Python, equivalente Go, parâmetros, defaults, formatos, limitações e diferenças de comportamento. Resultados numéricos são equivalentes dentro de tolerâncias, não necessariamente bit a bit.

## Ordem das especificações

O roteiro apresenta as entregas. Os critérios de teste se aplicam desde a fundação. O contrato de regressão orienta a implementação dos pacotes seguintes. As extensões de classificação e validação cruzada descritas nesses pacotes entram depois do primeiro ciclo de benchmarks.

- [00 · Roteiro de implementação](specs/00-roteiro.md)
- [01 · Fundação e contratos](specs/01-fundacao.md)
- [02 · Testes e validação de compatibilidade](specs/02-testes-validacao.md)
- [03 · Contrato proposto para o primeiro fluxo](specs/03-contrato-regressao.md)
- [04 · random_state](specs/04-random-state.md)
- [05 · metrics e model_selection](specs/05-metrics-model-selection.md)
- [06 · preprocessing](specs/06-preprocessing.md)
- [07 · linear_model](specs/07-linear-model.md)
- [08 · pipeline](specs/08-pipeline.md)
- [09 · Salvar e carregar modelos para inferência](specs/09-persistencia.md)
- [10 · Benchmarks de comparação com scikit-learn](specs/10-benchmarks.md)
- [Matriz de compatibilidade inicial](compatibilidade.md)
- [Implementação atual](implementacao.md)

## Decisões e questões abertas

- **Pendente antes da publicação:** caminho público do módulo, organização proprietária e licença. `go.mod` usa o caminho local `go-learn` durante desenvolvimento.
- **Versão Go:** mínimo declarado `go 1.22`; validado localmente com Go 1.26.1.
- **Diretriz:** manter nomes conceituais e defaults Python; adaptar exportação, tipos estáticos e erros às regras do Go. Definir nomes de pacotes Go concretos na primeira implementação e registrar divergências, inclusive quando não preservarem underscores Python.
- **Benchmarks:** implementados em `benchmarks/`; a comparação com scikit-learn é externa. A baseline inicial está documentada.
- **Primeiro fluxo implementado:** dados densos `float64` e regressão de saída única. Classificação binária com LogisticRegression, métricas e pipeline também está implementada. Validação cruzada é a próxima etapa.
- **Dados e concorrência:** contratos públicos devem descrever cópia, aliasing, mutação e segurança concorrente; `Fit` muta o estimador e não é concorrente por padrão.
- **Paridade aleatória:** mesma semente não garante a mesma sequência entre Go e NumPy. Só prometer paridade exata se o gerador e o algoritmo forem compatíveis e testados; caso contrário documentar a diferença.
