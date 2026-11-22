# Fundação e contratos

## Objetivo e dependências

Disponibilizar dados numéricos e contratos necessários aos primeiros estimadores, sem criar uma implementação genérica de NumPy. Não depende de outros módulos do projeto.

## Especificação

- Matriz densa bidimensional `float64`, em armazenamento contíguo com dimensões explícitas e indexação linha/coluna. Amostras em linhas, características em colunas. Definir construtores com cópia e, se necessário, compartilhamento explícito; proteger invariantes de tamanho e overflow.
- Alvos de regressão são vetores `[]float64`; rótulos de classificação serão representados por tipo explícito, inicialmente inteiros. Saída múltipla, tipos heterogêneos, dados esparsos e valores ausentes ficam fora do primeiro recorte, salvo suporte declarado pelo estimador.
- `Fit` altera o estado; cada chamada bem-sucedida substitui o ajuste anterior. Definir e testar o estado após falha antes de publicar: recomenda-se preservar o último ajuste válido. `Predict` e `Transform` antes de `Fit` retornam erro.
- Interfaces pequenas pertencem aos consumidores que precisam delas; separar regressão, classificação e transformação quando os tipos exigirem. Não impor métodos irrelevantes a todos os estimadores.
- Os nomes, defaults e semântica públicos seguem a referência Python, adaptados para identificadores Go exportados e retornos `error`. Opções devem distinguir parâmetro omitido de zero explícito; usar construtores com defaults e uma forma documentada de configurá-los.
- Validar tamanhos, quantidades de amostras e características, opções, valores não finitos e previsões com número incorreto de características. Especificar exceções por recurso caso a referência permita NaN. Não modificar dados de entrada salvo opção explicitamente documentada.
- `internal/linalg` fornece apenas operações necessárias. Para mínimos quadrados usar QR estável; detectar deficiência de posto e documentar erros ou a solução suportada. Como a referência pode produzir solução de norma mínima em casos deficientes, não declarar paridade nesses casos até implementá-la.
- Leituras concorrentes após ajuste são permitidas somente se cada estimador garantir ausência de mutação; `Fit` e leituras simultâneas não são seguros por padrão. Evitar geradores aleatórios globais compartilhados.

## Implementação inicial

O pacote público `matrix` implementa `Dense` em armazenamento contíguo row-major. `NewDense` e `NewDenseFromRows` copiam a entrada; `Values`, `Row` e `Clone` também retornam cópias. `At` segue a indexação de slices do Go e causa panic em índice inválido; `Set` e `Row` retornam erro. Shapes com dimensão zero são aceitas pela matriz, embora estimadores exijam dados não vazios.

Uma matriz densa ocupa `8 × rows × cols` bytes para os valores, além do cabeçalho e das alocações do runtime. Construção, clonagem, cópia de `Values` e transformação percorrem O(rows × cols) valores; `At` e `Set` são O(1); `Row` custa O(cols) porque devolve cópia. QR com pivoteamento custa O(rows × cols²) operações e O(rows × cols) memória adicional para a entrada de trabalho, assumindo rows ≥ cols.

O módulo usa `go 1.22` como versão mínima inicial e `go-learn` como caminho de módulo local durante desenvolvimento. O caminho público de importação e a licença continuam pendentes para publicação.

O primeiro solver usa QR de Householder com pivoteamento de colunas e limiar relativo `1e-12` na diagonal de R. Ele retorna erro em posto numericamente deficiente. Isso difere do solver por valores singulares do scikit-learn e limita a paridade nesses casos.

## Aceite e testes

Testar construção, indexação, cópia e aliasing, dimensões inválidas, overflow, preservação dos dados, ciclos de ajuste e previsões inválidas. Testar QR com sistemas conhecidos, mal condicionados e deficientes, medindo resíduos e tolerâncias. `go test ./...`, `go vet ./...` e `go test -race ./...` devem passar; testes de concorrência só afirmam garantias efetivamente publicadas. Documentar custos de memória e complexidade relevantes.
