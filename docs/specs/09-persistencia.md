# Salvar e carregar modelos para inferência

Proposta de pacote público modelio, usando encoding/json e io da biblioteca padrão. O primeiro formato cobre LinearRegression de saída única, sozinha ou no pipeline com StandardScaler. O ponto de entrada de leitura devolve um preditor com Predict. Treinamento incremental e retomada de um treino interrompido exigiriam outro contrato.

Exemplo proposto em memória. Para salvar em disco, passar um arquivo aberto para escrita em Save e um arquivo aberto para leitura em LoadRegressor.

```go
var artifact bytes.Buffer
if err := modelio.Save(&artifact, fittedPipeline); err != nil {
    return err
}
loaded, err := modelio.LoadRegressor(&artifact)
if err != nil {
    return err
}
predictions, err := loaded.Predict(xTest)
```

O JSON v1 contém format, version, número de características e lista ordenada de etapas com nome, tipo, opções e estado ajustado. Para o scaler, salvar indicadores de centralização e escala e os vetores efetivamente ajustados. Para a regressão, salvar coeficientes e intercepto. Salvar o pipeline inteiro preserva as unidades em que o regressor recebeu os dados durante o treino. A semente pode descrever o experimento, mas os parâmetros ajustados são suficientes para prever nesse fluxo.

**Validação.** Save retorna erro para modelo sem ajuste ou etapa sem representação suportada. LoadRegressor valida formato, versão, tipos conhecidos, ordem, dimensões, escalas positivas e finitude antes de devolver o preditor. O formato descreve dados e reconhece apenas etapas registradas na biblioteca. Para substituir um arquivo existente com segurança, o chamador escreve um temporário no mesmo diretório, verifica Save e Close e então renomeia o temporário para o destino.

**Aceite.** Salvar e recarregar regressor isolado e pipeline, incluindo scaler com coluna constante e opções desativadas. Conferir previsões em entradas inéditas com as mesmas tolerâncias do estimador. Manter um artefato v1 em testdata para testar leitura por versões futuras. Cobrir JSON truncado, versão desconhecida, etapa incompatível, dimensão errada e falha de escrita. A primeira versão lê o formato próprio do go-learn. Importação de pickle e joblib fica fora desse contrato.
