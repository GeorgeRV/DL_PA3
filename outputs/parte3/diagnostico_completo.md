# Diagnóstico completo da Parte 3

## Metodologia

- Modelo analisado: ResNet-18 treinado pelo FixMatch aligned da Parte 1.
- Checkpoint: `outputs/parte1/fixmatch_aligned/best_model.pth`.
- Explicabilidade: Grad-CAM aplicado à última unidade residual (`layer4[-1]`).
- A previsão foi usada como classe-alvo do mapa.
- Os erros exibidos são os três erros de maior confiança no `test_public`.
- Os acertos são os cinco exemplos de maior confiança, diversificados por classe.
- Todas as imagens foram redimensionadas pelo pipeline para 224 x 224.
- Bounding boxes do animal não estão disponíveis no CSV fornecido; portanto, a análise visual compara diretamente o mapa com o animal visível na imagem.

## Galerias

- [Erros com Grad-CAM](galeria_erros_gradcam.png)
- [Erros sem Grad-CAM](galeria_erros_sem_gradcam.png)
- [Acertos com Grad-CAM](galeria_acertos_gradcam.png)
- [Acertos sem Grad-CAM](galeria_acertos_sem_gradcam.png)
- [Tabela detalhada](galeria_detalhes.csv)

## Erros

### ece8e099eb242e5d

- Verdade: **dog**
- Predição: **chicken**
- Confiança: **99.83%**
- Dimensão original: 512 x 367 pixels
- Entrada do modelo: 224 x 224 pixels
- Fração da saliência nos 20% mais ativos: 38.01%
- Região dominante aproximada: centro / meio
- Diagnóstico: A imagem mostra um cachorro próximo de uma construção iluminada. O Grad-CAM destaca principalmente o fundo à direita e a vegetação iluminada, com menos evidência sobre o corpo do animal.
- Comparação sem mapa: [imagem original](erros_sem_saliencia/ece8e099eb242e5d_original.png)

### 2a9136fc4f93cea5

- Verdade: **american_black_bear**
- Predição: **eastern_fox_squirrel**
- Confiança: **99.16%**
- Dimensão original: 512 x 367 pixels
- Entrada do modelo: 224 x 224 pixels
- Fração da saliência nos 20% mais ativos: 46.88%
- Região dominante aproximada: centro / inferior
- Diagnóstico: A cena é dominada por vegetação e solo florestal. A saliência concentra-se no ambiente inferior e central, sugerindo que o modelo pode estar usando textura e contexto de floresta em vez de uma região clara sobre o urso.
- Comparação sem mapa: [imagem original](erros_sem_saliencia/2a9136fc4f93cea5_original.png)

### ef959725a770d810

- Verdade: **eastern_chipmunk**
- Predição: **eastern_fox_squirrel**
- Confiança: **98.55%**
- Dimensão original: 512 x 367 pixels
- Entrada do modelo: 224 x 224 pixels
- Fração da saliência nos 20% mais ativos: 32.87%
- Região dominante aproximada: centro / meio
- Diagnóstico: A imagem tem vegetação densa e pouca distinção visual entre o pequeno animal e o fundo. O mapa distribui a ativação por troncos e folhas, sem uma região isolada que confirme atenção consistente ao chipmunk.
- Comparação sem mapa: [imagem original](erros_sem_saliencia/ef959725a770d810_original.png)

## Acertos

A pergunta inversa é observar quais acertos têm saliência sobre o animal e quais parecem apoiar-se no fundo.

### e0f23fd97fb21b3e

- Classe prevista e verdadeira: **virginia_opossum**
- Confiança: **99.99%**
- Dimensão original: 512 x 270 pixels
- Entrada do modelo: 224 x 224 pixels
- Fração da saliência nos 20% mais ativos: 45.95%
- Região dominante aproximada: centro / meio
- Diagnóstico: O mapa está centrado na região do gambá, sendo o caso mais claro desta galeria de acerto sobre o animal.
- Comparação sem mapa: [imagem original](acertos_sem_saliencia/e0f23fd97fb21b3e_original.png)

### 2a0873bcaf6cbfbd

- Classe prevista e verdadeira: **white_tailed_deer**
- Confiança: **99.98%**
- Dimensão original: 512 x 367 pixels
- Entrada do modelo: 224 x 224 pixels
- Fração da saliência nos 20% mais ativos: 48.18%
- Região dominante aproximada: centro / inferior
- Diagnóstico: A ativação aparece sobre o cervo e seus contornos superiores. O fundo também recebe alguma ativação, mas o corpo do animal é uma referência central.
- Comparação sem mapa: [imagem original](acertos_sem_saliencia/2a0873bcaf6cbfbd_original.png)

### c80dcb0a6cacc16b

- Classe prevista e verdadeira: **eastern_fox_squirrel**
- Confiança: **99.98%**
- Dimensão original: 512 x 367 pixels
- Entrada do modelo: 224 x 224 pixels
- Fração da saliência nos 20% mais ativos: 34.89%
- Região dominante aproximada: centro / meio
- Diagnóstico: A saliência é ampla sobre vegetação e solo; não é possível afirmar que o modelo usou somente o esquilo, pois o contexto florestal recebe destaque.
- Comparação sem mapa: [imagem original](acertos_sem_saliencia/c80dcb0a6cacc16b_original.png)

### 784c50e9bcd996a3

- Classe prevista e verdadeira: **eastern_gray_squirrel**
- Confiança: **99.95%**
- Dimensão original: 512 x 270 pixels
- Entrada do modelo: 224 x 224 pixels
- Fração da saliência nos 20% mais ativos: 44.87%
- Região dominante aproximada: centro / meio
- Diagnóstico: O mapa cobre o tronco, o animal e a vegetação ao redor. O tronco parece particularmente relevante, indicando possível uso conjunto de animal e fundo.
- Comparação sem mapa: [imagem original](acertos_sem_saliencia/784c50e9bcd996a3_original.png)

### 7798c218eda5a8b1

- Classe prevista e verdadeira: **striped_skunk**
- Confiança: **99.94%**
- Dimensão original: 512 x 367 pixels
- Entrada do modelo: 224 x 224 pixels
- Fração da saliência nos 20% mais ativos: 42.92%
- Região dominante aproximada: centro / meio
- Diagnóstico: A ativação fica sobre o gambá-listrado, mas também se espalha pelo fundo azulado. O resultado é positivo, porém não totalmente isolado no animal.
- Comparação sem mapa: [imagem original](acertos_sem_saliencia/7798c218eda5a8b1_original.png)

## Síntese

- Os três erros são exemplos de alta confiança, portanto não parecem falhas próximas da fronteira de decisão.
- Há evidência visual de possível dependência de fundo em pelo menos dois erros de ambiente florestal.
- Nos acertos, a atenção é mais claramente localizada no animal em `virginia_opossum` e `white_tailed_deer`; os outros acertos mostram mistura entre animal e contexto.
- A ausência de bounding boxes impede medir quantos pixels da saliência recaíram formalmente dentro do animal. A conclusão final deve priorizar a comparação visual das versões com e sem mapa.
