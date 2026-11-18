# PA3 

## Parte 0: piso supervisionado

Implementação das duas referências supervisionadas solicitadas na Parte 0:

- `cnn_scratch`: CNN pequena, inicializada aleatoriamente;
- `resnet18`: ResNet-18 com pesos ImageNet-1K V1 e fine-tuning de toda a rede.

## Parte 3: galeria de erros e acertos

Depois de treinar ou disponibilizar o melhor checkpoint da Parte 2, gere os mapas
Grad-CAM com:

```bash
python -m src.parte3_gallery \
  --data-dir data/pa3_dados \
  --checkpoint outputs/parte1/fixmatch_aligned/best_model.pth \
  --output-dir outputs/parte3
```

O comando gera em `outputs/parte3`:

- `galeria_erros_gradcam.png`: três erros de maior confiança;
- `galeria_acertos_gradcam.png`: cinco acertos diversos por classe;
- `galeria_detalhes.csv`: verdade, previsão, confiança, tamanho da imagem e
  localização da saliência. Use o overlay para concluir se o modelo olhou para o
  animal ou para o fundo.