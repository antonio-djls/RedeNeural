# Rede Neural MNIST em C++

![](mnist.png)

Implementação completa de uma rede neural artificial para reconhecimento de dígitos manuscritos do dataset MNIST.

Aviso !: Para extrair os dados do dataset contido no repositório recomendo acessar este repositório abaixo. Ele explica muito bem como funciona  o processo de extração e contém um projeto pronto para processamento do arquivo.
Link: <https://github.com/wichtounet/mnist>

## 🎯 Objetivo

Classificar imagens de dígitos manuscritos (0-9) com alta acurácia usando uma rede neural feedforward com backpropagation.

## 🏗️ Arquitetura da Rede
```
Entrada (784) → Oculta 1 (256) → Oculta 2 (128) → Saída (10)
     ↓              ↓               ↓              ↓
  28×28 pixels   Sigmoid        Sigmoid       Sigmoid
```
- **Camada de entrada**: 784 neurônios (28×28 pixels)
- **Camada oculta 1**: 256 neurônios com bias
- **Camada oculta 2**: 128 neurônios com bias  
- **Camada de saída**: 10 neurônios (um para cada dígito)

### ✅ Dataset MNIST

### ✅ Avaliação e Testes

- **Métricas de acurácia**: Treinamento e teste
- **Teste individual**: Verificação imagem por imagem
- **Probabilidades**: Confiança da rede para cada predição
- **Visualização de resultados**: ✓ CORRETO / ✗ ERRADO

### Exemplo de Saída
```
Valor real: 7 | Predição: 7 | ✓ CORRETO
Probabilidades: [0.03, 0.00, 0.00, 0.06, 0.00, 0.02, 0.00, 0.98, 0.01, 0.03]
```
## 🔧 Componentes Técnicos

### Estruturas de Dados

- `neuronio_entrada`: Neurônios da camada de entrada
- `neuronio_camadaoculta_*`: Neurônios das camadas ocultas
- `neuronio_saida`: Neurônios da camada de saída
- `MNIST_Image/Label`: Estruturas para o dataset
