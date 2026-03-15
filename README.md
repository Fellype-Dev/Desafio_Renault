# Sistema de Inspeção de Peças - Desafio Renault

Sistema completo para inspeção automatizada de peças utilizando visão computacional e deep learning.

## 🚀 Funcionalidades

### 📸 Coleta dos dados
- Captura em tempo real via câmera USB
- Detecção automática de objetos circulares
- Classificação manual das peças (Boa/Defeito)
- Recorte automático da região de interesse
- Estatísticas em tempo real

### 🧠 Treinamento de Modelo
- Interface para configurar parâmetros de treinamento
- Verificação automática do dataset
- Treinamento de CNN com visualização em tempo real
- Gráficos de acurácia e perda
- Salvamento automático do modelo

### 🔍 Inspeção Automatizada
- Classificação em tempo real
- Feedback visual com códigos de cores
- Estatísticas da sessão de inspeção
- Teste com imagens estáticas
- Log detalhado das inspeções

### ⚙️ Configurações
- Ajuste de parâmetros da câmera
- Configuração do modelo
- Informações do sistema

## 🛠️ Tecnologias Utilizadas

- **OpenCV**: Processamento de imagem e detecção de objetos
- **TensorFlow/Keras**: Deep learning e CNN
- **Tkinter**: Interface gráfica
- **HoughCircles**: Detecção de objetos circulares
- **Matplotlib**: Visualização de gráficos

## 📋 Instalação e Uso

### 1. Executar a Interface Gráfica (Recomendado)
```bash
python interface_grafica.py
```

### 2. Scripts Individuais (Opcional)
```bash
# Coleta de dados
python coleta_avancada.py

# Treinamento
python treinar_simples.py

# Inspeção
python inspecao_avancada.py

# Teste com imagem
python teste_final_modelo.py
```

## 📊 Arquitetura do Sistema

### Detecção de Objetos
- **HoughCircles** (OpenCV): Detecta objetos circulares
- Rápido e eficiente para peças circulares
- Não requer treinamento adicional

### Classificação
- **CNN (Convolutional Neural Network)**:
  - 3 camadas convolucionais
  - 2 camadas de pooling
  - 2 camadas densas
  - Classificação binária: Boa vs Defeito

## 🎯 Fluxo de Trabalho

1. **Coleta**: Use a aba "Coleta de Dados" para capturar imagens
2. **Treinamento**: Na aba "Treinamento", configure e treine o modelo
3. **Inspeção**: Use a aba "Inspeção" para classificação em tempo real

## 📁 Estrutura do Projeto

```
dataset_final/
├── boas/          # Imagens de peças boas
└── com_defeito/   # Imagens de peças com defeito

*.h5               # Modelos treinados
*.png              # Gráficos de diagnóstico
```

## ⚡ Performance

- **Detecção**: ~30 FPS em tempo real
- **Classificação**: Instantânea após detecção
- **Treinamento**: Depende do dataset e hardware

## 🔧 Requisitos de Hardware

- Câmera USB para coleta e inspeção
- Mínimo 4GB RAM para treinamento
- CPU multi-core recomendado

## 📝 Notas

- Sistema otimizado para peças circulares/cilíndricas
- Funciona bem em ambientes com iluminação controlada
- Dataset recomendado: mínimo 50 imagens por classe
