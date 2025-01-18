---
title: ai-multiagents-committee
emoji: 🖥️🤖🤖🤖
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: "4.8.0"
app_file: app.py
pinned: false
---

# Multi-Agent AI Committee
Um sistema interativo de múltiplos agentes de IA que trabalham em conjunto para análise de texto, classificação de imagens e perguntas e respostas.

## Como Usar
O sistema oferece três funcionalidades principais:

1. **Análise de Sentimento**:
   - Digite um texto
   - Selecione os agentes (BERT, RoBERTa, DistilBERT)
   - Receba uma análise consolidada do sentimento

2. **Classificação de Imagens**:
   - Faça upload de uma imagem
   - Selecione os agentes (ResNet, ViT, BEiT)
   - Obtenha classificações detalhadas

3. **Perguntas e Respostas**:
   - Digite uma pergunta e forneça um contexto
   - Selecione os agentes (RoBERTa-SQuAD, BERT-SQuAD, DistilBERT-SQuAD)
   - Receba respostas baseadas no contexto

## Detalhes Técnicos
O projeto utiliza diversos modelos de IA de última geração:

### Análise de Texto:
- **BERT**: Modelo base multilingual para análise de sentimento
- **RoBERTa**: Versão otimizada do BERT para melhor performance
- **DistilBERT**: Versão compacta e rápida do BERT

### Classificação de Imagens:
- **ResNet**: Rede neural convolucional profunda
- **ViT**: Vision Transformer para processamento de imagens
- **BEiT**: BERT for Image Transformation

### Sistema de Perguntas e Respostas:
- **RoBERTa-SQuAD**: Otimizado para compreensão de texto
- **BERT-SQuAD**: Treinado no dataset SQuAD
- **DistilBERT-SQuAD**: Versão eficiente para QA

## Métricas e Avaliação
- Confiança média dos agentes
- Taxa de concordância entre agentes
- Distribuição dos resultados
- Análise detalhada por agente

## Informações Adicionais
- Desenvolvido por Ramon Mayor Martins (2025)
- E-mail: [rmayormartins@gmail.com](mailto:rmayormartins@gmail.com)
- Homepage: [https://rmayormartins.github.io/](https://rmayormartins.github.io/)
- Twitter: [@rmayormartins](https://twitter.com/rmayormartins)
- GitHub: [https://github.com/rmayormartins](https://github.com/rmayormartins)

## Notas
- O sistema utiliza uma abordagem de comitê, onde múltiplos agentes trabalham juntos
- As decisões são baseadas em votação majoritária e média de confiança
- Implementado em Python usando Transformers e Gradio
- Hospedado no Hugging Face Spaces