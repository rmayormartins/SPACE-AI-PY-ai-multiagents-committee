import gradio as gr
from transformers import pipeline
import torch
from PIL import Image
import numpy as np
from collections import Counter

# Definição dos modelos disponíveis
MODELS = {
    'texto': [
        {'name': 'BERT', 'model': 'nlptown/bert-base-multilingual-uncased-sentiment', 'framework': 'pt'},
        {'name': 'RoBERTa', 'model': 'cardiffnlp/twitter-roberta-base-sentiment', 'framework': 'pt'},
        {'name': 'DistilBERT', 'model': 'distilbert-base-uncased-finetuned-sst-2-english', 'framework': 'pt'}
    ],
    'imagem': [
        {'name': 'ResNet', 'model': 'microsoft/resnet-50', 'framework': 'pt'},
        {'name': 'ViT', 'model': 'google/vit-base-patch16-224', 'framework': 'pt'},
        {'name': 'BEiT', 'model': 'microsoft/beit-base-patch16-224', 'framework': 'pt'}
    ],
    'qa': [
        {'name': 'RoBERTa-SQuAD', 'model': 'deepset/roberta-base-squad2', 'framework': 'pt'},
        {'name': 'BERT-SQuAD', 'model': 'deepset/bert-base-cased-squad2', 'framework': 'pt'},
        {'name': 'DistilBERT-SQuAD', 'model': 'distilbert-base-cased-distilled-squad', 'framework': 'pt'}
    ]
}

# Carregar agentes
def load_agents():
    agents = {
        'texto': {},
        'imagem': {},
        'qa': {}
    }
    
    print("Carregando agentes... Isso pode demorar alguns minutos.")
    
    for tipo in MODELS:
        for model_info in MODELS[tipo]:
            try:
                agents[tipo][model_info['name']] = pipeline(
                    'sentiment-analysis' if tipo == 'texto' else
                    'image-classification' if tipo == 'imagem' else
                    'question-answering',
                    model=model_info['model'],
                    framework=model_info['framework']
                )
                print(f"✓ Carregado {model_info['name']} para {tipo}")
            except Exception as e:
                print(f"✗ Erro ao carregar {model_info['name']} para {tipo}: {str(e)}")
    
    return agents

# Funções de processamento
def analise_texto(texto, bert, roberta, distilbert):
    if not texto:
        return "Por favor, digite algum texto para análise."
    if not any([bert, roberta, distilbert]):
        return "Por favor, selecione pelo menos um agente."
    
    try:
        resultados = []
        confianças = []
        detalhes_agentes = []
        
        agents_selecionados = [
            ('BERT', bert),
            ('RoBERTa', roberta),
            ('DistilBERT', distilbert)
        ]
        
        for agent_name, is_selected in agents_selecionados:
            if is_selected:
                resultado = agents['texto'][agent_name](texto)
                
                # Padronização dos rótulos
                sentimento = resultado[0]['label']
                # Convertendo diferentes formatos para POSITIVO/NEGATIVO
                if sentimento in ['POSITIVE', 'LABEL_4', 'LABEL_5', '5 stars', '4 stars']:
                    sentimento = "POSITIVO"
                elif sentimento in ['NEGATIVE', 'LABEL_1', 'LABEL_2', '1 star', '2 stars']:
                    sentimento = "NEGATIVO"
                elif sentimento in ['LABEL_3', '3 stars']:
                    # Para casos neutros, decidimos baseado na confiança
                    sentimento = "POSITIVO" if resultado[0]['score'] > 0.5 else "NEGATIVO"
                
                confiança = float(resultado[0]['score'])
                
                resultados.append(sentimento)
                confianças.append(confiança)
                detalhes_agentes.append(f"{agent_name}: {sentimento} ({confiança:.2%})")
        
        # Análise dos resultados
        sentimento_final = max(set(resultados), key=resultados.count)
        confianca_media = np.mean(confianças)
        concordancia = resultados.count(sentimento_final) / len(resultados)
        distribuicao = dict(Counter(resultados))
        
        return (
            "📊 Resultado Final:\n"
            f"Sentimento: {sentimento_final}\n"
            f"Confiança média: {confianca_media:.2%}\n"
            f"Taxa de concordância: {concordancia:.2%}\n\n"
            "🤖 Detalhes por Agente:\n"
            f"{chr(10).join(detalhes_agentes)}\n\n"
            "📈 Distribuição dos votos:\n"
            f"{', '.join(f'{k}: {v}' for k, v in distribuicao.items())}"
        )
    except Exception as e:
        return f"Erro na análise: {str(e)}"

def classifica_imagem(imagem, resnet, vit, beit):
    if imagem is None:
        return "Por favor, faça upload de uma imagem."
    if not any([resnet, vit, beit]):
        return "Por favor, selecione pelo menos um agente."
    
    try:
        resultados = []
        confianças = []
        detalhes_agentes = []
        
        agents_selecionados = [
            ('ResNet', resnet),
            ('ViT', vit),
            ('BEiT', beit)
        ]
        
        for agent_name, is_selected in agents_selecionados:
            if is_selected:
                resultado = agents['imagem'][agent_name](imagem)
                
                classificacao = resultado[0]['label']
                confiança = float(resultado[0]['score'])
                
                resultados.append(classificacao)
                confianças.append(confiança)
                detalhes_agentes.append(f"{agent_name}: {classificacao} ({confiança:.2%})")
        
        classificacao_final = max(set(resultados), key=resultados.count)
        confianca_media = np.mean(confianças)
        concordancia = resultados.count(classificacao_final) / len(resultados)
        distribuicao = dict(Counter(resultados))
        
        return (
            "📊 Resultado Final:\n"
            f"Classificação: {classificacao_final}\n"
            f"Confiança média: {confianca_media:.2%}\n"
            f"Taxa de concordância: {concordancia:.2%}\n\n"
            "🤖 Detalhes por Agente:\n"
            f"{chr(10).join(detalhes_agentes)}\n\n"
            "📈 Distribuição das classificações:\n"
            f"{', '.join(f'{k}: {v}' for k, v in distribuicao.items())}"
        )
    except Exception as e:
        return f"Erro na classificação: {str(e)}"

def responde_pergunta(pergunta, contexto, roberta_squad, bert_squad, distilbert_squad):
    if not pergunta or not contexto:
        return "Por favor, forneça tanto a pergunta quanto o contexto."
    if not any([roberta_squad, bert_squad, distilbert_squad]):
        return "Por favor, selecione pelo menos um agente."
    
    try:
        resultados = []
        confianças = []
        detalhes_agentes = []
        
        agents_selecionados = [
            ('RoBERTa-SQuAD', roberta_squad),
            ('BERT-SQuAD', bert_squad),
            ('DistilBERT-SQuAD', distilbert_squad)
        ]
        
        for agent_name, is_selected in agents_selecionados:
            if is_selected:
                resultado = agents['qa'][agent_name](
                    question=pergunta,
                    context=contexto,
                    max_answer_len=50,
                    handle_impossible_answer=True
                )
                
                resposta = resultado['answer']
                confiança = float(resultado['score'])
                
                resultados.append(resposta)
                confianças.append(confiança)
                detalhes_agentes.append(f"{agent_name}: {resposta} ({confiança:.2%})")
        
        resposta_final = max(set(resultados), key=resultados.count)
        confianca_media = np.mean(confianças)
        concordancia = resultados.count(resposta_final) / len(resultados)
        
        nota = (
            'Alta confiança!' if confianca_media > 0.8 else
            'Confiança moderada.' if confianca_media > 0.5 else
            'Baixa confiança - considere reformular a pergunta.'
        )
        
        return (
            "📊 Resultado Final:\n"
            f"Resposta: {resposta_final}\n"
            f"Confiança média: {confianca_media:.2%}\n"
            f"Taxa de concordância: {concordancia:.2%}\n\n"
            "🤖 Detalhes por Agente:\n"
            f"{chr(10).join(detalhes_agentes)}\n\n"
            f"💡 Nota: {nota}"
        )
    except Exception as e:
        return f"Erro ao processar a pergunta: {str(e)}"

# Carregar os agentes
print("Inicializando sistema...")
agents = load_agents()
print("Sistema pronto!")

# Interface Gradio
with gr.Blocks(title="Multi-Agent AI Committee") as demo:
    gr.Markdown("# 🤖 Multi-Agent AI Committee")
    gr.Markdown("""
    <p>Ramon Mayor Martins: <a href="https://rmayormartins.github.io/" target="_blank">Website</a> | <a href="https://huggingface.co/rmayormartins" target="_blank">Spaces</a> | <a href="https://github.com/rmayormartins" target="_blank">Github</a></p>
    """)
    gr.Markdown("Selecione os agentes que deseja usar para cada tarefa e veja como eles trabalham em conjunto!")
    
    with gr.Tab("📝 Análise de Texto"):
        with gr.Row():
            with gr.Column():
                text_input = gr.Textbox(
                    label="Digite o texto para análise",
                    placeholder="Digite aqui o texto que você quer analisar...",
                    lines=3
                )
                with gr.Group():
                    gr.Markdown("Selecione os agentes:")
                    bert_check = gr.Checkbox(label="BERT", value=True)
                    roberta_check = gr.Checkbox(label="RoBERTa")
                    distilbert_check = gr.Checkbox(label="DistilBERT")
                text_button = gr.Button("Analisar Sentimento")
            with gr.Column():
                text_output = gr.Textbox(label="Resultado", lines=10)
        text_button.click(
            analise_texto,
            inputs=[text_input, bert_check, roberta_check, distilbert_check],
            outputs=text_output
        )
    
    with gr.Tab("🖼️ Classificação de Imagem"):
        with gr.Row():
            with gr.Column():
                image_input = gr.Image(type="pil", label="Upload da Imagem")
                with gr.Group():
                    gr.Markdown("Selecione os agentes:")
                    resnet_check = gr.Checkbox(label="ResNet", value=True)
                    vit_check = gr.Checkbox(label="ViT")
                    beit_check = gr.Checkbox(label="BEiT")
                image_button = gr.Button("Classificar Imagem")
            with gr.Column():
                image_output = gr.Textbox(label="Resultado", lines=10)
        image_button.click(
            classifica_imagem,
            inputs=[image_input, resnet_check, vit_check, beit_check],
            outputs=image_output
        )
    
    with gr.Tab("❓ Perguntas e Respostas"):
        with gr.Row():
            with gr.Column():
                question_input = gr.Textbox(
                    label="Digite sua pergunta",
                    placeholder="Ex: Qual é a capital do Brasil?"
                )
                context_input = gr.Textbox(
                    label="Digite o contexto",
                    placeholder="Ex: Brasília é a capital do Brasil, localizada no Distrito Federal...",
                    lines=3
                )
                with gr.Group():
                    gr.Markdown("Selecione os agentes:")
                    roberta_squad_check = gr.Checkbox(label="RoBERTa-SQuAD", value=True)
                    bert_squad_check = gr.Checkbox(label="BERT-SQuAD")
                    distilbert_squad_check = gr.Checkbox(label="DistilBERT-SQuAD")
                qa_button = gr.Button("Obter Resposta")
            with gr.Column():
                qa_output = gr.Textbox(label="Resultado", lines=10)
        qa_button.click(
            responde_pergunta,
            inputs=[question_input, context_input, roberta_squad_check, bert_squad_check, distilbert_squad_check],
            outputs=qa_output
        )

    gr.Markdown("""
    ### 📋 Instruções:
    1. Selecione um ou mais agentes em cada aba
    2. Forneça os dados de entrada (texto, imagem ou pergunta+contexto)
    3. Veja como os diferentes agentes trabalham juntos!
    
    ### 🔍 Sobre os Agentes:
    - **BERT**: Modelo base robusto
    - **RoBERTa**: Otimizado para melhor performance
    - **DistilBERT**: Versão mais leve e rápida
    
    ### 📊 Métricas:
    - **Confiança média**: Média da confiança de todos os agentes
    - **Taxa de concordância**: Quanto os agentes concordam entre si
    - **Distribuição**: Como os votos se dividem entre as opções
    """)

# Iniciar a interface
demo.launch()