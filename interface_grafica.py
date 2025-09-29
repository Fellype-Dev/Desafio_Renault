import tkinter as tk
from tkinter import ttk, messagebox, filedialog, scrolledtext
import cv2
import numpy as np
import os
import time
import threading
from PIL import Image, ImageTk
from tensorflow.keras.models import load_model
from tensorflow.keras import layers, models, Input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import subprocess
import sys

class InterfaceRenault:
    def __init__(self, root):
        self.root = root
        self.root.title("Sistema de Inspeção de Peças - Renault")
        self.root.geometry("1200x800")
        
        # Variáveis de controle
        self.camera = None
        self.is_collecting = False
        self.is_inspecting = False
        self.model = None
        self.thread_active = False
        
        # Configurações
        self.IMG_HEIGHT = 128
        self.IMG_WIDTH = 128
        self.CAMERA_INDEX = 0
        self.current_model_path = "modelo_pecas_pistao.h5"
        
        # Inicializar estatísticas
        self.total_inspecoes = 0
        self.pecas_boas = 0
        self.pecas_defeitos = 0
        
        # Sistema de debounce para estatísticas
        self.ultima_classificacao = None
        self.tempo_classificacao = 0
        self.frames_mesma_classificacao = 0
        self.FRAMES_NECESSARIOS = 30  # ~1 segundo a 30 FPS
        self.classificacao_ja_contada = False
        
        # Criar pastas se não existirem
        os.makedirs("dataset_final/boas", exist_ok=True)
        os.makedirs("dataset_final/com_defeito", exist_ok=True)
        
        self.setup_ui()
        
    def setup_ui(self):
        # Notebook (abas)
        notebook = ttk.Notebook(self.root)
        notebook.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Aba 1: Coleta de Dados
        self.frame_coleta = ttk.Frame(notebook)
        notebook.add(self.frame_coleta, text="📸 Coleta de Dados")
        self.setup_coleta_tab()
        
        # Aba 2: Treinamento
        self.frame_treinamento = ttk.Frame(notebook)
        notebook.add(self.frame_treinamento, text="🧠 Treinamento")
        self.setup_treinamento_tab()
        
        # Aba 3: Inspeção
        self.frame_inspecao = ttk.Frame(notebook)
        notebook.add(self.frame_inspecao, text="🔍 Inspeção")
        self.setup_inspecao_tab()
        
        # Aba 4: Configurações
        self.frame_config = ttk.Frame(notebook)
        notebook.add(self.frame_config, text="⚙️ Configurações")
        self.setup_config_tab()
        
    def setup_coleta_tab(self):
        # Frame principal
        main_frame = ttk.Frame(self.frame_coleta)
        main_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Frame esquerdo - controles
        left_frame = ttk.Frame(main_frame)
        left_frame.pack(side="left", fill="y", padx=(0, 10))
        
        ttk.Label(left_frame, text="Coleta de Dataset", font=("Arial", 14, "bold")).pack(pady=10)
        
        # Botões de controle
        self.btn_iniciar_coleta = ttk.Button(left_frame, text="📹 Iniciar Câmera", 
                                           command=self.iniciar_coleta)
        self.btn_iniciar_coleta.pack(pady=5, fill="x")
        
        self.btn_parar_coleta = ttk.Button(left_frame, text="⏹️ Parar Câmera", 
                                         command=self.parar_coleta, state="disabled")
        self.btn_parar_coleta.pack(pady=5, fill="x")
        
        ttk.Separator(left_frame, orient="horizontal").pack(fill="x", pady=10)
        
        # Botões para salvar imagens
        self.btn_salvar_boa = ttk.Button(left_frame, text="✅ Salvar como BOA", 
                                       command=self.salvar_boa, state="disabled")
        self.btn_salvar_boa.pack(pady=5, fill="x")
        
        self.btn_salvar_defeito = ttk.Button(left_frame, text="❌ Salvar como DEFEITO", 
                                           command=self.salvar_defeito, state="disabled")
        self.btn_salvar_defeito.pack(pady=5, fill="x")
        
        ttk.Separator(left_frame, orient="horizontal").pack(fill="x", pady=10)
        
        # Estatísticas
        ttk.Label(left_frame, text="Estatísticas:", font=("Arial", 10, "bold")).pack()
        self.label_boas = ttk.Label(left_frame, text="Boas: 0")
        self.label_boas.pack()
        self.label_defeitos = ttk.Label(left_frame, text="Defeitos: 0")
        self.label_defeitos.pack()
        
        # Frame direito - vídeo
        right_frame = ttk.Frame(main_frame)
        right_frame.pack(side="right", fill="both", expand=True)
        
        self.label_video_coleta = ttk.Label(right_frame, text="Câmera não iniciada", 
                                          background="gray", foreground="white")
        self.label_video_coleta.pack(fill="both", expand=True)
        
        # Log de atividades
        ttk.Label(right_frame, text="Log de Atividades:").pack(anchor="w", pady=(10,0))
        self.log_coleta = scrolledtext.ScrolledText(right_frame, height=6)
        self.log_coleta.pack(fill="x", pady=5)
        
    def setup_treinamento_tab(self):
        main_frame = ttk.Frame(self.frame_treinamento)
        main_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Frame esquerdo - controles
        left_frame = ttk.Frame(main_frame)
        left_frame.pack(side="left", fill="y", padx=(0, 10))
        
        ttk.Label(left_frame, text="Treinamento do Modelo", font=("Arial", 14, "bold")).pack(pady=10)
        
        # Configurações de treinamento
        ttk.Label(left_frame, text="Épocas:").pack(anchor="w")
        self.spin_epochs = ttk.Spinbox(left_frame, from_=1, to=100, value=30, width=10)
        self.spin_epochs.pack(anchor="w", pady=(0,10))
        
        ttk.Label(left_frame, text="Batch Size:").pack(anchor="w")
        self.spin_batch = ttk.Spinbox(left_frame, from_=1, to=32, value=8, width=10)
        self.spin_batch.pack(anchor="w", pady=(0,10))
        
        ttk.Label(left_frame, text="Nome do Modelo:").pack(anchor="w")
        self.entry_model_name = ttk.Entry(left_frame)
        self.entry_model_name.insert(0, "modelo_treinado.h5")
        self.entry_model_name.pack(fill="x", pady=(0,10))
        
        # Botões
        self.btn_verificar_dataset = ttk.Button(left_frame, text="📊 Verificar Dataset", 
                                              command=self.verificar_dataset)
        self.btn_verificar_dataset.pack(fill="x", pady=5)
        
        self.btn_iniciar_treinamento = ttk.Button(left_frame, text="🚀 Iniciar Treinamento", 
                                                command=self.iniciar_treinamento)
        self.btn_iniciar_treinamento.pack(fill="x", pady=5)
        
        self.btn_parar_treinamento = ttk.Button(left_frame, text="⏹️ Parar Treinamento", 
                                              command=self.parar_treinamento, state="disabled")
        self.btn_parar_treinamento.pack(fill="x", pady=5)
        
        # Frame direito - gráficos e logs
        right_frame = ttk.Frame(main_frame)
        right_frame.pack(side="right", fill="both", expand=True)
        
        # Área para gráficos
        self.frame_graficos = ttk.Frame(right_frame)
        self.frame_graficos.pack(fill="both", expand=True)
        
        # Log de treinamento
        ttk.Label(right_frame, text="Log de Treinamento:").pack(anchor="w", pady=(10,0))
        self.log_treinamento = scrolledtext.ScrolledText(right_frame, height=8)
        self.log_treinamento.pack(fill="x", pady=5)
        
        # Barra de progresso
        self.progress_treinamento = ttk.Progressbar(right_frame, mode='indeterminate')
        self.progress_treinamento.pack(fill="x", pady=5)
        
    def setup_inspecao_tab(self):
        main_frame = ttk.Frame(self.frame_inspecao)
        main_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Frame esquerdo - controles
        left_frame = ttk.Frame(main_frame)
        left_frame.pack(side="left", fill="y", padx=(0, 10))
        
        ttk.Label(left_frame, text="Inspeção em Tempo Real", font=("Arial", 14, "bold")).pack(pady=10)
        
        # Seleção do modelo
        ttk.Label(left_frame, text="Modelo para Inspeção:").pack(anchor="w")
        self.combo_modelo = ttk.Combobox(left_frame, state="readonly")
        self.combo_modelo.pack(fill="x", pady=(0,10))
        self.atualizar_lista_modelos()
        
        # Botões
        self.btn_carregar_modelo = ttk.Button(left_frame, text="📂 Carregar Modelo", 
                                            command=self.carregar_modelo)
        self.btn_carregar_modelo.pack(fill="x", pady=5)
        
        self.btn_iniciar_inspecao = ttk.Button(left_frame, text="🔍 Iniciar Inspeção", 
                                             command=self.iniciar_inspecao, state="disabled")
        self.btn_iniciar_inspecao.pack(fill="x", pady=5)
        
        self.btn_parar_inspecao = ttk.Button(left_frame, text="⏹️ Parar Inspeção", 
                                           command=self.parar_inspecao, state="disabled")
        self.btn_parar_inspecao.pack(fill="x", pady=5)
        
        ttk.Separator(left_frame, orient="horizontal").pack(fill="x", pady=10)
        
        # Teste com imagem
        ttk.Label(left_frame, text="Teste com Imagem:", font=("Arial", 10, "bold")).pack(anchor="w")
        self.btn_testar_imagem = ttk.Button(left_frame, text="📁 Selecionar Imagem", 
                                          command=self.testar_imagem, state="disabled")
        self.btn_testar_imagem.pack(fill="x", pady=5)
        
        # Estatísticas de inspeção
        ttk.Separator(left_frame, orient="horizontal").pack(fill="x", pady=10)
        ttk.Label(left_frame, text="Estatísticas da Sessão:", font=("Arial", 10, "bold")).pack()
        self.label_total_inspecoes = ttk.Label(left_frame, text="Total: 0")
        self.label_total_inspecoes.pack()
        self.label_pecas_boas = ttk.Label(left_frame, text="Boas: 0")
        self.label_pecas_boas.pack()
        self.label_pecas_defeitos = ttk.Label(left_frame, text="Defeitos: 0")
        self.label_pecas_defeitos.pack()
        
        # Frame direito - vídeo
        right_frame = ttk.Frame(main_frame)
        right_frame.pack(side="right", fill="both", expand=True)
        
        self.label_video_inspecao = ttk.Label(right_frame, text="Inspeção não iniciada", 
                                            background="gray", foreground="white")
        self.label_video_inspecao.pack(fill="both", expand=True)
        
        # Resultado atual
        self.label_resultado = ttk.Label(right_frame, text="", font=("Arial", 12, "bold"))
        self.label_resultado.pack(pady=10)
        
        # Log de inspeção
        ttk.Label(right_frame, text="Log de Inspeção:").pack(anchor="w", pady=(10,0))
        self.log_inspecao = scrolledtext.ScrolledText(right_frame, height=6)
        self.log_inspecao.pack(fill="x", pady=5)
        
    def setup_config_tab(self):
        main_frame = ttk.Frame(self.frame_config)
        main_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        ttk.Label(main_frame, text="Configurações do Sistema", font=("Arial", 14, "bold")).pack(pady=10)
        
        # Configurações da câmera
        camera_frame = ttk.LabelFrame(main_frame, text="Câmera")
        camera_frame.pack(fill="x", pady=10)
        
        ttk.Label(camera_frame, text="Índice da Câmera:").grid(row=0, column=0, sticky="w", padx=5, pady=5)
        self.spin_camera = ttk.Spinbox(camera_frame, from_=0, to=10, value=self.CAMERA_INDEX, width=10)
        self.spin_camera.grid(row=0, column=1, sticky="w", padx=5, pady=5)
        
        # Configurações do modelo
        modelo_frame = ttk.LabelFrame(main_frame, text="Modelo")
        modelo_frame.pack(fill="x", pady=10)
        
        ttk.Label(modelo_frame, text="Altura da Imagem:").grid(row=0, column=0, sticky="w", padx=5, pady=5)
        self.spin_height = ttk.Spinbox(modelo_frame, from_=64, to=512, value=self.IMG_HEIGHT, width=10)
        self.spin_height.grid(row=0, column=1, sticky="w", padx=5, pady=5)
        
        ttk.Label(modelo_frame, text="Largura da Imagem:").grid(row=1, column=0, sticky="w", padx=5, pady=5)
        self.spin_width = ttk.Spinbox(modelo_frame, from_=64, to=512, value=self.IMG_WIDTH, width=10)
        self.spin_width.grid(row=1, column=1, sticky="w", padx=5, pady=5)
        
        # Configurações de contagem
        contagem_frame = ttk.LabelFrame(main_frame, text="Contagem de Objetos")
        contagem_frame.pack(fill="x", pady=10)
        
        ttk.Label(contagem_frame, text="Tempo para contagem (segundos):").grid(row=0, column=0, sticky="w", padx=5, pady=5)
        self.spin_tempo_contagem = ttk.Spinbox(contagem_frame, from_=0.5, to=5.0, increment=0.5, value=1.0, width=10)
        self.spin_tempo_contagem.grid(row=0, column=1, sticky="w", padx=5, pady=5)
        
        ttk.Label(contagem_frame, text="Explicação: Tempo que o objeto deve manter\na mesma classificação para ser contado", 
                 justify="left", font=("Arial", 8)).grid(row=1, column=0, columnspan=2, sticky="w", padx=5, pady=2)
        
        # Botões
        btn_frame = ttk.Frame(main_frame)
        btn_frame.pack(fill="x", pady=20)
        
        ttk.Button(btn_frame, text="💾 Salvar Configurações", 
                  command=self.salvar_configuracoes).pack(side="left", padx=5)
        ttk.Button(btn_frame, text="🔄 Aplicar Configurações", 
                  command=self.aplicar_configuracoes).pack(side="left", padx=5)
        ttk.Button(btn_frame, text="📁 Abrir Pasta do Projeto", 
                  command=self.abrir_pasta_projeto).pack(side="left", padx=5)
        
        # Informações do sistema
        info_frame = ttk.LabelFrame(main_frame, text="Informações")
        info_frame.pack(fill="x", pady=10)
        
        info_text = f"""
Sistema de Inspeção de Peças - Versão 1.0
Desenvolvido para o Desafio Renault

Tecnologias utilizadas:
• OpenCV para processamento de imagem
• TensorFlow/Keras para deep learning
• Tkinter para interface gráfica

Pasta do projeto: {os.path.abspath('.')}
        """
        ttk.Label(info_frame, text=info_text, justify="left").pack(padx=10, pady=10)
        
    # Métodos da aba Coleta
    def iniciar_coleta(self):
        try:
            # Parar qualquer câmera anterior
            if hasattr(self, 'camera') and self.camera is not None:
                self.camera.release()
                time.sleep(0.5)
                
            self.CAMERA_INDEX = int(self.spin_camera.get())
            self.log_message("coleta", f"Tentando abrir câmera no índice {self.CAMERA_INDEX}...")
            
            self.camera = cv2.VideoCapture(self.CAMERA_INDEX)
            
            # Tentar várias vezes abrir a câmera
            tentativas = 3
            for i in range(tentativas):
                if self.camera.isOpened():
                    break
                self.camera.release()
                time.sleep(0.5)
                self.camera = cv2.VideoCapture(self.CAMERA_INDEX)
                self.log_message("coleta", f"Tentativa {i+1} de {tentativas} para abrir câmera...")
            
            if not self.camera.isOpened():
                self.log_message("coleta", f"ERRO: Não foi possível abrir a câmera {self.CAMERA_INDEX}")
                messagebox.showerror("Erro", f"Não foi possível abrir a câmera {self.CAMERA_INDEX}\n\nVerifique se:\n- A câmera está conectada\n- Nenhum outro programa está usando a câmera\n- O índice da câmera está correto nas configurações")
                return
                
            # Configurar propriedades da câmera
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.camera.set(cv2.CAP_PROP_FPS, 30)
            
            # Testar se consegue capturar um frame
            ret, test_frame = self.camera.read()
            if not ret or test_frame is None:
                self.log_message("coleta", "ERRO: Não foi possível capturar frame da câmera")
                self.camera.release()
                messagebox.showerror("Erro", "A câmera foi aberta mas não consegue capturar imagens")
                return
                
            self.is_collecting = True
            self.thread_active = True
            self.btn_iniciar_coleta.config(state="disabled")
            self.btn_parar_coleta.config(state="normal")
            self.btn_salvar_boa.config(state="normal")
            self.btn_salvar_defeito.config(state="normal")
            
            self.log_message("coleta", "✅ Câmera iniciada com sucesso!")
            self.atualizar_estatisticas_coleta()
            
            # Iniciar thread de captura
            threading.Thread(target=self.capturar_frames_coleta, daemon=True).start()
            
        except Exception as e:
            self.log_message("coleta", f"ERRO: {str(e)}")
            messagebox.showerror("Erro", f"Erro ao iniciar câmera: {str(e)}")
            
    def parar_coleta(self):
        self.log_message("coleta", "Parando câmera...")
        self.is_collecting = False
        self.thread_active = False
        
        # Aguardar um pouco para thread finalizar
        time.sleep(0.5)
        
        if hasattr(self, 'camera') and self.camera is not None:
            self.camera.release()
            self.camera = None
            
        self.btn_iniciar_coleta.config(state="normal")
        self.btn_parar_coleta.config(state="disabled")
        self.btn_salvar_boa.config(state="disabled")
        self.btn_salvar_defeito.config(state="disabled")
        
        self.label_video_coleta.config(image="", text="Câmera parada")
        self.log_message("coleta", "✅ Câmera parada com sucesso")
        
    def capturar_frames_coleta(self):
        while self.is_collecting and self.thread_active:
            try:
                ret, frame = self.camera.read()
                if not ret:
                    break
                    
                # Processar frame (detectar círculos)
                frame_processado = self.processar_frame_coleta(frame)
                
                # Converter para formato Tkinter
                frame_rgb = cv2.cvtColor(frame_processado, cv2.COLOR_BGR2RGB)
                frame_pil = Image.fromarray(frame_rgb)
                frame_pil = frame_pil.resize((400, 300), Image.Resampling.LANCZOS)
                frame_tk = ImageTk.PhotoImage(frame_pil)
                
                # Atualizar GUI (thread-safe)
                self.root.after(0, self.atualizar_video_coleta, frame_tk)
                
                # Salvar último frame para captura
                self.ultimo_frame = frame
                
                time.sleep(0.03)  # ~30 FPS
                
            except Exception as e:
                print(f"Erro na captura: {e}")
                break
                
    def processar_frame_coleta(self, frame):
        frame_resultado = frame.copy()
        
        # Detectar círculos
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (9, 9), 2)
        
        circles = cv2.HoughCircles(
            blurred, cv2.HOUGH_GRADIENT, dp=1.2, minDist=100,
            param1=50, param2=30, minRadius=20, maxRadius=100
        )
        
        if circles is not None:
            circles = np.uint16(np.around(circles))
            for circle in circles[0, :1]:  # Apenas o primeiro círculo
                center = (circle[0], circle[1])
                radius = circle[2]
                cv2.circle(frame_resultado, center, radius, (0, 255, 0), 3)
                cv2.putText(frame_resultado, "Objeto detectado", (10, 30), 
                          cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            cv2.putText(frame_resultado, "Nenhum objeto detectado", (10, 30), 
                      cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
        return frame_resultado
        
    def atualizar_video_coleta(self, frame_tk):
        self.label_video_coleta.config(image=frame_tk, text="")
        self.label_video_coleta.image = frame_tk
        
    def salvar_boa(self):
        self.salvar_imagem("dataset_final/boas", "boa")
        
    def salvar_defeito(self):
        self.salvar_imagem("dataset_final/com_defeito", "defeito")
        
    def salvar_imagem(self, pasta, tipo):
        if not hasattr(self, 'ultimo_frame'):
            messagebox.showwarning("Aviso", "Nenhum frame disponível para salvar")
            return
            
        try:
            # Detectar e recortar região de interesse
            gray = cv2.cvtColor(self.ultimo_frame, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (9, 9), 2)
            
            circles = cv2.HoughCircles(
                blurred, cv2.HOUGH_GRADIENT, dp=1.2, minDist=100,
                param1=50, param2=30, minRadius=20, maxRadius=100
            )
            
            if circles is not None:
                circle = circles[0, 0]
                x, y, r = circle.astype(int)
                
                # Recortar ROI
                start_x = max(x - r, 0)
                end_x = min(x + r, self.ultimo_frame.shape[1])
                start_y = max(y - r, 0)
                end_y = min(y + r, self.ultimo_frame.shape[0])
                
                crop_roi = self.ultimo_frame[start_y:end_y, start_x:end_x]
                
                if crop_roi.size > 0:
                    timestamp = int(time.time() * 1000)
                    filename = f"{tipo}_{timestamp}.jpg"
                    filepath = os.path.join(pasta, filename)
                    
                    cv2.imwrite(filepath, crop_roi)
                    self.log_message("coleta", f"Imagem salva: {filename}")
                    self.atualizar_estatisticas_coleta()
                else:
                    messagebox.showwarning("Aviso", "ROI vazia, não foi possível salvar")
            else:
                messagebox.showwarning("Aviso", "Nenhum objeto detectado para salvar")
                
        except Exception as e:
            messagebox.showerror("Erro", f"Erro ao salvar imagem: {str(e)}")
            
    def atualizar_estatisticas_coleta(self):
        try:
            boas = len(os.listdir("dataset_final/boas"))
            defeitos = len(os.listdir("dataset_final/com_defeito"))
            
            self.label_boas.config(text=f"Boas: {boas}")
            self.label_defeitos.config(text=f"Defeitos: {defeitos}")
        except:
            pass
            
    # Métodos da aba Treinamento
    def verificar_dataset(self):
        try:
            boas = len([f for f in os.listdir("dataset_final/boas") if f.endswith('.jpg')])
            defeitos = len([f for f in os.listdir("dataset_final/com_defeito") if f.endswith('.jpg')])
            
            total = boas + defeitos
            
            message = f"""
Dataset Verificado:
• Peças Boas: {boas} imagens
• Peças com Defeito: {defeitos} imagens
• Total: {total} imagens

Recomendação: Mínimo 50 imagens por classe para bom treinamento.
"""
            self.log_message("treinamento", message)
            messagebox.showinfo("Dataset", message)
            
        except Exception as e:
            messagebox.showerror("Erro", f"Erro ao verificar dataset: {str(e)}")
            
    def iniciar_treinamento(self):
        try:
            epochs = int(self.spin_epochs.get())
            batch_size = int(self.spin_batch.get())
            model_name = self.entry_model_name.get()
            
            if not model_name.endswith('.h5'):
                model_name += '.h5'
                
            self.btn_iniciar_treinamento.config(state="disabled")
            self.btn_parar_treinamento.config(state="normal")
            self.progress_treinamento.start()
            
            # Iniciar treinamento em thread separada
            threading.Thread(target=self.treinar_modelo, 
                           args=(epochs, batch_size, model_name), daemon=True).start()
            
        except Exception as e:
            messagebox.showerror("Erro", f"Erro ao iniciar treinamento: {str(e)}")
            
    def treinar_modelo(self, epochs, batch_size, model_name):
        try:
            self.log_message("treinamento", "Iniciando treinamento...")
            
            # Configurar data generators
            datagen = ImageDataGenerator(
                rescale=1./255,
                validation_split=0.2
            )
            
            train_generator = datagen.flow_from_directory(
                'dataset_final',
                target_size=(self.IMG_HEIGHT, self.IMG_WIDTH),
                batch_size=batch_size,
                class_mode='binary',
                subset='training'
            )
            
            validation_generator = datagen.flow_from_directory(
                'dataset_final',
                target_size=(self.IMG_HEIGHT, self.IMG_WIDTH),
                batch_size=batch_size,
                class_mode='binary',
                subset='validation'
            )
            
            # Criar modelo
            model = models.Sequential([
                Input(shape=(self.IMG_HEIGHT, self.IMG_WIDTH, 3)),
                layers.Conv2D(32, (3, 3), activation='relu'),
                layers.MaxPooling2D((2, 2)),
                layers.Conv2D(64, (3, 3), activation='relu'),
                layers.MaxPooling2D((2, 2)),
                layers.Conv2D(64, (3, 3), activation='relu'),
                layers.Flatten(),
                layers.Dense(64, activation='relu'),
                layers.Dense(1, activation='sigmoid')
            ])
            
            model.compile(optimizer='adam',
                         loss='binary_crossentropy',
                         metrics=['accuracy'])
            
            self.log_message("treinamento", f"Modelo criado. Iniciando treinamento por {epochs} épocas...")
            
            # Treinar modelo
            history = model.fit(
                train_generator,
                validation_data=validation_generator,
                epochs=epochs,
                verbose=0
            )
            
            # Salvar modelo
            model.save(model_name)
            self.log_message("treinamento", f"Modelo salvo como: {model_name}")
            
            # Gerar gráficos
            self.root.after(0, self.gerar_graficos_treinamento, history)
            
            # Finalizar treinamento
            self.root.after(0, self.finalizar_treinamento, True)
            
        except Exception as e:
            self.log_message("treinamento", f"Erro no treinamento: {str(e)}")
            self.root.after(0, self.finalizar_treinamento, False)
            
    def gerar_graficos_treinamento(self, history):
        try:
            # Limpar frame anterior
            for widget in self.frame_graficos.winfo_children():
                widget.destroy()
                
            # Criar figura
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
            
            # Gráfico de acurácia
            ax1.plot(history.history['accuracy'], label='Treino')
            ax1.plot(history.history['val_accuracy'], label='Validação')
            ax1.set_title('Acurácia por Época')
            ax1.set_xlabel('Época')
            ax1.set_ylabel('Acurácia')
            ax1.legend()
            ax1.grid(True)
            
            # Gráfico de perda
            ax2.plot(history.history['loss'], label='Treino')
            ax2.plot(history.history['val_loss'], label='Validação')
            ax2.set_title('Perda por Época')
            ax2.set_xlabel('Época')
            ax2.set_ylabel('Perda')
            ax2.legend()
            ax2.grid(True)
            
            plt.tight_layout()
            
            # Adicionar à interface
            canvas = FigureCanvasTkAgg(fig, self.frame_graficos)
            canvas.draw()
            canvas.get_tk_widget().pack(fill="both", expand=True)
            
        except Exception as e:
            self.log_message("treinamento", f"Erro ao gerar gráficos: {str(e)}")
            
    def finalizar_treinamento(self, sucesso):
        self.btn_iniciar_treinamento.config(state="normal")
        self.btn_parar_treinamento.config(state="disabled")
        self.progress_treinamento.stop()
        
        if sucesso:
            self.log_message("treinamento", "Treinamento concluído com sucesso!")
            self.atualizar_lista_modelos()
            messagebox.showinfo("Sucesso", "Treinamento concluído com sucesso!")
        else:
            messagebox.showerror("Erro", "Falha no treinamento. Verifique o log.")
            
    def parar_treinamento(self):
        # Implementar parada do treinamento (mais complexo)
        messagebox.showinfo("Info", "Para parar o treinamento, feche e reinicie o programa.")
        
    # Métodos da aba Inspeção
    def atualizar_lista_modelos(self):
        modelos = [f for f in os.listdir('.') if f.endswith('.h5')]
        self.combo_modelo['values'] = modelos
        if modelos and not self.combo_modelo.get():
            self.combo_modelo.set(modelos[0])
            
    def carregar_modelo(self):
        try:
            modelo_selecionado = self.combo_modelo.get()
            if not modelo_selecionado:
                messagebox.showwarning("Aviso", "Selecione um modelo primeiro")
                return
                
            self.model = load_model(modelo_selecionado)
            self.current_model_path = modelo_selecionado
            
            # Verificar as dimensões esperadas pelo modelo
            input_shape = self.model.input_shape
            self.log_message("inspecao", f"Modelo carregado: {modelo_selecionado}")
            self.log_message("inspecao", f"Shape de entrada esperado: {input_shape}")
            
            # Atualizar dimensões se necessário
            if len(input_shape) == 4:  # (batch, height, width, channels)
                expected_height = input_shape[1]
                expected_width = input_shape[2]
                
                if expected_height and expected_width:
                    self.IMG_HEIGHT = expected_height
                    self.IMG_WIDTH = expected_width
                    self.log_message("inspecao", f"Dimensões ajustadas para: {self.IMG_WIDTH}x{self.IMG_HEIGHT}")
                    
                    # Atualizar as configurações na interface
                    self.spin_height.delete(0, 'end')
                    self.spin_height.insert(0, str(self.IMG_HEIGHT))
                    self.spin_width.delete(0, 'end')
                    self.spin_width.insert(0, str(self.IMG_WIDTH))
            
            self.btn_iniciar_inspecao.config(state="normal")
            self.btn_testar_imagem.config(state="normal")
            
            messagebox.showinfo("Sucesso", f"Modelo {modelo_selecionado} carregado com sucesso!\nDimensões: {self.IMG_WIDTH}x{self.IMG_HEIGHT}")
            
        except Exception as e:
            self.log_message("inspecao", f"Erro ao carregar modelo: {str(e)}")
            messagebox.showerror("Erro", f"Erro ao carregar modelo: {str(e)}")
            
    def iniciar_inspecao(self):
        try:
            if not self.model:
                messagebox.showwarning("Aviso", "Carregue um modelo primeiro")
                return
                
            # Parar qualquer câmera anterior
            if hasattr(self, 'camera') and self.camera is not None:
                self.camera.release()
                time.sleep(0.5)  # Aguardar liberação
                
            self.CAMERA_INDEX = int(self.spin_camera.get())
            self.log_message("inspecao", f"Tentando abrir câmera no índice {self.CAMERA_INDEX}...")
            
            self.camera = cv2.VideoCapture(self.CAMERA_INDEX)
            
            # Tentar várias vezes abrir a câmera
            tentativas = 3
            for i in range(tentativas):
                if self.camera.isOpened():
                    break
                self.camera.release()
                time.sleep(0.5)
                self.camera = cv2.VideoCapture(self.CAMERA_INDEX)
                self.log_message("inspecao", f"Tentativa {i+1} de {tentativas} para abrir câmera...")
            
            if not self.camera.isOpened():
                self.log_message("inspecao", f"ERRO: Não foi possível abrir a câmera {self.CAMERA_INDEX}")
                messagebox.showerror("Erro", f"Não foi possível abrir a câmera {self.CAMERA_INDEX}\n\nVerifique se:\n- A câmera está conectada\n- Nenhum outro programa está usando a câmera\n- O índice da câmera está correto nas configurações")
                return
                
            # Configurar propriedades da câmera
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.camera.set(cv2.CAP_PROP_FPS, 30)
            
            # Testar se consegue capturar um frame
            ret, test_frame = self.camera.read()
            if not ret or test_frame is None:
                self.log_message("inspecao", "ERRO: Não foi possível capturar frame da câmera")
                self.camera.release()
                messagebox.showerror("Erro", "A câmera foi aberta mas não consegue capturar imagens")
                return
                
            self.is_inspecting = True
            self.thread_active = True
            self.btn_iniciar_inspecao.config(state="disabled")
            self.btn_parar_inspecao.config(state="normal")
            
            # Resetar estatísticas
            self.total_inspecoes = 0
            self.pecas_boas = 0
            self.pecas_defeitos = 0
            
            # Resetar sistema de debounce
            self.ultima_classificacao = None
            self.frames_mesma_classificacao = 0
            self.classificacao_ja_contada = False
            
            self.atualizar_estatisticas_inspecao()
            
            self.log_message("inspecao", "✅ Inspeção iniciada com sucesso!")
            
            # Iniciar thread de inspeção
            threading.Thread(target=self.capturar_frames_inspecao, daemon=True).start()
            
        except Exception as e:
            self.log_message("inspecao", f"ERRO: {str(e)}")
            messagebox.showerror("Erro", f"Erro ao iniciar inspeção: {str(e)}")
            
    def parar_inspecao(self):
        self.log_message("inspecao", "Parando inspeção...")
        self.is_inspecting = False
        self.thread_active = False
        
        # Aguardar um pouco para thread finalizar
        time.sleep(0.5)
        
        if hasattr(self, 'camera') and self.camera is not None:
            self.camera.release()
            self.camera = None
            
        self.btn_iniciar_inspecao.config(state="normal")
        self.btn_parar_inspecao.config(state="disabled")
        
        self.label_video_inspecao.config(image="", text="Inspeção parada")
        self.label_resultado.config(text="")
        self.log_message("inspecao", "✅ Inspeção parada com sucesso")
        
    def capturar_frames_inspecao(self):
        self.log_message("inspecao", "Thread de captura iniciada")
        frame_count = 0
        
        while self.is_inspecting and self.thread_active:
            try:
                if not self.camera or not self.camera.isOpened():
                    self.log_message("inspecao", "ERRO: Câmera não disponível")
                    break
                    
                ret, frame = self.camera.read()
                if not ret or frame is None:
                    self.log_message("inspecao", "ERRO: Falha ao capturar frame")
                    time.sleep(0.1)
                    continue
                    
                frame_count += 1
                if frame_count % 30 == 0:  # Log a cada 30 frames (~1 segundo)
                    self.log_message("inspecao", f"Frames processados: {frame_count}")
                    
                # Processar frame (detectar e classificar)
                frame_processado, resultado = self.processar_frame_inspecao(frame)
                
                # Converter para formato Tkinter
                try:
                    frame_rgb = cv2.cvtColor(frame_processado, cv2.COLOR_BGR2RGB)
                    frame_pil = Image.fromarray(frame_rgb)
                    frame_pil = frame_pil.resize((400, 300), Image.Resampling.LANCZOS)
                    frame_tk = ImageTk.PhotoImage(frame_pil)
                    
                    # Atualizar GUI (thread-safe)
                    self.root.after(0, self.atualizar_video_inspecao, frame_tk, resultado)
                    
                except Exception as img_error:
                    self.log_message("inspecao", f"Erro ao processar imagem: {img_error}")
                    continue
                
                time.sleep(0.03)  # ~30 FPS
                
            except Exception as e:
                self.log_message("inspecao", f"Erro na captura: {e}")
                time.sleep(0.1)
                continue
                
        self.log_message("inspecao", "Thread de captura finalizada")
                
    def processar_frame_inspecao(self, frame):
        frame_resultado = frame.copy()
        resultado = None
        
        # Detectar círculos
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (9, 9), 2)
        
        circles = cv2.HoughCircles(
            blurred, cv2.HOUGH_GRADIENT, dp=1.2, minDist=100,
            param1=50, param2=30, minRadius=20, maxRadius=100
        )
        
        if circles is not None:
            circle = circles[0, 0]
            center = (int(circle[0]), int(circle[1]))
            radius = int(circle[2])
            
            # Recortar ROI
            x, y, r = circle.astype(int)
            start_x = max(x - r, 0)
            end_x = min(x + r, frame.shape[1])
            start_y = max(y - r, 0)
            end_y = min(y + r, frame.shape[0])
            
            crop_roi = frame[start_y:end_y, start_x:end_x]
            
            if crop_roi.size > 0:
                try:
                    # Pré-processar para o modelo - GARANTIR TAMANHO CORRETO
                    roi_rgb = cv2.cvtColor(crop_roi, cv2.COLOR_BGR2RGB)
                    
                    # IMPORTANTE: Usar exatamente o mesmo tamanho do treinamento
                    img_resized = cv2.resize(roi_rgb, (self.IMG_WIDTH, self.IMG_HEIGHT), interpolation=cv2.INTER_AREA)
                    
                    # Verificar se o tamanho está correto
                    if img_resized.shape != (self.IMG_HEIGHT, self.IMG_WIDTH, 3):
                        self.log_message("inspecao", f"ERRO: Tamanho incorreto {img_resized.shape}, esperado ({self.IMG_HEIGHT}, {self.IMG_WIDTH}, 3)")
                        return frame_resultado, None
                    
                    img_array = np.expand_dims(img_resized, axis=0)
                    img_array = img_array.astype(np.float32) / 255.0
                    
                    # Verificar shape final antes da predição
                    expected_shape = (1, self.IMG_HEIGHT, self.IMG_WIDTH, 3)
                    if img_array.shape != expected_shape:
                        self.log_message("inspecao", f"ERRO: Shape incorreto {img_array.shape}, esperado {expected_shape}")
                        return frame_resultado, None
                    
                    # Fazer predição
                    prediction = self.model.predict(img_array, verbose=0)[0][0]
                    
                    # Determinar classificação
                    if prediction < 0.5:
                        status_atual = "BOA"
                        cor = (0, 255, 0)  # Verde
                    else:
                        status_atual = "DEFEITO"
                        cor = (0, 0, 255)  # Vermelho
                    
                    # Sistema de debounce para contagem
                    if self.ultima_classificacao == status_atual:
                        self.frames_mesma_classificacao += 1
                        
                        # Se ficou tempo suficiente na mesma classificação e ainda não contou
                        if (self.frames_mesma_classificacao >= self.FRAMES_NECESSARIOS and 
                            not self.classificacao_ja_contada):
                            
                            # Contar apenas uma vez
                            if status_atual == "BOA":
                                self.pecas_boas += 1
                            else:
                                self.pecas_defeitos += 1
                            
                            self.total_inspecoes += 1
                            self.classificacao_ja_contada = True
                            
                            self.log_message("inspecao", f"✅ Objeto classificado: {status_atual} (Confiança: {prediction:.2f})")
                            
                    else:
                        # Classificação mudou, resetar contadores
                        self.ultima_classificacao = status_atual
                        self.frames_mesma_classificacao = 1
                        self.classificacao_ja_contada = False
                    
                    # Desenhar resultado com indicador de estabilidade
                    cv2.circle(frame_resultado, center, radius, cor, 3)
                    cv2.putText(frame_resultado, f"Status: {status_atual}", (10, 30), 
                              cv2.FONT_HERSHEY_SIMPLEX, 1, cor, 2)
                    cv2.putText(frame_resultado, f"Confianca: {prediction:.2f}", (10, 70), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, cor, 2)
                    
                    # Indicador de estabilidade
                    frames_restantes = max(0, self.FRAMES_NECESSARIOS - self.frames_mesma_classificacao)
                    if frames_restantes > 0:
                        cv2.putText(frame_resultado, f"Estabilizando... {frames_restantes}", (10, 110), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                    elif self.classificacao_ja_contada:
                        cv2.putText(frame_resultado, "CONTADO!", (10, 110), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                    
                    resultado = f"{status_atual} (Confiança: {prediction:.2f})"
                    
                except Exception as pred_error:
                    self.log_message("inspecao", f"Erro na predição: {pred_error}")
                    cv2.putText(frame_resultado, "Erro na classificacao", (10, 30), 
                              cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                
        else:
            # Não há objeto detectado - resetar sistema de debounce
            self.ultima_classificacao = None
            self.frames_mesma_classificacao = 0
            self.classificacao_ja_contada = False
            
            cv2.putText(frame_resultado, "Nenhum objeto detectado", (10, 30), 
                      cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
            
        return frame_resultado, resultado
        
    def atualizar_video_inspecao(self, frame_tk, resultado):
        self.label_video_inspecao.config(image=frame_tk, text="")
        self.label_video_inspecao.image = frame_tk
        
        if resultado:
            if "BOA" in resultado:
                self.label_resultado.config(text=resultado, foreground="green")
            else:
                self.label_resultado.config(text=resultado, foreground="red")
            self.atualizar_estatisticas_inspecao()
            
    def atualizar_estatisticas_inspecao(self):
        self.label_total_inspecoes.config(text=f"Total: {self.total_inspecoes}")
        self.label_pecas_boas.config(text=f"Boas: {self.pecas_boas}")
        self.label_pecas_defeitos.config(text=f"Defeitos: {self.pecas_defeitos}")
        
    def testar_imagem(self):
        try:
            if not self.model:
                messagebox.showwarning("Aviso", "Carregue um modelo primeiro")
                return
                
            filepath = filedialog.askopenfilename(
                title="Selecionar Imagem",
                filetypes=[("Imagens", "*.jpg *.jpeg *.png *.bmp")]
            )
            
            if not filepath:
                return
                
            # Carregar e processar imagem
            img = cv2.imread(filepath)
            if img is None:
                messagebox.showerror("Erro", "Não foi possível carregar a imagem")
                return
                
            # Pré-processar imagem com verificações
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_resized = cv2.resize(img_rgb, (self.IMG_WIDTH, self.IMG_HEIGHT), interpolation=cv2.INTER_AREA)
            
            # Verificar se o tamanho está correto
            if img_resized.shape != (self.IMG_HEIGHT, self.IMG_WIDTH, 3):
                messagebox.showerror("Erro", f"Erro no redimensionamento. Shape: {img_resized.shape}, esperado: ({self.IMG_HEIGHT}, {self.IMG_WIDTH}, 3)")
                return
            
            img_array = np.expand_dims(img_resized, axis=0)
            img_array = img_array.astype(np.float32) / 255.0
            
            # Verificar shape final
            expected_shape = (1, self.IMG_HEIGHT, self.IMG_WIDTH, 3)
            if img_array.shape != expected_shape:
                messagebox.showerror("Erro", f"Shape incorreto: {img_array.shape}, esperado: {expected_shape}")
                return
            
            # Fazer predição
            prediction = self.model.predict(img_array)[0][0]
            
            if prediction < 0.5:
                status = "BOA"
                cor = "green"
            else:
                status = "COM DEFEITO"
                cor = "red"
                
            resultado = f"Resultado: {status}\nConfiança: {prediction:.4f}"
            
            self.log_message("inspecao", f"Teste de imagem: {os.path.basename(filepath)} - {resultado}")
            
            # Mostrar resultado
            msg = tk.Toplevel(self.root)
            msg.title("Resultado do Teste")
            msg.geometry("400x300")
            
            # Mostrar imagem redimensionada
            img_show = Image.fromarray(img_rgb)
            img_show = img_show.resize((200, 200), Image.Resampling.LANCZOS)
            img_tk = ImageTk.PhotoImage(img_show)
            
            tk.Label(msg, image=img_tk).pack(pady=10)
            tk.Label(msg, text=resultado, font=("Arial", 12, "bold"), 
                    foreground=cor).pack(pady=10)
            
            # Manter referência da imagem
            msg.image = img_tk
            
        except Exception as e:
            messagebox.showerror("Erro", f"Erro ao testar imagem: {str(e)}")
            
    # Métodos da aba Configurações
    def aplicar_configuracoes(self):
        try:
            self.CAMERA_INDEX = int(self.spin_camera.get())
            self.IMG_HEIGHT = int(self.spin_height.get())
            self.IMG_WIDTH = int(self.spin_width.get())
            
            # Atualizar tempo de contagem
            tempo_segundos = float(self.spin_tempo_contagem.get())
            self.FRAMES_NECESSARIOS = int(tempo_segundos * 30)  # 30 FPS
            
            self.log_message("inspecao", f"Configurações aplicadas: Câmera={self.CAMERA_INDEX}, Dimensões={self.IMG_WIDTH}x{self.IMG_HEIGHT}, Tempo contagem={tempo_segundos}s")
            messagebox.showinfo("Sucesso", f"Configurações aplicadas!\n\nCâmera: {self.CAMERA_INDEX}\nDimensões: {self.IMG_WIDTH}x{self.IMG_HEIGHT}\nTempo para contagem: {tempo_segundos}s")
            
        except Exception as e:
            messagebox.showerror("Erro", f"Erro ao aplicar configurações: {str(e)}")
            
    def salvar_configuracoes(self):
        try:
            config = {
                'camera_index': int(self.spin_camera.get()),
                'img_height': int(self.spin_height.get()),
                'img_width': int(self.spin_width.get())
            }
            
            import json
            with open('config.json', 'w') as f:
                json.dump(config, f, indent=2)
                
            messagebox.showinfo("Sucesso", "Configurações salvas em config.json")
            
        except Exception as e:
            messagebox.showerror("Erro", f"Erro ao salvar configurações: {str(e)}")
            
    def abrir_pasta_projeto(self):
        try:
            if sys.platform.startswith('win'):
                os.startfile('.')
            elif sys.platform.startswith('darwin'):
                subprocess.run(['open', '.'])
            else:
                subprocess.run(['xdg-open', '.'])
        except Exception as e:
            messagebox.showerror("Erro", f"Erro ao abrir pasta: {str(e)}")
            
    # Método auxiliar para logs
    def log_message(self, aba, message):
        timestamp = time.strftime("%H:%M:%S")
        log_text = f"[{timestamp}] {message}\n"
        
        if aba == "coleta":
            self.log_coleta.insert(tk.END, log_text)
            self.log_coleta.see(tk.END)
        elif aba == "treinamento":
            self.log_treinamento.insert(tk.END, log_text)
            self.log_treinamento.see(tk.END)
        elif aba == "inspecao":
            self.log_inspecao.insert(tk.END, log_text)
            self.log_inspecao.see(tk.END)
            
    def __del__(self):
        # Limpar recursos ao fechar
        if hasattr(self, 'camera') and self.camera:
            self.camera.release()

if __name__ == "__main__":
    root = tk.Tk()
    app = InterfaceRenault(root)
    
    # Configurar fechamento da aplicação
    def on_closing():
        app.thread_active = False
        if hasattr(app, 'camera') and app.camera:
            app.camera.release()
        cv2.destroyAllWindows()
        root.destroy()
        
    root.protocol("WM_DELETE_WINDOW", on_closing)
    root.mainloop()